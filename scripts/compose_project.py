"""Docker Compose helpers for the scripts that manage an isolated stack.

Two CLIs run a compose project of their own beside the dev stack (project
`doc-pilot`): scripts/demos/demo.py (`doc-pilot-demo`) and scripts/load/load.py
(`doc-pilot-load`). Each builds one ComposeProject with its project name as a
constant, and every docker call goes through it, so neither CLI can address the
dev stack or its volumes (the dev database volume was wiped once by a careless
`down -v`). The volume guard is written once, here: `down --volumes` and a
reset may only remove volumes this project declares, named `<project>_*` and
labelled with the project by Docker.

Stdlib only: the CLIs run with whichever Python is at hand.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_LABEL = "com.docker.compose.project"


class CliError(Exception):
    """A failure to report on stderr and turn into a non-zero exit code."""


def compose_env() -> dict[str, str]:
    """The environment compose runs with: the caller's, minus
    COMPOSE_PROFILES. A profile exported in the shell (e.g. `tracing`)
    would start the project's own jaeger, whose published ports could
    collide with another stack's jaeger. Profiles a CLI needs are passed as
    --profile flags instead."""
    return {k: v for k, v in os.environ.items() if k != "COMPOSE_PROFILES"}


def declared_volume_names(config: dict) -> list[str]:
    """Names of the volumes `down --volumes` deletes: compose removes every
    non-external volume in the resolved config by its resolved name, with no
    check of the volume's labels."""
    return [
        volume["name"]
        for volume in config.get("volumes", {}).values()
        if not volume.get("external")
    ]


def parse_ps_records(output: str) -> list[dict]:
    """The container records of `compose ps --format json`: one JSON object
    per line, or (older versions) one array."""
    output = output.strip()
    if not output:
        return []
    if output.startswith("["):
        return json.loads(output)
    return [json.loads(line) for line in output.splitlines()]


def check_one_shots_and_worker(states: dict[str, dict]) -> None:
    """`up --wait` passes for a one-shot that exited, whatever its exit code,
    so check the migrate one-shot here, and that the worker's heartbeat
    healthcheck reports healthy."""
    migrate = states.get("migrate", {})
    if migrate.get("State") != "exited" or migrate.get("ExitCode") != 0:
        raise CliError(
            f"migrate did not complete: state {migrate.get('State')!r}, "
            f"exit code {migrate.get('ExitCode')!r}"
        )
    worker = states.get("worker", {})
    if worker.get("State") != "running" or worker.get("Health") != "healthy":
        raise CliError(
            f"worker is not healthy: state {worker.get('State')!r}, "
            f"health {worker.get('Health')!r}"
        )


@dataclass(frozen=True)
class ComposeProject:
    """One compose project: its name and the compose files that define it."""

    name: str
    files: tuple[str, ...]

    def compose_args(self, *args: str) -> list[str]:
        """The full `docker compose` command line for this project."""
        cmd = ["docker", "compose", "-p", self.name]
        for compose_file in self.files:
            cmd += ["-f", compose_file]
        return [*cmd, *args]

    def run(
        self, cmd: list[str], *, capture: bool = False, stdin: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Run a docker command from the repo root (the compose files and the
        build contexts are relative to it), turning a missing docker into a
        CliError instead of a traceback. `stdin` is text for the command's
        standard input (a password), never an argument."""
        try:
            return subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=compose_env(),
                input=stdin,
                capture_output=capture,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise CliError("docker was not found on PATH") from exc

    def compose(self, *args: str) -> int:
        """Run `docker compose` for this project, output to the terminal."""
        return self.run(self.compose_args(*args)).returncode

    def docker_output(self, cmd: list[str], *, stdin: str | None = None) -> str:
        """stdout of a docker command; CliError with its stderr if it fails
        (typically Docker Desktop not running). Only the command's first three
        words are named in the error: the rest could hold anything."""
        result = self.run(cmd, capture=True, stdin=stdin)
        if result.returncode != 0:
            detail = result.stderr.strip() or f"exit code {result.returncode}"
            raise CliError(f"`{' '.join(cmd[:3])} ...` failed: {detail}")
        return result.stdout

    def resolved_config(self, *, all_profiles: bool = False) -> dict:
        """The project's fully merged compose config. It holds every
        interpolated environment value, a key included: callers read it, they
        never print it. `all_profiles` includes the services behind a profile,
        for a check that must cover every service the project could start."""
        profiles = ["--profile", "*"] if all_profiles else []
        return json.loads(
            self.docker_output(self.compose_args(*profiles, "config", "--format", "json"))
        )

    def existing_volume_projects(self) -> dict[str, str]:
        """Every volume on this Docker host mapped to its compose project label
        ("" when it has none)."""
        out = self.docker_output(
            [
                "docker",
                "volume",
                "ls",
                "--format",
                f'{{{{.Name}}}}\t{{{{.Label "{PROJECT_LABEL}"}}}}',
            ]
        )
        projects = {}
        for line in out.splitlines():
            name, _, project = line.partition("\t")
            projects[name] = project
        return projects

    def volume_problems(self, config: dict, existing: dict[str, str]) -> list[str]:
        """Reasons `down --volumes` must not run; empty when it is safe.

        config is `compose config --format json` and existing maps every volume
        on the host to its compose project label. Two independent views of the
        same set must agree: what compose would delete (config) and what Docker
        labels as this project's (existing)."""
        problems = []
        if config.get("name") != self.name:
            problems.append(f"compose project is {config.get('name')!r}, not {self.name!r}")
        declared = declared_volume_names(config)
        for name in declared:
            if not name.startswith(f"{self.name}_"):
                problems.append(f"{name}: would be removed but is not named {self.name}_*")
            elif name in existing and existing[name] != self.name:
                problems.append(f"{name}: exists with project label {existing[name]!r}")
        for name, project in existing.items():
            if project == self.name and name not in declared:
                problems.append(
                    f"{name}: labelled {self.name} but not declared in the config"
                )
        return problems

    def report_volumes(self, config: dict, existing: dict[str, str]) -> list[str]:
        """Print the volumes `down --volumes` would remove, with whether each
        exists, then the reasons it must not run (volume_problems) on stderr.
        Returns those reasons: a caller removes nothing unless it is empty."""
        declared = declared_volume_names(config)
        print(f"Volumes that `down --volumes` would remove (project {self.name}):")
        for name in declared:
            print(f"  {name}  ({'exists' if name in existing else 'not created yet'})")
        if not declared:
            print("  (none)")
        problems = self.volume_problems(config, existing)
        if problems:
            print("Refusing to remove volumes:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
        return problems

    def container_records(self, *, include_stopped: bool = True) -> list[dict]:
        """Every container of this project as `compose ps` reports it, one
        record per container (a scaled service has several)."""
        args = ["ps", "--format", "json"]
        if include_stopped:
            args.insert(1, "--all")
        return parse_ps_records(self.docker_output(self.compose_args(*args)))

    def service_states(self) -> dict[str, dict]:
        """`compose ps --all` as {service: container record}; a scaled service
        is represented by its last record."""
        return {record["Service"]: record for record in self.container_records()}
