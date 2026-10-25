"""Tests for the demo CLI's safety guards (run: python -m pytest ../scripts/demos -q)."""

from __future__ import annotations

import argparse
import ast
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import demo
import encode
import pytest


def config(volumes: dict, name: str = "doc-pilot-demo") -> dict:
    return {"name": name, "volumes": volumes}


DEMO_VOLUMES = {
    "db_data": {"name": "doc-pilot-demo_db_data"},
    "uploads": {"name": "doc-pilot-demo_uploads"},
}


def test_all_demo_names_pass() -> None:
    existing = {
        "doc-pilot-demo_db_data": "doc-pilot-demo",
        "doc-pilot-demo_uploads": "doc-pilot-demo",
        "doc-pilot_db_data": "doc-pilot",
    }
    assert demo.volume_problems(config(DEMO_VOLUMES), existing) == []


def test_volumes_not_created_yet_pass() -> None:
    assert demo.volume_problems(config(DEMO_VOLUMES), {}) == []


def test_dev_volume_name_refuses() -> None:
    volumes = {**DEMO_VOLUMES, "db_data": {"name": "doc-pilot_db_data"}}
    existing = {"doc-pilot_db_data": "doc-pilot"}
    problems = demo.volume_problems(config(volumes), existing)
    assert any("doc-pilot_db_data" in p for p in problems)


def test_external_volumes_are_ignored() -> None:
    volumes = {
        **DEMO_VOLUMES,
        "shared": {"name": "doc-pilot_db_data", "external": True},
    }
    assert demo.declared_volume_names(config(volumes)) == [
        "doc-pilot-demo_db_data",
        "doc-pilot-demo_uploads",
    ]
    assert demo.volume_problems(config(volumes), {}) == []


def test_declared_volume_with_foreign_label_refuses() -> None:
    existing = {"doc-pilot-demo_db_data": "doc-pilot"}
    assert demo.volume_problems(config(DEMO_VOLUMES), existing)


def test_labelled_but_undeclared_volume_refuses() -> None:
    existing = {"doc-pilot-demo_extra": "doc-pilot-demo"}
    assert demo.volume_problems(config(DEMO_VOLUMES), existing)


def test_wrong_project_name_refuses() -> None:
    assert demo.volume_problems(config(DEMO_VOLUMES, name="doc-pilot"), {})


def test_compose_env_drops_compose_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPOSE_PROFILES", "tracing")
    assert "COMPOSE_PROFILES" not in demo.compose_env()


def test_worker_or_migrate_failure_is_reported() -> None:
    ok = {
        "migrate": {"State": "exited", "ExitCode": 0},
        "worker": {"State": "running", "Health": "healthy"},
    }
    demo.check_one_shots_and_worker(ok)
    with pytest.raises(demo.CliError, match="worker"):
        demo.check_one_shots_and_worker({**ok, "worker": {"State": "exited"}})
    with pytest.raises(demo.CliError, match="worker"):
        demo.check_one_shots_and_worker(
            {**ok, "worker": {"State": "running", "Health": "unhealthy"}}
        )
    with pytest.raises(demo.CliError, match="migrate"):
        demo.check_one_shots_and_worker(
            {**ok, "migrate": {"State": "exited", "ExitCode": 1}}
        )


def test_seed_runs_the_one_shot_in_the_seed_profile() -> None:
    command = demo.seed_command(reset=False)
    assert command[:2] == ["--profile", "seed"]
    assert command[2:4] == ["run", "--rm"]
    assert command[-1] == "seed"
    assert "--reset" not in command


def test_seed_reset_passes_reset_to_the_script() -> None:
    assert demo.seed_command(reset=True)[-3:] == [
        "python",
        "scripts/seed_demo.py",
        "--reset",
    ]


def test_seed_parser_reads_reset() -> None:
    parser = demo.build_parser()
    assert parser.parse_args(["seed"]).reset is False
    assert parser.parse_args(["seed", "--reset"]).reset is True


def test_record_all_skips_the_smoke_scenario(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(demo, "SCENARIOS", {"smoke": None, "search-hybrid": None})
    assert demo.resolve_slugs(["all"]) == ["search-hybrid"]


def test_record_all_puts_the_dashboard_after_the_live_takes() -> None:
    # The dashboard shows today's real runs, which the live takes create.
    slugs = demo.resolve_slugs(["all"])
    assert slugs.index("monitoring-dashboard") > slugs.index("document-chat")
    assert "monitoring-dashboard" not in demo.LIVE_SLUGS


def test_record_all_with_only_smoke_registered_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(demo, "SCENARIOS", {"smoke": None})
    with pytest.raises(demo.CliError, match="no feature scenarios"):
        demo.resolve_slugs(["all"])


def test_record_refuses_an_unknown_slug() -> None:
    with pytest.raises(demo.CliError, match="unknown scenario nope"):
        demo.resolve_slugs(["smoke", "nope"])
    assert demo.resolve_slugs(["smoke"]) == ["smoke"]


def test_smoke_never_reaches_the_committed_gifs() -> None:
    assert demo.gif_destination("smoke") is None
    assert demo.gif_destination("search-hybrid") == (
        demo.REPO_ROOT / "screenshots" / "features" / "search-hybrid.gif"
    )


def test_a_ready_stack_has_no_problems() -> None:
    assert demo.readiness_problems(True, True, 25) == []


def test_a_down_or_empty_stack_says_how_to_fix_it() -> None:
    problems = demo.readiness_problems(False, False, 0)
    assert len(problems) == 2
    assert all("demo.py stack up" in p for p in problems)
    (empty,) = demo.readiness_problems(True, True, 0)
    assert "demo.py seed --reset" in empty


def test_record_parser_takes_slugs() -> None:
    parser = demo.build_parser()
    assert parser.parse_args(["record", "smoke", "x"]).slugs == ["smoke", "x"]
    assert parser.parse_args(["record", "all"]).slugs == ["all"]


def feature_gifs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, infos: dict[str, encode.MediaInfo]
) -> None:
    for name in infos:
        (tmp_path / name).write_bytes(b"gif")
    monkeypatch.setattr(demo, "FEATURES_DIR", tmp_path)
    monkeypatch.setattr(encode, "require_tools", lambda: None)
    monkeypatch.setattr(encode, "probe", lambda path: infos[path.name])


def test_check_passes_gifs_within_the_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_gifs(tmp_path, monkeypatch, {"a.gif": encode.MediaInfo(9.0, 960, 800_000)})
    assert demo.check(argparse.Namespace()) == 0


def test_check_fails_when_any_gif_is_over_a_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    feature_gifs(
        tmp_path,
        monkeypatch,
        {
            "ok.gif": encode.MediaInfo(9.0, 960, 800_000),
            "big.gif": encode.MediaInfo(16.0, 960, 1_200_000),
        },
    )
    assert demo.check(argparse.Namespace()) == 1
    err = capsys.readouterr().err
    assert "big.gif" in err and "ok.gif" not in err


def test_check_with_no_gifs_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_gifs(tmp_path, monkeypatch, {})
    assert demo.check(argparse.Namespace()) == 0


def test_a_keyed_capped_budget_with_room_has_no_live_problems() -> None:
    assert demo.live_problems(True, 1.5, 0.0) == []
    assert demo.live_problems(True, 1.0, 0.9) == []


def test_a_live_take_needs_the_key() -> None:
    (problem,) = demo.live_problems(False, 1.5, 0.0)
    assert "ANTHROPIC_API_KEY" in problem


def test_a_live_take_refuses_a_budget_over_the_cap_or_off() -> None:
    (raised,) = demo.live_problems(True, 5.0, 0.0)
    assert "$5.00" in raised
    (off,) = demo.live_problems(True, None, 0.0)
    assert "off" in off


def test_a_live_take_refuses_once_the_budget_is_spent() -> None:
    (problem,) = demo.live_problems(True, 1.5, 1.5)
    assert "429" in problem


def test_only_the_model_scenarios_are_live() -> None:
    assert demo.LIVE_SLUGS == {"ask-agent", "document-chat"}
    assert demo.LIVE_SLUGS <= set(demo.SCENARIOS)


# --- the demo sign-in ---------------------------------------------------------

COOKIE_NAME = "docpilot_demo_session"
SESSION_TOKEN = "session-token-1"


class FakeDemoApi:
    """A local stand-in for the demo API's sign-in, documents, and stats: it
    accepts the demo user from the frontend's origin only, sets the session
    cookie, and answers 401 to everything else without that cookie."""

    def __init__(self) -> None:
        self.origin = ""  # the frontend's origin: this server's own address, set below
        self.logins = 0
        # Captured now, so a test that changes demo.DEMO_USER_PASSWORD is refused.
        self.credentials = {
            "email": demo.DEMO_USER_EMAIL,
            "password": demo.DEMO_USER_PASSWORD,
        }
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def send(self, status: int, body: object = None, headers: dict | None = None) -> None:
                payload = json.dumps(body).encode() if body is not None else b""
                self.send_response(status)
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self) -> None:
                api.logins += 1
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path != "/auth/login" or self.headers["Origin"] != api.origin:
                    return self.send(403, {"detail": "origin not allowed"})
                if body != api.credentials:
                    return self.send(401, {"detail": "bad credentials"})
                cookie = f"{COOKIE_NAME}={SESSION_TOKEN}; HttpOnly; Max-Age=43200; Path=/; SameSite=lax"
                self.send(200, {"email": body["email"]}, {"Set-Cookie": cookie})

            def do_GET(self) -> None:
                if self.path == "/healthz" or self.path == "/":
                    return self.send(200, {"status": "ok"})
                if f"{COOKIE_NAME}={SESSION_TOKEN}" not in (self.headers["Cookie"] or ""):
                    return self.send(401, {"detail": "not signed in"})
                if self.path == "/documents":
                    return self.send(200, [{"id": "a"}, {"id": "b"}])
                if self.path == "/stats":
                    return self.send(
                        200, {"budget": {"daily_budget_usd": 1.5, "spent_today_usd": 0.25}}
                    )
                self.send(404, {"detail": "not found"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = self.origin = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def demo_api(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeDemoApi]:
    """The fake API standing in for both demo URLs, with no cached opener."""
    api = FakeDemoApi()
    monkeypatch.setattr(demo, "API_URL", api.url)
    monkeypatch.setattr(demo, "FRONTEND_URL", api.origin)
    monkeypatch.setattr(demo, "_opener", None)
    yield api
    api.close()


def test_sign_in_sends_the_demo_credentials_from_the_frontends_origin(
    demo_api: FakeDemoApi,
) -> None:
    opener = demo.sign_in_opener()
    # The jar holds the cookie, so a read that needs the session now works.
    assert json.loads(demo.http_get(f"{demo_api.url}/documents", opener)) == [
        {"id": "a"},
        {"id": "b"},
    ]
    assert demo.http_get(f"{demo_api.url}/documents") is None  # no cookie


def test_a_sign_in_the_api_refuses_says_how_to_fix_it(
    demo_api: FakeDemoApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(demo, "DEMO_USER_PASSWORD", "not the password")
    with pytest.raises(demo.CliError, match=r"HTTP 401.*demo\.py seed --reset"):
        demo.sign_in_opener()
    monkeypatch.setattr(demo, "DEMO_USER_PASSWORD", "docpilot-demo!")
    monkeypatch.setattr(demo, "FRONTEND_URL", "http://other.test:3100")
    with pytest.raises(demo.CliError, match=r"HTTP 403.*CORS_ORIGINS"):
        demo.sign_in_opener()


def test_a_sign_in_to_a_down_api_is_a_cli_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(demo, "API_URL", "http://127.0.0.1:9")
    with pytest.raises(demo.CliError, match="could not reach"):
        demo.sign_in_opener()


def test_the_opener_signs_in_once_for_the_whole_run(demo_api: FakeDemoApi) -> None:
    assert demo.authenticated_opener() is demo.authenticated_opener()
    assert demo_api.logins == 1


def test_the_stack_is_ready_when_the_signed_in_list_has_documents(
    demo_api: FakeDemoApi,
) -> None:
    demo.ensure_stack_ready()
    assert demo_api.logins == 1


def test_budget_state_reads_stats_through_the_session(demo_api: FakeDemoApi) -> None:
    assert demo.budget_state() == (1.5, 0.25)


def test_a_down_api_is_reported_without_signing_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(demo, "API_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(demo, "FRONTEND_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(demo, "_opener", None)
    with pytest.raises(demo.CliError, match="demo.py stack up"):
        demo.ensure_stack_ready()
    assert demo._opener is None


DEMO_SEED = demo.REPO_ROOT / "backend" / "app" / "demo_seed.py"


def literal_assignments(path: Path, names: set[str]) -> dict[str, str]:
    """The module-level `NAME = "literal"` assignments of `path` for `names`,
    read from its syntax tree: the backend is not importable here."""
    found = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                found[target.id] = ast.literal_eval(node.value)
    return found


def test_the_demo_credentials_match_the_seed() -> None:
    seeded = literal_assignments(DEMO_SEED, {"DEMO_USER_EMAIL", "DEMO_USER_PASSWORD"})
    assert seeded == {
        "DEMO_USER_EMAIL": demo.DEMO_USER_EMAIL,
        "DEMO_USER_PASSWORD": demo.DEMO_USER_PASSWORD,
    }


def test_record_signs_the_browser_in_as_the_demo_user() -> None:
    # Playwright is not imported here, so the call is read, not run: record()
    # passes the demo credentials to the harness's recording().
    source = (Path(demo.__file__)).read_text(encoding="utf-8")
    call = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "recording"
    )
    keywords = {k.arg: ast.unparse(k.value) for k in call.keywords}
    assert keywords == {"email": "DEMO_USER_EMAIL", "password": "DEMO_USER_PASSWORD"}
