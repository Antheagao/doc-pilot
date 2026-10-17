#!/usr/bin/env python
"""Check the pre-registered bar for making extraction schema v2 the default.

Takes a model's v1 eval artifact and its v2 eval artifact and prints each term
of the bar (app/evals/bar.py) as PASS or FAIL with the numbers behind it. Exits
0 only when every term passes. Free and offline: it reads two files.

Run with the backend venv's interpreter, from the repo root:

    backend/.venv/bin/python evals/check_bar.py <v1-artifact.json> <v2-artifact.json>
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _load(path: str) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read eval artifact {path}: {exc}", file=sys.stderr)
        sys.exit(2)
    if not isinstance(data, dict):
        print(f"error: {path} is not an eval artifact (a JSON object)", file=sys.stderr)
        sys.exit(2)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the schema v2 acceptance bar.")
    parser.add_argument("v1_artifact", help="the model's extract_v1 run (evals/results/*.json)")
    parser.add_argument("v2_artifact", help="the same model's run with evidence (extract_v2 or extract_v3)")
    args = parser.parse_args()

    from app.evals.bar import evaluate_bar

    v1, v2 = _load(args.v1_artifact), _load(args.v2_artifact)
    try:
        terms = evaluate_bar(v1, v2)
    except KeyError as exc:
        print(f"error: not an eval artifact, missing key {exc}", file=sys.stderr)
        return 2

    for term in terms:
        print(f"{'PASS' if term.passed else 'FAIL'}  {term.name}: {term.detail}")
    failed = [term.name for term in terms if not term.passed]
    print(f"\nbar: {'met' if not failed else 'NOT met (' + ', '.join(failed) + ')'}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
