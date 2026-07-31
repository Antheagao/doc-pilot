"""Offline eval tooling: dataset loading (dataset.py) and field-level
scoring (scoring.py) for the labeled documents in repo-root evals/.

Deliberately pure/offline: nothing under app.evals imports app.worker or
opens a DB session, so this package stays importable (and testable) with
Postgres stopped. See dataset.py and scoring.py module docstrings.
"""
