"""Download and verify the pinned embedding model; print its directory.

The API and worker do this on their own at first use. This script is for
doing it ahead of time -- CI caches the result, and a host can pre-fetch
before going offline (then point EMBEDDING_MODEL_PATH at the printed
directory):

    python scripts/fetch_embedding_model.py [--dest DIR] [--model NAME]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.retrieval.model_fetch import ensure_pinned_model


def main() -> int:
    # INFO shows each file's URL and the revision the host served.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", default=settings.embedding_cache_dir, help="default: EMBEDDING_CACHE_DIR")
    parser.add_argument("--model", default=settings.embedding_model)
    args = parser.parse_args()
    path = ensure_pinned_model(args.model, args.dest)
    if path is None:
        print(f"{args.model} is not pinned; fastembed will download it itself", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
