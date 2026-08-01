"""Pure PDF utilities for oversized-document chunking (H6).

pypdf + stdlib ONLY -- no app imports, no DB. app.extraction (the only
caller) is imported by app.evals.runner, and the eval suite must stay
runnable with Postgres down (see backend/tests/test_evals_scoring.py /
test_evals_runner.py, which are run with the db container stopped) --
so nothing in this module may pull in app.db, app.models, or anything
else that touches the database.
"""

from io import BytesIO

from pypdf import PdfReader, PdfWriter


def count_pdf_pages(raw: bytes) -> int:
    """Return the page count of a PDF given as raw bytes.

    Raises plain ValueError on unreadable or encrypted input -- callers
    (app.extraction.extract_document) treat that as a deterministic
    failure, not something worth reading further, and translate it into
    NonRetryableExtractionError before spending any API budget.
    """
    try:
        reader = PdfReader(BytesIO(raw))
        if reader.is_encrypted:
            raise ValueError("PDF is encrypted")
        return len(reader.pages)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"unreadable PDF: {exc}") from exc


def split_pdf_pages(raw: bytes) -> list[bytes]:
    """Split a PDF into one single-page PDF (as bytes) per page, in page
    order, via PdfWriter into a BytesIO buffer per page.

    Raises plain ValueError on unreadable or encrypted input, same as
    count_pdf_pages -- and for the same reason (a caller with an
    already-open try/except ValueError shouldn't need a second
    exception type to handle).
    """
    try:
        reader = PdfReader(BytesIO(raw))
        if reader.is_encrypted:
            raise ValueError("PDF is encrypted")
        pages: list[bytes] = []
        for page in reader.pages:
            writer = PdfWriter()
            writer.add_page(page)
            buffer = BytesIO()
            writer.write(buffer)
            pages.append(buffer.getvalue())
        return pages
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"unreadable PDF: {exc}") from exc
