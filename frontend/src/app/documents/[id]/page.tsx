"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import {
  getDocument,
  documentFileUrl,
  ApiError,
  type DocumentDetail,
} from "@/lib/api";
import ExtractionPanel from "@/components/ExtractionPanel";

const POLL_INTERVAL_MS = 2000;
const IN_FLIGHT_STATUSES = new Set(["uploaded", "processing"]);

function DocumentPreview({ doc }: { doc: DocumentDetail }) {
  const url = documentFileUrl(doc.id);
  if (doc.mime_type === "application/pdf") {
    return (
      <div className="doc-preview">
        <iframe src={url} title={doc.filename} />
      </div>
    );
  }
  return (
    <div className="doc-preview">
      {/* eslint-disable-next-line @next/next/no-img-element -- external
          binary served by the FastAPI backend, not a build-time asset. */}
      <img src={url} alt={doc.filename} />
    </div>
  );
}

export default function DocumentPage() {
  const params = useParams<{ id: string }>();
  const id = params.id;

  const [doc, setDoc] = useState<DocumentDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Bumped after every fetch attempt, success or failure — see the same
  // pattern (and rationale) in app/page.tsx.
  const [tick, setTick] = useState(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const refresh = useCallback(async () => {
    try {
      const fetched = await getDocument(id);
      setDoc(fetched);
      setError(null);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? `API error: ${err.message}`
          : "Could not reach the API. Is the backend running?"
      );
    } finally {
      setTick((t) => t + 1);
    }
  }, [id]);

  useEffect(() => {
    // Initial load; setState happens after the awaited fetch resolves, not
    // synchronously in the effect body, so this isn't the render loop the
    // set-state-in-effect rule guards against. No query library in scope
    // per spec (plain fetch/poll), so this is the intended pattern.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    // `doc === null` covers both "hasn't loaded yet" and "every attempt so
    // far has failed" — including the upload-then-navigate race, where the
    // first GET can 404/fail before the document row is visible yet. A
    // failed attempt never touches `doc`, so this keeps retrying instead
    // of the chain dying on the first failure. `tick` forces this effect
    // to re-run after a failed attempt even though `doc` didn't change.
    const shouldContinue = doc === null || IN_FLIGHT_STATUSES.has(doc.status);
    if (shouldContinue) {
      timerRef.current = setTimeout(() => void refresh(), POLL_INTERVAL_MS);
    }
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [doc, tick, refresh]);

  return (
    <div>
      <Link href="/" className="detail-back">
        ← back to documents
      </Link>

      {/* Non-fatal: the split view below (once loaded) stays visible
          through a transient poll failure instead of being replaced. */}
      {error && <div className="error-banner">{error}</div>}

      {doc === null && <div className="empty-state">Loading…</div>}

      {doc !== null && (
        <>
          <div className="detail-title">{doc.filename}</div>
          <div className="split-view">
            <DocumentPreview doc={doc} />

            {IN_FLIGHT_STATUSES.has(doc.status) && (
              <div className="results-panel">
                <div className="status-panel">
                  <div className="spinner" />
                  <div>extracting…</div>
                </div>
              </div>
            )}

            {doc.status === "failed" && (
              <div className="results-panel">
                <div className="failed-state">
                  <div>Extraction failed</div>
                  <div className="hint">
                    Check the worker logs for this document&apos;s job.
                  </div>
                </div>
              </div>
            )}

            {doc.status === "refused" && (
              <div className="results-panel">
                <div className="failed-state">
                  <div>Extraction refused</div>
                  <div className="hint">
                    The model declined to extract this document — nothing was
                    stored. This isn&apos;t retried automatically.
                  </div>
                </div>
              </div>
            )}

            {doc.status === "extracted" &&
              (doc.extraction ? (
                <ExtractionPanel extraction={doc.extraction} />
              ) : (
                <div className="results-panel">
                  <div className="empty-state">
                    Marked extracted but no extraction data was found.
                  </div>
                </div>
              ))}
          </div>
        </>
      )}
    </div>
  );
}
