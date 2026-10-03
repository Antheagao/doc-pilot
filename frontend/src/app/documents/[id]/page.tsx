"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import {
  getDocument,
  documentFileUrl,
  ApiError,
  type DocumentDetail,
} from "@/lib/api";
import ExtractionPanel, { fieldElementId } from "@/components/ExtractionPanel";
import ChatPanel, { type SelectedCitation } from "@/components/ChatPanel";

const POLL_INTERVAL_MS = 2000;
const IN_FLIGHT_STATUSES = new Set(["uploaded", "processing"]);

/** `page`, for a PDF, is the page a selected chat citation is on. */
function DocumentPreview({ doc, page }: { doc: DocumentDetail; page: number | null }) {
  const url = documentFileUrl(doc.id);
  if (doc.mime_type === "application/pdf") {
    const src = page ? `${url}#page=${page}` : url;
    return (
      <div className="doc-preview">
        {/* Keyed by src: a fragment change alone doesn't move the viewer. */}
        <iframe key={src} src={src} title={doc.filename} />
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
  // The chat citation a person selected: a field citation lights up its
  // row of the extraction; a text citation turns a PDF to its page.
  const [selected, setSelected] = useState<SelectedCitation | null>(null);
  const citation = selected?.citation ?? null;
  const highlight = useMemo(
    () => new Set(citation?.kind === "record" ? citation.fields : []),
    [citation]
  );
  const previewPage = citation && citation.kind !== "record" ? citation.page_number : null;

  useEffect(() => {
    const first = citation?.kind === "record" ? citation.fields[0] : undefined;
    if (first) {
      document
        .getElementById(fieldElementId(first))
        ?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }, [citation]);

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
            {/* The chat sits under the preview, beside the extraction, so
                the field a citation points at is in view next to it. */}
            <div className="detail-column">
              <DocumentPreview doc={doc} page={previewPage} />
              {doc.status === "extracted" && doc.extraction && (
                <ChatPanel
                  key={doc.id}
                  documentId={doc.id}
                  selected={selected}
                  onSelect={setSelected}
                />
              )}
            </div>

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
                <ExtractionPanel extraction={doc.extraction} highlight={highlight} />
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
