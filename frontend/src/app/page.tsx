"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { listDocuments, ApiError, type DocumentListItem } from "@/lib/api";
import StatusChip from "@/components/StatusChip";
import UploadZone from "@/components/UploadZone";
import StatsStrip from "@/components/StatsStrip";

const POLL_INTERVAL_MS = 3000;
const IN_FLIGHT_STATUSES = new Set(["uploaded", "processing"]);

function formatTime(iso: string): string {
  return new Date(iso).toLocaleString();
}

export default function HomePage() {
  const router = useRouter();
  const [documents, setDocuments] = useState<DocumentListItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Bumped after every fetch attempt, success or failure, purely to force
  // the scheduling effect below to re-run and queue the next poll even
  // when the attempt failed (a failure only changes `error`, which isn't
  // a dependency of that effect on purpose — see its comment).
  const [tick, setTick] = useState(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const refresh = useCallback(async () => {
    try {
      const fetched = await listDocuments();
      setDocuments(fetched);
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
  }, []);

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
    // `documents === null` covers both "hasn't loaded yet" and "every
    // attempt so far has failed" (a failed attempt never touches
    // `documents`), so a backend that's down at mount — or one that goes
    // down mid-session — keeps getting retried instead of the polling
    // chain dying after a single failure. `tick` forces this effect to
    // re-run after a failed attempt even though `documents` didn't change.
    const shouldContinue =
      documents === null || documents.some((d) => IN_FLIGHT_STATUSES.has(d.status));
    if (shouldContinue) {
      timerRef.current = setTimeout(() => void refresh(), POLL_INTERVAL_MS);
    }
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [documents, tick, refresh]);

  return (
    <div>
      <UploadZone onUploaded={(id) => router.push(`/documents/${id}`)} />

      <StatsStrip />

      <div className="section-heading">Documents</div>

      {/* Non-fatal: last-known-good list below stays visible through a
          transient poll failure instead of being replaced by the error. */}
      {error && <div className="error-banner">{error}</div>}

      {documents === null && !error && (
        <div className="empty-state">Loading…</div>
      )}

      {documents !== null && documents.length === 0 && (
        <div className="empty-state">
          No documents yet — upload one above to get started.
        </div>
      )}

      {documents !== null && documents.length > 0 && (
        <div className="doc-list">
          {documents.map((doc) => (
            <Link key={doc.id} href={`/documents/${doc.id}`} className="doc-row">
              <span className="doc-filename">{doc.filename}</span>
              <StatusChip status={doc.status} />
              <span className="doc-time">{formatTime(doc.created_at)}</span>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
