"use client";

import { useEffect, useState } from "react";
import { getStats, type Stats } from "@/lib/api";

function formatCost(value: number | null): string {
  return typeof value === "number" ? `$${value.toFixed(4)}` : "—";
}

function formatLatency(value: number | null): string {
  return typeof value === "number" ? `${Math.round(value)}ms` : "—";
}

/** Home-page stats strip: documents processed, cost per document
 * (extraction + transcription), total spend across every model-calling
 * stage, latency, review queue depth, and the most recent eval accuracy.
 *
 * These numbers don't move second-to-second the way the document list
 * does, so this fetches once on mount (the ReviewQueueLink pattern)
 * rather than polling. A fetch failure renders nothing — the page
 * already has its own error banner for a down backend, and this strip
 * isn't essential enough to duplicate it. */
export default function StatsStrip() {
  const [stats, setStats] = useState<Stats | null>(null);

  useEffect(() => {
    let cancelled = false;
    getStats()
      .then((s) => {
        if (!cancelled) setStats(s);
      })
      .catch(() => {
        if (!cancelled) setStats(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (stats === null) return null;

  return (
    <div className="stats-strip">
      <div className="stats-strip__cell">
        <div className="stats-strip__label">docs processed</div>
        <div className="stats-strip__value">{stats.documents_processed}</div>
      </div>
      <div className="stats-strip__cell">
        <div className="stats-strip__label">avg $/doc</div>
        <div className="stats-strip__value">
          {formatCost(stats.mean_pipeline_cost_per_doc)}
        </div>
        <div className="stats-strip__sub">extract + transcribe</div>
      </div>
      <div className="stats-strip__cell">
        <div className="stats-strip__label">total spend</div>
        <div className="stats-strip__value">{formatCost(stats.spend.total_usd)}</div>
        <div className="stats-strip__sub">
          {stats.ask.runs} question{stats.ask.runs === 1 ? "" : "s"} ·{" "}
          {formatCost(stats.spend.agent_usd + stats.spend.judge_usd)}
        </div>
      </div>
      <div className="stats-strip__cell">
        <div className="stats-strip__label">p50 / p95 latency</div>
        <div className="stats-strip__value">
          {formatLatency(stats.latency_p50_ms)} /{" "}
          {formatLatency(stats.latency_p95_ms)}
        </div>
      </div>
      <div className="stats-strip__cell">
        <div className="stats-strip__label">pending review</div>
        <div className="stats-strip__value">{stats.review.pending}</div>
      </div>
      {stats.last_eval && (
        <div className="stats-strip__cell">
          <div className="stats-strip__label">eval accuracy</div>
          <div className="stats-strip__value">
            {typeof stats.last_eval.overall_accuracy === "number"
              ? `${(stats.last_eval.overall_accuracy * 100).toFixed(1)}%`
              : "—"}
          </div>
          <div className="stats-strip__sub">
            {stats.last_eval.model ?? "n/a"} / {stats.last_eval.prompt_version ?? "n/a"}
          </div>
        </div>
      )}
    </div>
  );
}
