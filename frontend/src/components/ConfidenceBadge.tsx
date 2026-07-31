const LOW_CONFIDENCE_THRESHOLD = 0.8;

export default function ConfidenceBadge({
  confidence,
  needsReview,
}: {
  confidence: number;
  /** Defaults to confidence < 0.8 when the backend doesn't supply a flag
   * (e.g. per-cell line-item confidence, which has no ExtractedField row
   * of its own). */
  needsReview?: boolean;
}) {
  const low = needsReview ?? confidence < LOW_CONFIDENCE_THRESHOLD;
  return (
    <span className={`confidence-badge${low ? " low" : ""}`}>
      {Math.round(confidence * 100)}%
    </span>
  );
}
