import type { DocumentStatus } from "@/lib/api";

export default function StatusChip({ status }: { status: DocumentStatus }) {
  return <span className={`status-chip status-chip--${status}`}>{status}</span>;
}
