"use client";

import { useRef, useState } from "react";
import { uploadDocument } from "@/lib/api";

export default function UploadZone({
  onUploaded,
}: {
  onUploaded: (documentId: string) => void;
}) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleFile(file: File | undefined | null) {
    if (!file || uploading) return;
    setUploading(true);
    setError(null);
    try {
      const doc = await uploadDocument(file);
      onUploaded(doc.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  function openPicker() {
    // Ignore clicks/keypresses mid-upload so a second click doesn't fire a
    // second upload while the first is still in flight.
    if (uploading) return;
    inputRef.current?.click();
  }

  return (
    <div>
      <div
        className={`upload-zone${dragging ? " dragging" : ""}`}
        role="button"
        tabIndex={0}
        aria-label="Upload a document"
        aria-busy={uploading}
        aria-disabled={uploading}
        onClick={openPicker}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            openPicker();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          if (!uploading) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          void handleFile(e.dataTransfer.files?.[0]);
        }}
      >
        <div>{uploading ? "Uploading…" : "Drop a document here, or click to choose a file"}</div>
        <div className="hint">JPEG, PNG, WEBP, GIF, or PDF — up to 20MB</div>
        {/* Visually hidden rather than display:none — some browsers won't
            honor a synthetic .click() on a display:none input, and this
            keeps it out of the tab order (tabIndex -1) since the zone
            itself is the keyboard-operable control above. */}
        <input
          ref={inputRef}
          className="visually-hidden"
          tabIndex={-1}
          type="file"
          disabled={uploading}
          accept="image/jpeg,image/png,image/webp,image/gif,application/pdf"
          onChange={(e) => {
            void handleFile(e.target.files?.[0]);
            // Reset so re-selecting the same file still fires onChange.
            e.target.value = "";
          }}
        />
      </div>
      {error && <div className="error-state">{error}</div>}
    </div>
  );
}
