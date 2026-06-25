/**
 * Drag-and-drop upload panel for LISS-IV imagery (.npy / .tif / .tiff).
 *
 * On success it reports the backend `file_id` (and full upload metadata) to the
 * parent via `onUploaded`, which the PipelinePanel uses to drive detect /
 * reconstruct.
 */

import { useCallback, useRef, useState } from "react";
import { uploadImage, type UploadResponse } from "../api/cloudvision";

interface UploadPanelProps {
  onUploaded: (fileId: string, info: UploadResponse) => void;
  disabled?: boolean;
}

const ACCEPTED = [".npy", ".tif", ".tiff"];

export default function UploadPanel({ onUploaded, disabled = false }: UploadPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastFile, setLastFile] = useState<string | null>(null);

  const handleFile = useCallback(
    async (file: File) => {
      const ext = "." + (file.name.split(".").pop() ?? "").toLowerCase();
      if (!ACCEPTED.includes(ext)) {
        setError(`Unsupported file type "${ext}". Accepted: ${ACCEPTED.join(", ")}`);
        return;
      }
      setError(null);
      setBusy(true);
      try {
        const info = await uploadImage(file);
        setLastFile(`${info.filename} · [${info.shape.join(" × ")}]`);
        onUploaded(info.file_id, info);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Upload failed");
      } finally {
        setBusy(false);
      }
    },
    [onUploaded],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      if (disabled || busy) return;
      const file = e.dataTransfer.files?.[0];
      if (file) void handleFile(file);
    },
    [disabled, busy, handleFile],
  );

  return (
    <div className="w-full">
      <div
        role="button"
        tabIndex={0}
        aria-disabled={disabled}
        onClick={() => !disabled && !busy && inputRef.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={[
          "flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed px-6 py-10 text-center transition-all",
          disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer",
          dragging
            ? "border-orange-400 bg-orange-400/10"
            : "border-white/20 bg-white/5 hover:border-white/40 hover:bg-white/[0.07]",
        ].join(" ")}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED.join(",")}
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void handleFile(file);
            e.target.value = "";
          }}
        />

        <svg width="34" height="34" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M12 16V4m0 0l-4 4m4-4l4 4" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" stroke="white" strokeOpacity="0.6" strokeWidth="1.5" strokeLinecap="round" />
        </svg>

        <div>
          <p className="text-white text-sm font-medium">
            {busy ? "Uploading…" : "Drop a LISS-IV scene here, or click to browse"}
          </p>
          <p className="text-white/40 text-xs mt-1">
            {ACCEPTED.join(" · ")} · 3-band Green/Red/NIR
          </p>
        </div>
      </div>

      {lastFile && !error && (
        <p className="mt-2 text-xs text-green-400/80">Loaded: {lastFile}</p>
      )}
      {error && <p className="mt-2 text-xs text-orange-400">{error}</p>}
    </div>
  );
}
