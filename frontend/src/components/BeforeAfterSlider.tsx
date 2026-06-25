/**
 * Before/After comparison slider for satellite imagery.
 *
 * Drag left → see more of the cloudy input (LISS-IV original).
 * Drag right → see more of the AI-reconstructed output.
 *
 * Designed for scientific visualization — no "artistic" filters or
 * color corrections are applied. The raw band data is shown as-is.
 */

import { useCallback, useRef, useState } from "react";

interface BeforeAfterSliderProps {
  beforeSrc: string;
  afterSrc: string;
  beforeLabel?: string;
  afterLabel?: string;
  className?: string;
}

export default function BeforeAfterSlider({
  beforeSrc,
  afterSrc,
  beforeLabel = "Cloudy LISS-IV Input",
  afterLabel = "AI Reconstruction",
  className = "",
}: BeforeAfterSliderProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [split, setSplit] = useState(50); // percentage
  const dragging = useRef(false);

  const clampedSplit = Math.max(2, Math.min(98, split));

  const updateSplit = useCallback((clientX: number) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setSplit(Math.max(2, Math.min(98, pct)));
  }, []);

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    dragging.current = true;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    updateSplit(e.clientX);
  }, [updateSplit]);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (!dragging.current) return;
    updateSplit(e.clientX);
  }, [updateSplit]);

  const onPointerUp = useCallback(() => {
    dragging.current = false;
  }, []);

  return (
    <div
      ref={containerRef}
      className={`relative select-none overflow-hidden rounded-2xl ${className}`}
      style={{ cursor: dragging.current ? "ew-resize" : "col-resize" }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    >
      {/* ── After (AI reconstruction) — full width base layer ── */}
      <img
        src={afterSrc}
        alt={afterLabel}
        className="w-full h-full object-cover block"
        draggable={false}
      />

      {/* ── Before (cloudy input) — clipped to left of split ── */}
      <div
        className="absolute inset-0 overflow-hidden"
        style={{ width: `${clampedSplit}%` }}
      >
        <img
          src={beforeSrc}
          alt={beforeLabel}
          className="w-full h-full object-cover block"
          style={{ width: containerRef.current?.offsetWidth ?? "100%" }}
          draggable={false}
        />
      </div>

      {/* ── Divider line ── */}
      <div
        className="absolute inset-y-0 pointer-events-none"
        style={{ left: `${clampedSplit}%`, transform: "translateX(-50%)" }}
      >
        <div className="absolute inset-y-0 w-px bg-white/70" />
        {/* Handle */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-white shadow-lg flex items-center justify-center">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M4 7H2M12 7h-2M7 4V2M7 12v-2" stroke="#111" strokeWidth="1.5" strokeLinecap="round" />
            <circle cx="7" cy="7" r="1.5" fill="#111" />
          </svg>
        </div>
      </div>

      {/* ── Labels ── */}
      <span className="absolute top-3 left-3 bg-black/50 backdrop-blur-sm text-white text-xs px-3 py-1 rounded-full pointer-events-none">
        {beforeLabel}
      </span>
      <span className="absolute top-3 right-3 bg-black/50 backdrop-blur-sm text-white text-xs px-3 py-1 rounded-full pointer-events-none">
        {afterLabel}
      </span>
    </div>
  );
}
