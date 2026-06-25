import { useEffect, useRef } from "react";

const LERP = 0.1;

export function useSpotlight(targetRef: React.RefObject<HTMLElement | null>) {
  const mouse  = useRef({ x: -999, y: -999 });
  const smooth = useRef({ x: -999, y: -999 });
  const active = useRef(false);
  const rafId  = useRef(0);

  useEffect(() => {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const setVars = (x: number, y: number) => {
      const el = targetRef.current;
      if (!el) return;
      el.style.setProperty("--cx", `${x}px`);
      el.style.setProperty("--cy", `${y}px`);
    };

    const tick = () => {
      if (active.current) {
        smooth.current.x += (mouse.current.x - smooth.current.x) * LERP;
        smooth.current.y += (mouse.current.y - smooth.current.y) * LERP;
        setVars(smooth.current.x, smooth.current.y);
      }
      rafId.current = requestAnimationFrame(tick);
    };

    const onMove = (e: PointerEvent) => {
      mouse.current = { x: e.clientX, y: e.clientY };
      if (!active.current) {
        smooth.current = { x: e.clientX, y: e.clientY };
        active.current = true;
      }
      if (reduced) setVars(e.clientX, e.clientY);
    };

    const onLeave = () => {
      active.current = false;
      setVars(-999, -999);
    };

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerleave", onLeave);

    if (!reduced) {
      rafId.current = requestAnimationFrame(tick);
    }

    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerleave", onLeave);
      cancelAnimationFrame(rafId.current);
    };
  }, [targetRef]);
}
