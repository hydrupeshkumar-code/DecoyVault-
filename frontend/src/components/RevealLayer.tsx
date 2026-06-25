import { forwardRef } from "react";
import { SPOTLIGHT_RADIUS } from "../constants/hero";

interface RevealLayerProps {
  image: string;
}

const r = SPOTLIGHT_RADIUS;
const MASK_GRADIENT =
  `radial-gradient(circle ${r}px at var(--cx, -999px) var(--cy, -999px),` +
  ` rgba(255,255,255,1) 0%,` +
  ` rgba(255,255,255,1) 40%,` +
  ` rgba(255,255,255,0.75) 60%,` +
  ` rgba(255,255,255,0.40) 75%,` +
  ` rgba(255,255,255,0.12) 88%,` +
  ` rgba(255,255,255,0) 100%)`;

const RevealLayer = forwardRef<HTMLDivElement, RevealLayerProps>(
  ({ image }, ref) => (
    <div
      ref={ref}
      className="absolute inset-0 bg-center bg-cover bg-no-repeat z-30 pointer-events-none"
      style={{
        backgroundImage:  `url(${image})`,
        maskImage:        MASK_GRADIENT,
        WebkitMaskImage:  MASK_GRADIENT,
      } as React.CSSProperties}
    />
  )
);

RevealLayer.displayName = "RevealLayer";

export default RevealLayer;
