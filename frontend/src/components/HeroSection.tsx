import { useRef } from "react";
import Navigation from "./Navigation";
import RevealLayer from "./RevealLayer";
import { useSpotlight } from "../hooks/useSpotlight";
import { BG_IMAGE_CLOUDY, BG_IMAGE_RECONSTRUCTED, METRICS } from "../constants/hero";

export default function HeroSection() {
  const revealRef = useRef<HTMLDivElement>(null);
  useSpotlight(revealRef);

  return (
    <>
      {/* ── Fixed Navigation (z-100) ── */}
      <Navigation />

      {/* ── Hero section ── */}
      <section
        aria-label="CloudVision AI hero — cursor spotlight reveals AI reconstruction"
        className="h-screen-safe relative w-full overflow-hidden bg-black"
      >
        {/* Layer 1 — Cloudy base image (z-10) */}
        <div
          className="hero-zoom absolute inset-0 z-10 bg-center bg-cover bg-no-repeat"
          style={{ backgroundImage: `url(${BG_IMAGE_CLOUDY})` }}
        />

        {/* Layer 2 — AI reconstruction, revealed by spotlight (z-30) */}
        <RevealLayer ref={revealRef} image={BG_IMAGE_RECONSTRUCTED} />

        {/* Layer 3 — Dark gradient overlay (z-40) */}
        <div
          className="absolute inset-0 z-40 pointer-events-none"
          style={{
            background:
              "linear-gradient(to bottom, rgba(0,0,0,0.55) 0%, rgba(0,0,0,0.08) 28%, rgba(0,0,0,0.08) 58%, rgba(0,0,0,0.80) 100%)",
          }}
        />

        {/* Layer 4 — All UI content (z-50) */}
        <div className="absolute inset-0 z-50 pointer-events-none select-none">

          {/* ── Floating image labels ── */}
          <span className="absolute top-20 sm:top-24 left-5 sm:left-8 bg-black/30 backdrop-blur-md border border-white/10 rounded-full px-4 py-2 text-xs text-white/80">
            Cloudy LISS-IV Input
          </span>
          <span className="absolute top-20 sm:top-24 right-5 sm:right-8 bg-black/30 backdrop-blur-md border border-white/10 rounded-full px-4 py-2 text-xs text-white/80">
            AI Reconstruction
          </span>

          {/* ── Hero heading + metrics ── */}
          <div className="absolute top-[14%] left-0 right-0 flex flex-col items-center text-center px-5">
            <h1 className="text-white leading-[0.95]">
              <span
                className="hero-anim hero-reveal block font-extrabold text-5xl sm:text-7xl md:text-8xl"
                style={{ letterSpacing: "-0.08em", animationDelay: "0.25s" }}
              >
                CloudVision
              </span>
              <span
                className="hero-anim hero-reveal block font-light text-5xl sm:text-7xl md:text-8xl -mt-1"
                style={{ letterSpacing: "-0.08em", animationDelay: "0.42s" }}
              >
                AI
              </span>
            </h1>

            <p className="max-w-3xl mt-6 text-white/75 text-sm sm:text-base md:text-lg leading-relaxed">
              Generative AI-Based Cloud Removal and Reconstruction for LISS-IV
              Satellite Imagery
            </p>

            {/* Metrics strip */}
            <div
              className="hero-anim hero-fade mt-8 flex flex-wrap justify-center gap-3"
              style={{ animationDelay: "1s" }}
            >
              {METRICS.map(({ label, value, unit }) => (
                <div
                  key={label}
                  className="bg-white/10 backdrop-blur-md border border-white/10 rounded-2xl px-5 py-3 flex flex-col items-center min-w-[76px]"
                >
                  <span className="text-white/50 text-[10px] font-semibold uppercase tracking-widest mb-1">
                    {label}
                  </span>
                  <span className="text-white text-lg font-bold leading-none">
                    {value}
                    {unit && (
                      <span className="text-white/55 text-sm font-normal ml-0.5">
                        {unit}
                      </span>
                    )}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* ── Bottom-left: context paragraph ── */}
          <div
            className="hero-anim hero-fade hidden sm:block absolute bottom-14 left-10 md:left-14 max-w-[280px]"
            style={{ animationDelay: "0.7s" }}
          >
            <p className="text-white/50 text-xs leading-relaxed">
              Cloud cover obscures critical Earth observation data used in
              agriculture, disaster response, environmental monitoring, and
              urban planning.
            </p>
          </div>

          {/* ── Bottom-right: instruction + CTA ── */}
          <div
            className="hero-anim hero-fade absolute bottom-10 sm:bottom-24 left-5 right-5 sm:left-auto sm:right-10 md:right-14 max-w-full sm:max-w-[300px] flex flex-col items-start gap-4 sm:gap-5"
            style={{ animationDelay: "0.85s" }}
          >
            <p className="text-white/60 text-xs sm:text-sm leading-relaxed">
              Move your cursor across the scene to reveal the AI reconstructed
              image generated from cloud-obscured satellite observations.
            </p>

            <button
              type="button"
              className="pointer-events-auto bg-orange-500 hover:bg-orange-600 text-white text-sm font-medium px-7 py-3 rounded-full transition-all hover:scale-[1.03] active:scale-95 hover:shadow-lg hover:shadow-orange-500/30 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-orange-400"
            >
              Explore Reconstruction
            </button>
          </div>

        </div>
      </section>
    </>
  );
}
