import HeroSection from "./components/HeroSection";
import PipelinePanel from "./components/PipelinePanel";

export default function App() {
  return (
    <div className="min-h-screen bg-black tracking-[-0.02em]">
      <HeroSection />

      {/* ── Live pipeline workspace ── */}
      <section
        id="pipeline"
        aria-label="Live cloud-removal pipeline"
        className="relative w-full bg-black px-5 sm:px-8 py-16 sm:py-24"
      >
        <div className="max-w-5xl mx-auto mb-10 text-center">
          <h2 className="text-white text-3xl sm:text-4xl font-extrabold" style={{ letterSpacing: "-0.05em" }}>
            Live Demo Pipeline
          </h2>
          <p className="text-white/50 text-sm sm:text-base mt-3 max-w-2xl mx-auto">
            Run cloud detection → reconstruction → scientific validation on a bundled
            ISRO scene, or upload your own LISS-IV image.
          </p>
        </div>
        <PipelinePanel />
      </section>
    </div>
  );
}
