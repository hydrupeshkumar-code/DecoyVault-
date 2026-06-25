/**
 * ISRO demo-mode scene selector.
 *
 * Presents three bundled synthetic LISS-IV scenes (agriculture / urban / water)
 * that exercise the full pipeline with NO upload, checkpoint, or GPU required.
 * Selecting a scene hands its demo file ID (`demo_<scene>`) to the parent — the
 * backend resolves that ID to datasets/demo/<scene>/{cloudy,clear,mask}.npy.
 *
 * Thumbnails are produced by datasets/demo/generate_demo.py into
 * frontend/public/demo/, so they are served at /demo/<scene>_before.png.
 */

export interface DemoScene {
  id: string;          // scene key, e.g. "agriculture"
  fileId: string;      // backend demo file ID, e.g. "demo_agriculture"
  title: string;
  blurb: string;
  thumb: string;       // before (cloudy) preview
  accent: string;      // tailwind text colour
}

export const DEMO_SCENES: DemoScene[] = [
  {
    id: "agriculture",
    fileId: "demo_agriculture",
    title: "Agriculture",
    blurb: "High-NDVI farmland mosaic — vegetation index fidelity is critical.",
    thumb: "/demo/agriculture_before.png",
    accent: "text-green-400",
  },
  {
    id: "urban",
    fileId: "demo_urban",
    title: "Urban",
    blurb: "Built-up grid — fine-scale texture recovery under cloud.",
    thumb: "/demo/urban_before.png",
    accent: "text-amber-400",
  },
  {
    id: "water",
    fileId: "demo_water",
    title: "Water",
    blurb: "Reservoir / coast — low and negative NDVI, specular reflection.",
    thumb: "/demo/water_before.png",
    accent: "text-sky-400",
  },
];

interface ISRODemoModeProps {
  onSelectScene: (sceneId: string, demoFileId: string) => void;
  activeScene?: string | null;
  disabled?: boolean;
}

export default function ISRODemoMode({ onSelectScene, activeScene = null, disabled = false }: ISRODemoModeProps) {
  return (
    <div>
      <div className="mb-3">
        <h3 className="text-white text-sm font-semibold tracking-wide">ISRO Demo Scenes</h3>
        <p className="text-white/40 text-xs mt-0.5">
          No upload needed — runs end-to-end on bundled synthetic LISS-IV data.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {DEMO_SCENES.map((scene) => {
          const active = activeScene === scene.id;
          return (
            <button
              key={scene.id}
              type="button"
              disabled={disabled}
              onClick={() => onSelectScene(scene.id, scene.fileId)}
              className={[
                "group text-left rounded-2xl overflow-hidden border transition-all",
                disabled ? "opacity-50 cursor-not-allowed" : "hover:scale-[1.01]",
                active ? "border-orange-400 ring-2 ring-orange-400/40" : "border-white/10 hover:border-white/30",
              ].join(" ")}
            >
              <div className="aspect-video w-full overflow-hidden bg-white/5">
                <img
                  src={scene.thumb}
                  alt={`${scene.title} demo scene (cloudy)`}
                  className="w-full h-full object-cover"
                  draggable={false}
                />
              </div>
              <div className="px-3 py-2.5 bg-white/5">
                <span className={`text-xs font-semibold ${scene.accent}`}>{scene.title}</span>
                <p className="text-white/40 text-[10px] leading-snug mt-0.5">{scene.blurb}</p>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
