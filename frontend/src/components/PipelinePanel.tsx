/**
 * End-to-end pipeline workspace.
 *
 * Source selection (upload OR a bundled demo scene) → detect → reconstruct →
 * metrics (when a clear reference exists) → report, with live per-step status.
 *
 * Demo scenes carry a clear reference, so the full scientific metric suite and
 * verdict are available. Uploaded cloudy scenes have no ground truth, so the
 * metrics step is skipped and the report is generated without it.
 */

import { useCallback, useMemo, useState } from "react";
import {
  detectClouds,
  reconstruct,
  computeMetrics,
  generateReport,
  reportDownloadUrl,
  type MetricsResponse,
} from "../api/cloudvision";
import UploadPanel from "./UploadPanel";
import ISRODemoMode, { DEMO_SCENES } from "./ISRODemoMode";
import MetricsDashboard from "./MetricsDashboard";
import BeforeAfterSlider from "./BeforeAfterSlider";

type Status = "idle" | "running" | "done" | "error" | "skipped";
type StepKey = "detect" | "reconstruct" | "metrics" | "report";

interface Source {
  kind: "demo" | "upload";
  fileId: string;
  label: string;
  sceneId?: string;
  hasReference: boolean;
}

const STEP_LABELS: Record<StepKey, string> = {
  detect: "Detect clouds",
  reconstruct: "Reconstruct (remove clouds)",
  metrics: "Compute scientific metrics",
  report: "Generate report",
};

const INITIAL_STEPS: Record<StepKey, Status> = {
  detect: "idle",
  reconstruct: "idle",
  metrics: "idle",
  report: "idle",
};

function StatusDot({ status }: { status: Status }) {
  const map: Record<Status, string> = {
    idle: "bg-white/20",
    running: "bg-orange-400 animate-pulse",
    done: "bg-green-400",
    error: "bg-red-500",
    skipped: "bg-white/10",
  };
  return <span className={`inline-block w-2.5 h-2.5 rounded-full ${map[status]}`} />;
}

export default function PipelinePanel() {
  const [source, setSource] = useState<Source | null>(null);
  const [steps, setSteps] = useState<Record<StepKey, Status>>(INITIAL_STEPS);
  const [running, setRunning] = useState(false);
  const [cloudFraction, setCloudFraction] = useState<number | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [reportPath, setReportPath] = useState<string | null>(null);
  const [fallbackUsed, setFallbackUsed] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const demoThumbs = useMemo(
    () => DEMO_SCENES.find((s) => s.id === source?.sceneId),
    [source],
  );

  const reset = useCallback(() => {
    setSteps(INITIAL_STEPS);
    setCloudFraction(null);
    setMetrics(null);
    setReportPath(null);
    setFallbackUsed(false);
    setError(null);
  }, []);

  const selectDemo = useCallback(
    (sceneId: string, fileId: string) => {
      reset();
      setSource({ kind: "demo", fileId, sceneId, label: `Demo · ${sceneId}`, hasReference: true });
    },
    [reset],
  );

  const selectUpload = useCallback(
    (fileId: string) => {
      reset();
      setSource({ kind: "upload", fileId, label: "Uploaded scene", hasReference: false });
    },
    [reset],
  );

  const setStep = (key: StepKey, status: Status) =>
    setSteps((prev) => ({ ...prev, [key]: status }));

  const runPipeline = useCallback(async () => {
    if (!source) return;
    setRunning(true);
    reset();
    try {
      // 1 — detect
      setStep("detect", "running");
      const det = await detectClouds(source.fileId);
      setCloudFraction(det.cloud_fraction);
      setStep("detect", "done");

      // 2 — reconstruct
      setStep("reconstruct", "running");
      const rec = await reconstruct(source.fileId, det.mask_id);
      setFallbackUsed(rec.fallback);
      setStep("reconstruct", "done");

      // 3 — metrics (only when a clear reference exists)
      if (source.hasReference) {
        setStep("metrics", "running");
        const m = await computeMetrics({
          pred_id: rec.result_id,
          target_id: source.fileId,
          mask_id: det.mask_id,
          cloudy_id: source.fileId,
        });
        setMetrics(m);
        setStep("metrics", "done");
      } else {
        setStep("metrics", "skipped");
      }

      // 4 — report
      setStep("report", "running");
      const rep = await generateReport({
        pred_id: rec.result_id,
        file_id: source.fileId,
        target_id: source.hasReference ? source.fileId : null,
        mask_id: det.mask_id,
        scene_id: source.sceneId ?? source.label,
      });
      setReportPath(rep.report_path);
      setStep("report", "done");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Pipeline failed");
      setSteps((prev) => {
        const next = { ...prev };
        (Object.keys(next) as StepKey[]).forEach((k) => {
          if (next[k] === "running") next[k] = "error";
        });
        return next;
      });
    } finally {
      setRunning(false);
    }
  }, [source, reset]);

  return (
    <div className="text-white max-w-5xl mx-auto">
      {/* Source selection */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <ISRODemoMode
          onSelectScene={selectDemo}
          activeScene={source?.kind === "demo" ? source.sceneId : null}
          disabled={running}
        />
        <div>
          <div className="mb-3">
            <h3 className="text-white text-sm font-semibold tracking-wide">Upload Your Own</h3>
            <p className="text-white/40 text-xs mt-0.5">
              3-band LISS-IV scene · metrics need a clear reference (demo scenes include one).
            </p>
          </div>
          <UploadPanel onUploaded={selectUpload} disabled={running} />
        </div>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4 mb-6">
        <button
          type="button"
          disabled={!source || running}
          onClick={() => void runPipeline()}
          className={[
            "px-6 py-2.5 rounded-full text-sm font-semibold transition-all",
            !source || running
              ? "bg-white/10 text-white/40 cursor-not-allowed"
              : "bg-orange-500 hover:bg-orange-600 text-white hover:scale-[1.02]",
          ].join(" ")}
        >
          {running ? "Running…" : "Run Pipeline"}
        </button>
        {source && (
          <span className="text-xs text-white/50">
            Source: <span className="text-white/80">{source.label}</span>
          </span>
        )}
        {cloudFraction !== null && (
          <span className="text-xs text-white/50">
            Cloud cover: <span className="text-white/80">{(cloudFraction * 100).toFixed(1)}%</span>
          </span>
        )}
      </div>

      {/* Step status */}
      <div className="flex flex-col gap-2 mb-8">
        {(Object.keys(STEP_LABELS) as StepKey[]).map((key) => (
          <div key={key} className="flex items-center gap-3 text-sm">
            <StatusDot status={steps[key]} />
            <span className={steps[key] === "skipped" ? "text-white/30 line-through" : "text-white/80"}>
              {STEP_LABELS[key]}
            </span>
            {steps[key] === "skipped" && (
              <span className="text-white/30 text-xs">(no clear reference)</span>
            )}
          </div>
        ))}
      </div>

      {error && (
        <div className="mb-6 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      {fallbackUsed && (
        <div className="mb-6 rounded-xl border border-amber-400/30 bg-amber-400/10 px-4 py-3 text-xs text-amber-200">
          No trained checkpoint is loaded — cloud pixels were filled with the clear-region
          per-band mean (a baseline, not a real Restormer reconstruction). Train a model and
          mount its checkpoint to see true results.
        </div>
      )}

      {/* Before/after (demo scenes only — they ship preview thumbnails) */}
      {demoThumbs && (
        <div className="mb-8">
          <h3 className="text-white text-sm font-semibold tracking-wide mb-3">Before / After</h3>
          <BeforeAfterSlider
            beforeSrc={`/demo/${demoThumbs.id}_before.png`}
            afterSrc={`/demo/${demoThumbs.id}_after.png`}
            beforeLabel="Cloudy LISS-IV Input"
            afterLabel="Clear Reference (target)"
            className="aspect-video w-full max-h-[420px]"
          />
        </div>
      )}

      {/* Metrics */}
      {metrics && (
        <div className="mb-8">
          <MetricsDashboard metrics={metrics} />
        </div>
      )}

      {/* Report download */}
      {reportPath && (
        <a
          href={reportDownloadUrl(reportPath)}
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-white text-black text-sm font-semibold hover:bg-gray-100 transition-all"
        >
          Download Report
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M12 4v12m0 0l-4-4m4 4l4-4M4 20h16" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </a>
      )}
    </div>
  );
}
