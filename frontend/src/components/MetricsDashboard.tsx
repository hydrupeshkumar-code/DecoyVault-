/**
 * Scientific Metrics Dashboard for ISRO Hackathon presentation.
 *
 * Displays the full validation suite:
 *   - Global image quality metrics (PSNR, SSIM, SAM, RMSE)
 *   - Cloud-region metrics (the actual scientific task)
 *   - Clear-region preservation (model quality check)
 *   - Vegetation fidelity (NDVI, for agricultural applications)
 *
 * Designed to communicate scientific rigor to ISRO reviewers.
 */


// ── Types ──────────────────────────────────────────────────────────────────

interface MetricsPayload {
  // Global
  psnr_db: number;
  ssim: number;
  sam_deg: number;
  rmse: number;
  ndvi_mae_global: number;
  // Cloud-region
  cloud_psnr_db?: number;
  cloud_ssim?: number;
  cloud_sam_deg?: number;
  cloud_rmse?: number;
  cloud_pixel_count?: number;
  // Clear preservation
  clear_psnr_db?: number;
  // Vegetation
  veg_ndvi_mae?: number;
  veg_ndvi_bias?: number;
  ndvi_correlation?: number;
  // Improvement
  improvement_db?: number;
}

interface MetricsDashboardProps {
  metrics: MetricsPayload;
}

// ── Sub-components ─────────────────────────────────────────────────────────

interface MetricCardProps {
  label: string;
  value: number | undefined;
  unit?: string;
  precision?: number;
  good?: (v: number) => boolean;
  description: string;
}

function MetricCard({ label, value, unit = "", precision = 2, good, description }: MetricCardProps) {
  const formatted = value !== undefined && !Number.isNaN(value)
    ? value.toFixed(precision)
    : "—";

  const status = value !== undefined && good
    ? good(value) ? "good" : "warn"
    : "neutral";

  return (
    <div className="bg-white/5 border border-white/10 rounded-2xl px-5 py-4 flex flex-col gap-1 min-w-[130px]">
      <span className="text-white/40 text-[10px] font-semibold uppercase tracking-widest">{label}</span>
      <span
        className={[
          "text-2xl font-bold leading-none",
          status === "good" ? "text-green-400" : status === "warn" ? "text-orange-400" : "text-white",
        ].join(" ")}
      >
        {formatted}
        {unit && <span className="text-base font-normal ml-1 text-white/50">{unit}</span>}
      </span>
      <span className="text-white/35 text-[10px] leading-snug mt-0.5">{description}</span>
    </div>
  );
}

interface SectionProps {
  title: string;
  subtitle: string;
  children: React.ReactNode;
}

function Section({ title, subtitle, children }: SectionProps) {
  return (
    <div className="mb-8">
      <div className="mb-3">
        <h3 className="text-white text-sm font-semibold tracking-wide">{title}</h3>
        <p className="text-white/40 text-xs mt-0.5">{subtitle}</p>
      </div>
      <div className="flex flex-wrap gap-3">{children}</div>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────

export default function MetricsDashboard({ metrics: m }: MetricsDashboardProps) {

  return (
    <div className="text-white">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-lg font-bold tracking-tight">Scientific Validation Report</h2>
        <p className="text-white/40 text-xs mt-1">
          LISS-IV Cloud Removal • Resourcesat-2 • Green / Red / NIR
        </p>
      </div>

      {/* 1. Global Metrics */}
      <Section
        title="Global Image Quality"
        subtitle="Measured across the full scene including cloud-free regions"
      >
        <MetricCard
          label="PSNR"
          value={m.psnr_db}
          unit="dB"
          precision={2}
          good={(v) => v >= 30}
          description="Peak signal-to-noise ratio"
        />
        <MetricCard
          label="SSIM"
          value={m.ssim}
          precision={4}
          good={(v) => v >= 0.85}
          description="Structural similarity index"
        />
        <MetricCard
          label="SAM"
          value={m.sam_deg}
          unit="°"
          precision={2}
          good={(v) => v <= 6}
          description="Spectral angle mapper"
        />
        <MetricCard
          label="RMSE"
          value={m.rmse}
          precision={4}
          good={(v) => v <= 0.03}
          description="Root mean square error"
        />
        <MetricCard
          label="NDVI MAE"
          value={m.ndvi_mae_global}
          precision={4}
          good={(v) => v <= 0.05}
          description="Global vegetation index error"
        />
      </Section>

      {/* 2. Cloud-Region Metrics (primary scientific task) */}
      {m.cloud_psnr_db !== undefined && (
        <Section
          title="Cloud-Region Reconstruction"
          subtitle="Primary evaluation — metrics computed only within cloud-masked pixels"
        >
          <MetricCard
            label="Cloud PSNR"
            value={m.cloud_psnr_db}
            unit="dB"
            precision={2}
            good={(v) => v >= 28}
            description="Reconstruction quality in cloud area"
          />
          <MetricCard
            label="Cloud SSIM"
            value={m.cloud_ssim}
            precision={4}
            good={(v) => v >= 0.80}
            description="Structural fidelity within clouds"
          />
          <MetricCard
            label="Cloud SAM"
            value={m.cloud_sam_deg}
            unit="°"
            precision={2}
            good={(v) => v <= 7}
            description="Spectral accuracy in cloud region"
          />
          <MetricCard
            label="Cloud RMSE"
            value={m.cloud_rmse}
            precision={4}
            good={(v) => v <= 0.04}
            description="Pixel error in cloud region"
          />
          {m.improvement_db !== undefined && (
            <MetricCard
              label="Improvement"
              value={m.improvement_db}
              unit="dB"
              precision={2}
              good={(v) => v > 0}
              description="PSNR gain vs. cloudy baseline"
            />
          )}
        </Section>
      )}

      {/* 3. Clear-Region Preservation */}
      {m.clear_psnr_db !== undefined && (
        <Section
          title="Clear-Region Preservation"
          subtitle="Model should not modify cloud-free pixels — higher is better"
        >
          <MetricCard
            label="Clear PSNR"
            value={m.clear_psnr_db}
            unit="dB"
            precision={2}
            good={(v) => v >= 40}
            description="Identity preservation in cloud-free areas"
          />
        </Section>
      )}

      {/* 4. Vegetation Metrics */}
      {m.veg_ndvi_mae !== undefined && (
        <Section
          title="Vegetation Fidelity (NDVI)"
          subtitle="Critical for agriculture, forestry, and land-cover monitoring applications"
        >
          <MetricCard
            label="NDVI MAE"
            value={m.veg_ndvi_mae}
            precision={4}
            good={(v) => v <= 0.04}
            description="Mean absolute error in vegetation NDVI"
          />
          <MetricCard
            label="NDVI RMSE"
            value={m.veg_ndvi_mae}
            precision={4}
            good={(v) => v <= 0.06}
            description="Root mean square NDVI error"
          />
          <MetricCard
            label="NDVI Bias"
            value={m.veg_ndvi_bias}
            precision={4}
            good={(v) => Math.abs(v) <= 0.02}
            description="Signed bias (+ = over-estimated)"
          />
          {m.ndvi_correlation !== undefined && (
            <MetricCard
              label="NDVI r"
              value={m.ndvi_correlation}
              precision={4}
              good={(v) => v >= 0.95}
              description="Pearson correlation with true NDVI"
            />
          )}
        </Section>
      )}

      {/* Legend */}
      <div className="flex items-center gap-4 mt-2 text-[10px] text-white/30">
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-green-400 inline-block" />
          Meets target threshold
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-orange-400 inline-block" />
          Below target threshold
        </span>
      </div>
    </div>
  );
}
