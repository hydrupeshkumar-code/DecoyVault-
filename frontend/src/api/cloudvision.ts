/**
 * Type-safe client for the CloudVision FastAPI backend.
 *
 * The base URL comes from Vite's build-time env (`VITE_API_URL`). When unset it
 * falls back to a relative path, which in dev is forwarded to the backend by the
 * Vite proxy (see vite.config.ts). NOTE: the variable MUST be `VITE_`-prefixed —
 * Vite silently ignores `REACT_APP_`-style names.
 */

// `import.meta.env` typing is provided in src/vite-env.d.ts
const API_BASE: string = import.meta.env.VITE_API_URL ?? "";

// ── Response types (mirror backend/schemas/*.py) ────────────────────────────

export interface UploadResponse {
  file_id: string;
  filename: string;
  format: string;
  shape: number[];
  size_bytes: number;
  is_georeferenced: boolean;
  crs: string | null;
  message: string;
}

export interface DetectResponse {
  file_id: string;
  cloud_fraction: number;
  mask_id: string;
  message: string;
}

export interface ReconstructResponse {
  file_id: string;
  result_id: string;
  elapsed_s: number;
  tiff_saved: boolean;
  fallback: boolean;
  message: string;
}

export interface MetricsResponse {
  pred_id: string;
  // Global
  psnr_db: number;
  ssim: number;
  sam_rad: number;
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
  veg_ndvi_rmse?: number;
  veg_ndvi_bias?: number;
  ndvi_correlation?: number;
  // Improvement
  ndvi_improvement?: number;
  improvement_db?: number;
}

export interface ReportResponse {
  report_path: string;
  format: string;
  message: string;
}

// ── Low-level helpers ───────────────────────────────────────────────────────

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body — keep status text */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return handle<T>(res);
}

// ── Public API ──────────────────────────────────────────────────────────────

export async function uploadImage(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/upload/`, { method: "POST", body: form });
  return handle<UploadResponse>(res);
}

export function detectClouds(fileId: string, threshold = 0.5): Promise<DetectResponse> {
  return postJSON<DetectResponse>("/detect/", { file_id: fileId, threshold });
}

export function reconstruct(
  fileId: string,
  maskId?: string | null,
  opts: { tile_size?: number; overlap?: number; preserve_clear?: boolean } = {},
): Promise<ReconstructResponse> {
  return postJSON<ReconstructResponse>("/reconstruct/", {
    file_id: fileId,
    mask_id: maskId ?? null,
    tile_size: opts.tile_size ?? 256,
    overlap: opts.overlap ?? 32,
    preserve_clear: opts.preserve_clear ?? true,
  });
}

export function computeMetrics(args: {
  pred_id: string;
  target_id: string;
  mask_id?: string | null;
  cloudy_id?: string | null;
}): Promise<MetricsResponse> {
  return postJSON<MetricsResponse>("/metrics/", {
    pred_id: args.pred_id,
    target_id: args.target_id,
    mask_id: args.mask_id ?? null,
    cloudy_id: args.cloudy_id ?? null,
  });
}

export function generateReport(args: {
  pred_id: string;
  file_id?: string | null;
  target_id?: string | null;
  mask_id?: string | null;
  scene_id?: string;
}): Promise<ReportResponse> {
  return postJSON<ReportResponse>("/report/generate", {
    pred_id: args.pred_id,
    file_id: args.file_id ?? null,
    target_id: args.target_id ?? null,
    mask_id: args.mask_id ?? null,
    scene_id: args.scene_id ?? "unknown",
  });
}

/** Absolute URL to download a generated report by filename. */
export function reportDownloadUrl(reportPath: string): string {
  const filename = reportPath.split("/").pop() ?? reportPath;
  return `${API_BASE}/report/download/${filename}`;
}

export async function health(): Promise<{ status: string; version: string }> {
  const res = await fetch(`${API_BASE}/health`);
  return handle(res);
}
