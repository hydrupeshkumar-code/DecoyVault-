# API Schema Reference

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

---

## POST `/upload/`

Upload a LISS-IV satellite image in NumPy `.npy` format.

**Request**: `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | File | `.npy` float32 array `[H, W, 3]` ∈ [0,1] |

**Response** `200`

```json
{
  "file_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "scene_001.npy",
  "shape": [512, 512, 3],
  "size_bytes": 3145728,
  "message": "Upload successful"
}
```

**Errors**: `400` (not .npy), `422` (wrong shape)

---

## POST `/detect/`

Run cloud detection on an uploaded image.

**Request** `application/json`

```json
{
  "file_id": "550e8400-e29b-41d4-a716-446655440000",
  "threshold": 0.5
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file_id` | string | required | ID from `/upload/` |
| `threshold` | float | 0.5 | Cloud binarisation threshold [0,1] |

**Response** `200`

```json
{
  "file_id": "550e8400-e29b-41d4-a716-446655440000",
  "cloud_fraction": 0.342,
  "mask_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "message": "Detection complete"
}
```

---

## POST `/reconstruct/`

Remove clouds using the Restormer model.

**Request** `application/json`

```json
{
  "file_id": "550e8400-e29b-41d4-a716-446655440000",
  "mask_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "tile_size": 256,
  "overlap": 32,
  "preserve_clear": true
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file_id` | string | required | Cloudy image ID |
| `mask_id` | string | null | Optional cloud mask ID |
| `tile_size` | int | 256 | Tile size [64–1024] |
| `overlap` | int | 32 | Tile overlap [0–128] |
| `preserve_clear` | bool | true | Keep clear pixels unchanged |

**Response** `200`

```json
{
  "file_id": "550e8400-e29b-41d4-a716-446655440000",
  "result_id": "a3f4b2c1-d5e6-7890-abcd-ef1234567890",
  "elapsed_s": 2.743,
  "message": "Reconstruction complete"
}
```

---

## POST `/metrics/`

Compute quality metrics comparing a reconstruction to ground truth.

**Request** `application/json`

```json
{
  "pred_id": "a3f4b2c1-d5e6-7890-abcd-ef1234567890",
  "target_id": "bb8d3e21-f0a1-4b2c-9876-543210fedcba",
  "mask_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "cloudy_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Response** `200`

```json
{
  "pred_id": "a3f4b2c1-d5e6-7890-abcd-ef1234567890",
  "psnr_db": 32.41,
  "ssim": 0.917,
  "sam_rad": 0.083,
  "sam_deg": 4.76,
  "rmse": 0.031,
  "ndvi_mae_global": 0.042,
  "ndvi_mae_cloud": 0.061,
  "ndvi_improvement": 0.129
}
```

---

## POST `/report/generate`

Generate a PDF quality report.

**Request** `application/json`

```json
{
  "pred_id": "a3f4b2c1-d5e6-7890-abcd-ef1234567890",
  "target_id": "bb8d3e21-f0a1-4b2c-9876-543210fedcba",
  "mask_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "scene_id": "ROI_0042_Karnataka_2024"
}
```

**Response** `200`

```json
{
  "report_path": "/app/outputs/reports/a3f4b2c1_report.pdf",
  "format": "pdf",
  "message": "Report generated"
}
```

---

## GET `/report/download/{report_filename}`

Download a generated report file.

**Response**: Binary file stream (`application/octet-stream`)

---

## GET `/health`

Liveness check.

**Response** `200`

```json
{
  "status": "ok",
  "version": "2.0.0"
}
```
