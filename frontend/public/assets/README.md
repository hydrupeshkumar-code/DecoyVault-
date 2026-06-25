# Asset Placeholders

Place the following images here before running the frontend:

| Filename                  | Description                              |
|---------------------------|------------------------------------------|
| `liss4_cloudy.jpg`        | Cloud-covered LISS-IV satellite scene    |
| `liss4_reconstructed.jpg` | AI cloud-removed reconstruction output  |

Both should be the same spatial extent and resolution.
Recommended: 1920×1080 or 2560×1440 JPEG, ≤ 2 MB each.

The `BG_IMAGE_1` / `BG_IMAGE_2` constants in `HeroSection.tsx` point to these paths.
