#!/usr/bin/env bash
#
# Dataset bootstrap for ChaturVyuha CloudVision AI cloud-removal training.
#
# Scaffolds the dataset directory tree, fetches the small smoke-test dataset
# (RICE) automatically, and prints exact next-step commands for the large /
# login-gated datasets (SEN12MS-CR, Bhoonidhi LISS-IV) that cannot be pulled
# non-interactively.
#
# Usage:
#   bash download_datasets.sh           # scaffold + fetch RICE (smoke set)
#   bash download_datasets.sh --dirs    # scaffold directories only
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW="${ROOT}/datasets/raw"

echo "==> Scaffolding dataset directories under ${ROOT}/datasets"
mkdir -p "${ROOT}/datasets/sen12ms_cr/"{cloudy,clear,masks}
mkdir -p "${ROOT}/datasets/liss4/"{cloudy,clear,masks}
mkdir -p "${ROOT}/datasets/rice2/"{cloudy,clear,masks}
mkdir -p "${RAW}"

if [[ "${1:-}" == "--dirs" ]]; then
  echo "==> Directories created. Skipping downloads (--dirs)."
  exit 0
fi

# --- RICE (smoke test): small, public GitHub repo, ships masks for RICE2 ------
if command -v git >/dev/null 2>&1; then
  if [[ ! -d "${RAW}/RICE_DATASET" ]]; then
    echo "==> Cloning RICE dataset (smoke-test set) ..."
    git clone --depth 1 https://github.com/BUPTLdy/RICE_DATASET.git "${RAW}/RICE_DATASET" \
      || echo "!! RICE clone failed (offline?). Fetch it manually: https://github.com/BUPTLdy/RICE_DATASET"
  else
    echo "==> RICE already present at ${RAW}/RICE_DATASET"
  fi
else
  echo "!! git not found — skipping RICE clone. Install git or download manually."
fi

cat <<EOF

================================================================================
NEXT STEPS
================================================================================

A) SMOKE SET (RICE2 — thick clouds, ships masks).  Prepare it:

   python -m ai.dataset_tools.prepare_dataset \\
       --cloudy-dir datasets/raw/RICE_DATASET/RICE2/cloud \\
       --clear-dir  datasets/raw/RICE_DATASET/RICE2/label \\
       --mask-dir   datasets/raw/RICE_DATASET/RICE2/mask  \\
       --output     datasets/rice2 --normalize auto --match order

   (Folder names inside RICE2 vary by release — point --cloudy-dir/--clear-dir
    at the cloudy and ground-truth folders you actually see.)

B) PRIMARY TRAINING SET (SEN12MS-CR — real Sentinel-2, correct G/R/NIR bands).
   Large + needs a browser/account, so download manually:

     https://patricktum.github.io/cloud_removal/sen12mscr/
     -> MediaTUM: https://mediatum.ub.tum.de/1554803

   Put the cloudy tiles and cloud-free tiles in two folders, then:

   python -m ai.dataset_tools.prepare_dataset \\
       --cloudy-dir datasets/raw/SEN12MSCR/s2_cloudy \\
       --clear-dir  datasets/raw/SEN12MSCR/s2_cloudfree \\
       --output     datasets/sen12ms_cr \\
       --bands 2,3,7 --normalize scale --scale 10000 \\
       --match order --max 4000

C) LISS-IV FINE-TUNE (Bhoonidhi — real Resourcesat-2, the ISRO differentiator).
   Free account + manual scene selection:

     https://bhoonidhi.nrsc.gov.in/

   Prepare temporally-close clear/cloudy scenes into datasets/liss4 the same way
   (LISS-IV is already 3-band Green/Red/NIR, so: --normalize percentile).

Then: audit -> smoke-train -> full train.  See TRAINING.md for the full sequence.
================================================================================
EOF
