#!/usr/bin/env bash
set -euo pipefail
source "$HOME/.holosoma_deps/miniconda3/etc/profile.d/conda.sh"
conda activate hoi-retarget
lift_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python "$lift_script_dir/t2_batch_retarget.py" \
  --manifest "$HOME/holosoma/t1/run05/manifest.json" \
  --repo "$HOME/hoi-retarget" \
  --out "$HOME/holosoma/t2/hoi_all_lift_candidates01" \
  --include-pending \
  --limit 0 \
  "$@"
