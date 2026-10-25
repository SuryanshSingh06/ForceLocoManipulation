#!/usr/bin/env bash
set -euo pipefail
source "$HOME/.holosoma_deps/miniconda3/etc/profile.d/conda.sh"
conda activate hsretargeting
lift_batch_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python "$lift_batch_dir/t2_holosoma_lift_batch.py" "$@"
