#!/usr/bin/env bash
set -euo pipefail

mkdir -p checkpoints/scannet200 scripts/scannet200

# checkpoint
curl -L \
  https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/scannet200/scannet200_benchmark.ckpt \
  -o checkpoints/scannet200/scannet200_benchmark.ckpt

# matching config script
curl -L \
  https://raw.githubusercontent.com/JonasSchult/Mask3D/main/scripts/scannet200/scannet200_benchmark.sh \
  -o scripts/scannet200/scannet200_benchmark.sh

echo "Done."
echo "Checkpoint: checkpoints/scannet200/scannet200_benchmark.ckpt"
echo "Config:      scripts/scannet200/scannet200_benchmark.sh"