#!/usr/bin/env bash
set -e

ME="third_party/MinkowskiEngine"

echo "Patching MinkowskiEngine for CUDA12 / PyTorch2..."

# 1. Comment out pip uninstall in setup.py
sed -i 's/^\([[:space:]]*\)run_command("pip", "uninstall", "MinkowskiEngine", "-y")/\1# run_command("pip", "uninstall", "MinkowskiEngine", "-y")/' "$ME/setup.py"

# 2. Replace deprecated thrust execution policy everywhere
find "$ME" \( -name "*.cu" -o -name "*.cuh" -o -name "*.hpp" -o -name "*.cpp" \) \
  -exec sed -i 's/thrust::device/thrust::cuda::par/g' {} +

# 3. Ensure all needed thrust headers exist in every CUDA file
for f in $(find "$ME/src" -name "*.cu"); do
  grep -q '#include <thrust/execution_policy.h>' "$f" || sed -i '1i #include <thrust/execution_policy.h>' "$f"
  grep -q '#include <thrust/sort.h>' "$f"             || sed -i '1i #include <thrust/sort.h>' "$f"
  grep -q '#include <thrust/remove.h>' "$f"           || sed -i '1i #include <thrust/remove.h>' "$f"
  grep -q '#include <thrust/unique.h>' "$f"           || sed -i '1i #include <thrust/unique.h>' "$f"
  grep -q '#include <thrust/reduce.h>' "$f"           || sed -i '1i #include <thrust/reduce.h>' "$f"
done

echo "Patch complete."