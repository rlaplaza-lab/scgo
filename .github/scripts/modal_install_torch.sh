#!/usr/bin/env bash
# Install CUDA torch for Modal GPU CI images (cu128 -> cu126 -> cu124).
set -euo pipefail

TORCH_SPEC="${TORCH_SPEC:-torch>=2.12.0,<2.13}"
PYPI_INDEX="${PYPI_INDEX:-https://pypi.org/simple}"

ok=0
for index in \
  "https://download.pytorch.org/whl/cu128" \
  "https://download.pytorch.org/whl/cu126" \
  "https://download.pytorch.org/whl/cu124"; do
  if pip install --no-cache-dir "${TORCH_SPEC}" \
    --index-url "${index}" \
    --extra-index-url "${PYPI_INDEX}"; then
    echo "Installed torch from ${index}"
    ok=1
    break
  fi
  echo "Torch install failed from ${index}; trying next CUDA index"
done

if [ "${ok}" != 1 ]; then
  echo "Failed to install torch from all CUDA indexes" >&2
  exit 1
fi
