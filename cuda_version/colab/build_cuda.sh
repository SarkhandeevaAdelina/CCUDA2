#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

if ! command -v cmake >/dev/null 2>&1; then
    echo "Installing CMake and build essentials..."
    sudo apt-get update
    sudo apt-get install -y cmake build-essential
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "CUDA Toolkit is required. In Google Colab, enable GPU via Runtime > Change runtime type."
    exit 1
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --config Release -j"$(nproc)"

echo ""
echo "Running CUDA diffraction solver..."
"${BUILD_DIR}/CudaDiffr"


