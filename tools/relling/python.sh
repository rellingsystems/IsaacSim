#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Prefer this build's CUDA libraries over older system installations.
set -euo pipefail
relling_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
relling_build="$relling_root/_build/linux-x86_64/release"
if [[ ! -f "$relling_build/python.sh" ]]; then
    echo "Build the release checkout before running its Python launcher." >&2
    exit 1
fi
relling_cuda_paths=()
for relling_lib in "$relling_build"/exts/isaacsim.pip.nv/pip_prebundle/nvidia/*/lib; do
    [[ ! -d "$relling_lib" ]] || relling_cuda_paths+=("$relling_lib")
done
if [[ ${#relling_cuda_paths[@]} -eq 0 ]]; then
    echo "The build's NVIDIA runtime libraries are missing." >&2
    exit 1
fi
relling_cuda_path="$(IFS=:; echo "${relling_cuda_paths[*]}")"
export LD_LIBRARY_PATH="$relling_cuda_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec bash "$relling_build/python.sh" "$@"
