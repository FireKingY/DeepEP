# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepEP is a communication library for Mixture-of-Experts (MoE) and expert parallelism (EP). It provides high-throughput and low-latency all-to-all GPU kernels (MoE dispatch and combine) with support for FP8 low-precision operations.

**Key Features:**
- **Normal kernels**: High-throughput NVLink (intranode) and RDMA (internode) forwarding for training and inference prefilling
- **Low-latency kernels**: Pure RDMA for inference decoding with hook-based communication-computation overlapping (no SM occupation)
- **Data types**: BF16, FP8 (E4M3)

## Build System

### Dependencies
- **Required**: CUDA 12.3+ (SM90/Hopper) or CUDA 13.0+ (SM100/Blackwell), PyTorch 2.1+, Python 3.8+
- **Optional**: NVSHMEM 3.3.9+ (required for internode and low-latency features)
- **Legacy**: CUDA 11.0+ (SM80/A100) with `DISABLE_SM90_FEATURES=1`

### Build Commands

```bash
# Build with NVSHMEM (full features)
NVSHMEM_DIR=/path/to/nvshmem python setup.py build

# Build without NVSHMEM (intranode only)
python setup.py build

# Install
NVSHMEM_DIR=/path/to/nvshmem python setup.py install

# Create symbolic link for development (adjust SO name for your platform)
ln -sf build/lib.linux-x86_64-cpython-312/deep_ep_cpp.cpython-312-x86_64-linux-gnu.so .
```

### GPU Architecture Mapping

| GPU | Architecture | TORCH_CUDA_ARCH_LIST |
|-----|-------------|---------------------|
| A100 | SM80 | `8.0` |
| H100/H800 | SM90 | `9.0` |
| B200 | SM100 | `10.0` |

### Build Environment Variables
- `NVSHMEM_DIR`: Path to NVSHMEM (disables internode/low-latency if unset)
- `FORCE_DISABLE_NVSHMEM=1`: Force disable NVSHMEM even if installed (for intranode-only builds)
- `TORCH_CUDA_ARCH_LIST`: Target architectures (e.g., `"9.0"` for H100/H800, `"10.0"` for B200)
- `DISABLE_SM90_FEATURES=1`: Disable SM90 features (FP8, TMA) — required for CUDA 11 or A100
- `DISABLE_AGGRESSIVE_PTX_INSTRS=1`: Disable aggressive PTX instructions (default on non-SM90)
- `TOPK_IDX_BITS`: Set topk_idx dtype bits (32 or 64, default: 64)

### B200 CUDA Runtime Linking Caveat

On systems where multiple CUDA runtimes coexist (e.g. system-installed CUDA 12 alongside CUDA 13),
the extension `.so` can accidentally link against the older `libcudart.so.12`. The old runtime does
not recognise B200 and returns `multiProcessorCount=1`, causing a divide-by-zero SIGFPE in the
Buffer constructor.

**Fix:** pass `LDFLAGS` so the linker finds `libcudart.so.13` first:

```bash
LDFLAGS="-L/usr/local/cuda-13.1/lib64 -Wl,-rpath,/usr/local/cuda-13.1/lib64"
```

Verify after building:

```bash
ldd build/lib.*/deep_ep_cpp*.so | grep cudart
# Should show libcudart.so.13, NOT libcudart.so.12
```

### Current Dev Environment (this machine)

| Item | Value |
|------|-------|
| GPU | 8× NVIDIA B200 (SM100, 148 SMs) |
| Driver | 590.48.01 |
| CUDA Toolkit | 13.1 (`/usr/local/cuda-13.1`) |
| PyTorch | 2.9.1+cu130 |
| Python | 3.12.3 (venv at `/home/fire/workspace/venv`) |

### Example: Intranode-only Build for B200 (no NVSHMEM)

```bash
# 0. Environment setup (required for every shell session)
export PATH=/home/fire/workspace/venv/bin:/usr/local/cuda-13.1/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH
export LD_LIBRARY_PATH=/home/fire/workspace/venv/lib/python3.12/site-packages/nvidia/cu13/lib:/home/fire/workspace/venv/lib/python3.12/site-packages/cusparselt/lib:/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-13.1
export PYTHONPATH="/home/fire/workspace/DeepEP:$PYTHONPATH"

# 1. Clean and build (LDFLAGS is critical — see caveat above)
rm -rf build/
FORCE_DISABLE_NVSHMEM=1 \
  TORCH_CUDA_ARCH_LIST="10.0" \
  LDFLAGS="-L/usr/local/cuda-13.1/lib64 -Wl,-rpath,/usr/local/cuda-13.1/lib64" \
  python setup.py build

# 2. Create symlink for development
ln -sf build/lib.linux-x86_64-cpython-312/deep_ep_cpp.cpython-312-x86_64-linux-gnu.so .

# 3. Run intranode tests
python tests/test_intranode.py --num-processes 8
```

### Example: Build with NVSHMEM for B200 (intranode + low-latency)

Low-latency kernels require NVSHMEM even on single-node NVLink-only setups. IBGDA is
automatically disabled when there are no RDMA ranks (single-node), so no InfiniBand
hardware is needed.

```bash
# 0. Environment setup (same as above)
export PATH=/home/fire/workspace/venv/bin:/usr/local/cuda-13.1/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH
export LD_LIBRARY_PATH=/home/fire/workspace/venv/lib/python3.12/site-packages/nvidia/cu13/lib:/home/fire/workspace/venv/lib/python3.12/site-packages/cusparselt/lib:/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-13.1
export PYTHONPATH="/home/fire/workspace/DeepEP:$PYTHONPATH"

# 1. Install NVSHMEM (if not already installed)
pip install nvidia-nvshmem-cu12   # or set NVSHMEM_DIR manually

# 2. Clean and build (no FORCE_DISABLE_NVSHMEM)
rm -rf build/
TORCH_CUDA_ARCH_LIST="10.0" \
  LDFLAGS="-L/usr/local/cuda-13.1/lib64 -Wl,-rpath,/usr/local/cuda-13.1/lib64" \
  python setup.py build

# 3. Create symlink for development
ln -sf build/lib.linux-x86_64-cpython-312/deep_ep_cpp.cpython-312-x86_64-linux-gnu.so .

# 4. Run tests
python tests/test_intranode.py --num-processes 8
python tests/test_low_latency.py --num-processes 8
```

### Tuning & Benchmarking

```bash
# BF16/FP8 dispatch SM sweep (generates plots)
python tests/tune_bf16_sweep.py --num-processes 8   # → /tmp/tune_bf16_sweep.png
python tests/tune_fp8_sweep.py  --num-processes 8   # → /tmp/fp8_graph_all.png

# Re-plot without re-running sweep (if results JSON already exists)
python tests/tune_fp8_sweep.py --plot-only
```

## Code Formatting

```bash
# Format all changed Python/C++/CUDA files from main branch
bash format.sh

# Format all files in the repo
bash format.sh --all
```

Tools used:
- **yapf** (0.40.2): Python formatting (config in `pyproject.toml`)
- **ruff** (0.6.5): Python linting (config in `pyproject.toml`)
- **clang-format** (15.0.7): C++/CUDA formatting

Install formatters: `pip install -r requirements-lint.txt`

## Testing

Tests require multi-GPU setup (typically 8 GPUs per node for intranode, multiple nodes for internode).

```bash
# Intranode tests (single node, NVLink only)
python tests/test_intranode.py --num-processes 8

# Internode tests (multi-node, NVLink + RDMA, requires NVSHMEM build)
# Modify `init_dist` in tests/utils.py for your cluster settings
python tests/test_internode.py

# Low-latency tests (requires NVSHMEM build)
python tests/test_low_latency.py

# Tuning / benchmarking
python tests/tune_bf16_sweep.py --num-processes 8    # BF16 dispatch SM sweep
python tests/tune_bf16_dispatch.py --num-processes 8  # BF16 dispatch tuning
python tests/tune_fp8_sweep.py --num-processes 8      # FP8 dispatch SM sweep

# Sender implementation comparison
python tests/compare_sender_implementations.py
python tests/validate_sender_correctness.py
```

### Test Configuration
Edit `tests/utils.py:init_dist()` to configure cluster settings (MASTER_ADDR, MASTER_PORT, etc.).

## Architecture Overview

### Directory Structure

```
deep_ep/                  # Python package
├── __init__.py          # Exports Buffer, EventOverlap, Config, topk_idx_t
├── buffer.py            # Main Buffer class with dispatch/combine APIs
└── utils.py             # Helper functions (FP8 casting, benchmarking)

csrc/                     # C++/CUDA source
├── deep_ep.cpp          # Pybind11 bindings
├── deep_ep.hpp          # Buffer C++ class definition
├── config.hpp           # Config and LowLatencyBuffer structures
├── event.hpp            # CUDA event management
└── kernels/             # CUDA kernels
    ├── api.cuh          # Kernel APIs and type definitions
    ├── configs.cuh      # Tuning configurations
    ├── runtime.cu       # Kernel launchers
    ├── layout.cu        # Dispatch layout computation
    ├── intranode.cu     # NVLink-only kernels
    ├── internode.cu     # NVLink + RDMA kernels
    └── internode_ll.cu  # Low-latency RDMA kernels

tests/                    # Test suite & benchmarks
├── test_intranode.py
├── test_internode.py
├── test_low_latency.py
├── tune_bf16_dispatch.py
├── tune_bf16_sweep.py
├── tune_fp8_sweep.py
├── compare_sender_implementations.py
├── validate_sender_correctness.py
└── utils.py

third-party/              # NVSHMEM installation guide
```

### Key Abstractions

**Buffer**: Core communication buffer managing NVLink and RDMA memory.
- `dispatch()`: Send tokens to experts (all-to-all)
- `combine()`: Reduce tokens from experts (all-to-all)
- `get_dispatch_layout()`: Pre-compute communication patterns
- `low_latency_dispatch/combine()`: Low-latency variants for inference

**Config**: Tuning configuration for kernels
- `num_sms`: Number of SMs to use
- `num_max_nvl_chunked_send/recv_tokens`: NVLink chunk sizes
- `num_max_rdma_chunked_send/recv_tokens`: RDMA chunk sizes

**EventOverlap**: CUDA event management for async operations and communication-computation overlap.

### Kernel Types

1. **Intranode**: Uses NVLink only, operates within a single node (up to 8 GPUs)
2. **Internode**: Uses NVLink within node, RDMA between nodes (hierarchical)
3. **Low-latency**: Uses NVSHMEM for inference decoding with minimal latency. Supports both RDMA (multi-node, with IBGDA) and NVLink-only (single-node, IBGDA auto-disabled)

### Communication Patterns

**Normal kernels** (training/prefilling):
- Dispatch: Tokens → Experts (with topk_idx/topk_weights)
- Combine: Expert outputs → Original tokens (weighted reduction)
- Implicit CPU wait for GPU receive count (not CUDA-graph compatible unless `num_worst_tokens` specified)

**Low-latency kernels** (decoding):
- Hook-based receive: RDMA happens in background, `hook()` completes transfer without SM usage
- Double-buffering: Only 2 buffers, cannot hold more than 2 kernel results simultaneously
- CUDA-graph compatible

### Important Implementation Details

- **TopK index dtype**: `deep_ep.topk_idx_t` (typically `torch.int64`, configurable to `int32` via `TOPK_IDX_BITS`)
- **Buffer sizing**: Use `Buffer.get_low_latency_rdma_size_hint()` and `config.get_nvl/rdma_buffer_size_hint()` for proper sizing
- **SM control**: `Buffer.set_num_sms()` controls SM usage for normal kernels (default: 20, must be even)
- **NVSHMEM IBGDA**: Required for multi-node RDMA; auto-disabled on single-node NVLink-only setups. Configured via environment variables in `buffer.py`
- **Aggressive PTX**: Uses `ld.global.nc.L1::no_allocate.L2::256B` for volatile reads on Hopper; disable with `DISABLE_AGGRESSIVE_PTX_INSTRS=1` if compatibility issues arise

### Multi-Node Launch

Tests use `torch.multiprocessing.spawn` for single-node launching. For multi-node:
1. Set `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK` environment variables
2. Modify `init_dist()` in `tests/utils.py` for your cluster's network configuration
3. Launch with `torchrun` or `mpirun` across nodes
