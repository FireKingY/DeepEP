# Direct-Write Dispatch Bench/Test Guide

This note records how to reproduce the `direct-write` correctness tests and benchmarks for DeepEP, based on the `benchmark-direct-write-dispatch` session and the runs I verified on `2026-04-08`.

## Scope

Relevant scripts:

- `tests/test_direct_write_dispatch.py`
- `tests/bench_direct_write_dispatch.py`
- `tests/utils.py`

What they do:

- `test_direct_write_dispatch.py` checks BF16/FP8 correctness, combine compatibility, fallback behavior, SM-count behavior, and handle parity.
- `bench_direct_write_dispatch.py` benchmarks standard dispatch vs direct-write dispatch, and does a preflight correctness check before timing.
- Both scripts use `torch.multiprocessing.spawn(...)` internally. Do not wrap them with `torchrun`.

## Verified Environment

The commands below were verified in this repo on branch `direct-write-dispatch` with:

- GPU: `NVIDIA B200`
- Driver: `590.48.01`
- CUDA toolkit: `/usr/local/cuda-13.1`
- Python: `3.12.13`
- PyTorch: `2.10.0+cu130`
- Extension runtime CUDA: `libcudart.so.13`
- Virtualenv: `/home/hao.xingyan/workspace/.venv-vllm-sm100f`

Sanity checks:

```bash
nvidia-smi -L

source /home/hao.xingyan/workspace/.venv-vllm-sm100f/bin/activate
export PATH=/home/hao.xingyan/workspace/.venv-vllm-sm100f/bin:/usr/local/cuda-13.1/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH
export LD_LIBRARY_PATH=/home/hao.xingyan/workspace/.venv-vllm-sm100f/lib/python3.12/site-packages/nvidia/cu13/lib:/home/hao.xingyan/workspace/.venv-vllm-sm100f/lib/python3.12/site-packages/cusparselt/lib:/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-13.1
export PYTHONPATH=/home/hao.xingyan/workspace/DeepEP:$PYTHONPATH

python -c "import sys, torch, deep_ep; print(sys.executable); print(sys.version.split()[0]); print(torch.__version__); print(torch.version.cuda); print(deep_ep.Buffer.is_sm90_compiled())"

ldd build/lib.linux-x86_64-cpython-312/deep_ep_cpp.cpython-312-x86_64-linux-gnu.so | rg cudart
```

Expected key points:

- `deep_ep` imports successfully.
- `torch.version.cuda` prints `13.0`.
- `deep_ep.Buffer.is_sm90_compiled()` prints `True`.
- `ldd` shows `libcudart.so.13`, not `libcudart.so.12`.

## Required Shell Setup

Use this at the start of every shell session:

```bash
source /home/hao.xingyan/workspace/.venv-vllm-sm100f/bin/activate

export PATH=/home/hao.xingyan/workspace/.venv-vllm-sm100f/bin:/usr/local/cuda-13.1/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH
export LD_LIBRARY_PATH=/home/hao.xingyan/workspace/.venv-vllm-sm100f/lib/python3.12/site-packages/nvidia/cu13/lib:/home/hao.xingyan/workspace/.venv-vllm-sm100f/lib/python3.12/site-packages/cusparselt/lib:/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-13.1
export PYTHONPATH=/home/hao.xingyan/workspace/DeepEP:$PYTHONPATH

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8361
unset WORLD_SIZE
unset RANK
```

Notes:

- `MASTER_PORT` must be unique if another distributed job is already using `8361`.
- For single-node runs, leave `WORLD_SIZE` and `RANK` unset.
- If you want to pin to a GPU subset, set `CUDA_VISIBLE_DEVICES` and make `--num-processes` match the number of visible GPUs.

Example:

```bash
export CUDA_VISIBLE_DEVICES=0,3,5,7
python tests/test_direct_write_dispatch.py --num-processes 4
```

## How The Scripts Interpret Your Config

Important behavior from the scripts:

- `--num-processes` must match the number of visible GPUs.
- `--num-experts` must be divisible by `--num-processes`.
- `--num-sms` must be even.
- In `bench_direct_write_dispatch.py`, if `--num-worst-tokens 0`, it becomes `num_tokens * num_ranks`.
- The benchmark allocates two buffers per rank:
  - one for standard dispatch
  - one for direct-write dispatch
- Actual allocated NVLink buffer size is:
  - `max(buffer_size_mib, std_hint, direct_write_hint)`

This last point is the main memory trap. Lowering `--buffer-size-mib` does not force a smaller allocation if the script-computed hints are larger.

## Correctness Test Commands

### Full local correctness test

Use this when you want the whole direct-write validation suite:

```bash
python tests/test_direct_write_dispatch.py --num-processes 8
```

If some GPUs are busy, use a clean subset instead:

```bash
export CUDA_VISIBLE_DEVICES=0,3,5,7
python tests/test_direct_write_dispatch.py --num-processes 4
```

What it covers:

- BF16 direct-write vs standard dispatch
- FP8 direct-write vs standard dispatch
- direct-write dispatch followed by standard combine
- repeated dispatch/combine cycles
- fallback paths and negative cases
- launch grid-size checks
- handle parity

### Verified test run

I ran:

```bash
python tests/test_direct_write_dispatch.py --num-processes 4
```

Observed result:

- Test completed successfully with `Success!`
- All correctness sections passed, including handle parity

Observed warnings at shutdown:

- `destroy_process_group() was not called before program exit`
- `TCPStore ... Connection was likely closed`

These warnings happened after success was already printed. They are teardown warnings, not test failures.

## Benchmark Commands

### Clean 4-GPU benchmark

This is the best command on a busy shared box, because it pins the run to a known idle subset:

```bash
export CUDA_VISIBLE_DEVICES=0,3,5,7
python tests/bench_direct_write_dispatch.py --num-processes 4 --json
```

Verified result:

```json
{
  "config": {
    "num_processes": 4,
    "num_tokens": 131072,
    "hidden": 7168,
    "num_topk": 8,
    "num_experts": 256,
    "num_sms": 24,
    "num_worst_tokens": 524288
  },
  "standard": {
    "avg_us": 23460.7461628161,
    "effective_gbps": 289.475942191636,
    "grid_size": 24
  },
  "direct_write": {
    "avg_us": 16614.615390175266,
    "effective_gbps": 408.7558718943273,
    "grid_size": 12
  },
  "speedup_direct_write_vs_standard": 1.4120547248231314
}
```

Interpretation:

- direct-write was about `1.41x` faster than standard dispatch
- grid size matched the expected `24 -> 12`

### 8-GPU benchmark on a shared machine

If all 8 GPUs are not fully idle, reduce `--num-tokens` first:

```bash
python tests/bench_direct_write_dispatch.py --num-processes 8 --num-tokens 16384 --json
```

Verified result:

```json
{
  "config": {
    "num_processes": 8,
    "num_tokens": 16384,
    "hidden": 7168,
    "num_topk": 8,
    "num_experts": 256,
    "num_sms": 24,
    "num_worst_tokens": 131072
  },
  "standard": {
    "avg_us": 3351.38696118405,
    "effective_gbps": 371.1829586997321,
    "grid_size": 24
  },
  "direct_write": {
    "avg_us": 2796.781464626914,
    "effective_gbps": 444.7890347292274,
    "grid_size": 12
  },
  "speedup_direct_write_vs_standard": 1.1983013344344797
}
```

Interpretation:

- direct-write was about `1.20x` faster
- this run is useful as a shared-box sanity benchmark
- do not treat it as the cleanest performance number if other GPU jobs are active

### 8-GPU default benchmark

Use this only if all 8 GPUs have plenty of free memory:

```bash
python tests/bench_direct_write_dispatch.py --num-processes 8 --json
```

On this machine, with `num_tokens=131072`, the script computes:

- `std_hint ~= 384.19 MiB`
- `direct_write_hint ~= 14774.38 MiB`

Because the benchmark allocates both a standard buffer and a direct-write buffer, that means roughly:

- `~29.5 GiB` of NVLink buffers per rank before counting input tensors and other allocations

This is why the default 8-GPU run failed with `cuda out of memory` on a GPU that only had about `16.5 GiB` free.

## Recommended Workflow

If you just want a reliable answer quickly:

1. Use a clean GPU subset.
2. Run the correctness test first.
3. Run the benchmark with `--json`.
4. If you need 8-GPU numbers, first make sure all 8 GPUs are actually idle.

Minimal sequence:

```bash
source /home/hao.xingyan/workspace/.venv-vllm-sm100f/bin/activate
export PATH=/home/hao.xingyan/workspace/.venv-vllm-sm100f/bin:/usr/local/cuda-13.1/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH
export LD_LIBRARY_PATH=/home/hao.xingyan/workspace/.venv-vllm-sm100f/lib/python3.12/site-packages/nvidia/cu13/lib:/home/hao.xingyan/workspace/.venv-vllm-sm100f/lib/python3.12/site-packages/cusparselt/lib:/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-13.1
export PYTHONPATH=/home/hao.xingyan/workspace/DeepEP:$PYTHONPATH
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8361
export CUDA_VISIBLE_DEVICES=0,3,5,7

python tests/test_direct_write_dispatch.py --num-processes 4
python tests/bench_direct_write_dispatch.py --num-processes 4 --json
```

## Common Failure Modes

### `ModuleNotFoundError: No module named 'torch'`

Cause:

- wrong Python environment

Fix:

- activate `/home/hao.xingyan/workspace/.venv-vllm-sm100f`

### `NCCL error ... cuda out of memory`

Cause:

- not enough free GPU memory for `std_hint` plus `direct_write_hint`
- often triggered by the default 8-GPU benchmark on a shared box

Fix:

- reduce `--num-tokens`
- reduce `--num-processes`
- pin to an idle GPU subset with `CUDA_VISIBLE_DEVICES`

### Port collision / process-group init failure

Cause:

- another local job is already using the same `MASTER_PORT`

Fix:

- choose a different `MASTER_PORT`, for example `8391`

### Benchmark numbers look inconsistent

Cause:

- the machine is shared and some visible GPUs are already running other jobs

Fix:

- run on an idle GPU subset
- or wait until all requested GPUs are free

## Bottom Line

Recommended command on this machine today:

```bash
export CUDA_VISIBLE_DEVICES=0,3,5,7
python tests/test_direct_write_dispatch.py --num-processes 4
python tests/bench_direct_write_dispatch.py --num-processes 4 --json
```

If you need the full 8-GPU benchmark, make the GPUs idle first. Otherwise, reduce `--num-tokens`, because the default direct-write buffer requirement is large enough to OOM even on B200 when some GPUs are already occupied.
