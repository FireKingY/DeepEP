# Intranode Dispatch Direct-Write Optimization

## Goal Description

Implement a direct-write path for intranode dispatch when `num_worst_tokens > 0`. Instead of the current 3-step data path (`x → ring_buffer → recv_x`), the sender writes directly to the receiver's output tensors via NVLink (`x → recv_x`), eliminating the ring buffer intermediate copy and receiver SMs entirely. Output tensors are allocated within the existing IPC shared buffer (`buffer_ptrs[rank]`), which is already cross-GPU accessible. This is dispatch-only (combine remains unchanged) and must produce combine-compatible handles.

## Acceptance Criteria

- AC-1: Direct-write dispatch produces correct results for BF16
  - Positive Tests:
    - `recv_x[:actual_recv]` from direct-write matches `recv_x[:actual_recv]` from standard dispatch for identical inputs
    - `recv_topk_idx[:actual_recv]` matches standard dispatch
    - `recv_topk_weights[:actual_recv]` matches standard dispatch
    - `recv_src_idx[:actual_recv]` matches standard dispatch
  - Negative Tests:
    - Passing `num_worst_tokens=0` does NOT trigger the direct-write path (falls back to ring buffer)
    - Passing mismatched registered shape (e.g., wrong hidden dim) raises an assertion error

- AC-2: Direct-write dispatch produces correct results for FP8 (E4M3)
  - Positive Tests:
    - FP8 `recv_x[:actual_recv]` and `recv_x_scales[:actual_recv]` match standard dispatch
  - Negative Tests:
    - FP8 dispatch with unregistered layout raises an error

- AC-3: Tail padding is correct when `actual_recv < num_worst_tokens`
  - Positive Tests:
    - `recv_topk_idx[actual_recv:]` is filled with `-1`
    - `recv_x.size(0) == num_worst_tokens`
  - Negative Tests:
    - If `actual_recv > num_worst_tokens`, kernel fails deterministically (assertion/trap), never silently writes OOB

- AC-4: Combine works correctly with handles produced by direct-write dispatch
  - Positive Tests:
    - `buffer.combine(expert_output, handle)` produces results matching standard dispatch+combine round-trip
    - `recv_topk_weights` from combine matches standard path
  - Negative Tests:
    - Corrupted `send_head` tensor causes combine timeout (not silent wrong results)

- AC-5: IPC buffer regions for dispatch outputs and combine scratch are disjoint
  - Positive Tests:
    - A dispatch followed by combine followed by another dispatch produces correct results (no aliasing corruption)
    - Repeated dispatch/combine cycles (5+ rounds) all produce correct results
  - Negative Tests:
    - Insufficient `num_nvl_bytes` for both direct-write slab + combine scratch raises assertion at registration time

- AC-6: Direct-write dispatch uses fewer SMs than standard dispatch
  - Positive Tests:
    - Kernel launch grid size is `num_sms / 2` (sender SMs only) instead of `num_sms` (sender + receiver)
  - Negative Tests:
    - Standard dispatch (num_worst_tokens=0) still uses `num_sms` blocks (no regression)

- AC-7: Registration API for direct-write layout
  - Positive Tests:
    - `buffer.register_direct_write_layout(num_worst_tokens, hidden, num_topk, num_scales, dtype)` succeeds and stores layout
    - Subsequent dispatches with matching params use direct-write path
  - Negative Tests:
    - Dispatch with `num_worst_tokens > 0` without prior registration falls back to standard ring buffer path (no crash)

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
Full direct-write dispatch implementation with BF16+FP8 support, combine compatibility, async mode support, registration API, repeated dispatch/combine cycle correctness, and comprehensive tests matching `test_intranode.py` coverage.

### Lower Bound (Minimum Acceptable Scope)
Direct-write dispatch for BF16 with `num_worst_tokens > 0`, combine compatibility for the non-cached handle path, basic registration, and a test that validates correctness against standard dispatch output.

### Allowed Choices
- Can use: existing IPC buffer (`buffer_ptrs`) as output storage, existing barrier primitives (`barrier_block`), existing `notify_dispatch` kernel unchanged, `torch::from_blob` for tensor views
- Can use: new kernel function in `intranode.cu` or a new file (e.g., `intranode_direct.cu`)
- Cannot use: `cudaDeviceEnablePeerAccess` (adds new dependency), per-dispatch IPC handle exchange (too expensive), modification to the default `num_worst_tokens=0` dispatch path

## Feasibility Hints and Suggestions

### Conceptual Approach

**IPC Buffer Layout (per rank):**
```
Region A (existing): rank_prefix_matrix + notify metadata
Region B (new, direct-write outputs):
  recv_x:            [num_worst_tokens × hidden × elem_size]
  recv_topk_idx:     [num_worst_tokens × num_topk × sizeof(topk_idx_t)]
  recv_topk_weights: [num_worst_tokens × num_topk × sizeof(float)]
  recv_src_idx:      [num_worst_tokens × sizeof(int)]
  recv_x_scales:     [num_worst_tokens × num_scales × sizeof(float)]
  recv_channel_offset: [num_ranks × num_channels × sizeof(int)]
Region C (existing, for combine): combine ring buffer scratch
```

**Registration flow:**
```
Python: buffer.register_direct_write_layout(num_worst_tokens, hidden, ...)
  → C++: compute offsets for Region B, validate fits in num_nvl_bytes alongside Region C
  → Store offsets in Buffer member variables
  → No cross-GPU exchange needed (all GPUs compute same offsets deterministically)
```

**Kernel pseudocode (sender SM only):**
```
// After notify_dispatch barrier completes:
rank_offset = (rank > 0) ? rank_prefix_matrix_on_dst[(rank-1)*R + dst_rank] : 0
channel_offset = (channel > 0) ? channel_prefix_matrix[dst_rank * C + channel - 1] : 0
running_count = 0

for token_idx in channel_range:
    if not is_token_in_rank[token_idx][dst_rank]:
        send_head[token_idx * R + dst_rank] = -1
        continue

    dst_offset = rank_offset + channel_offset + running_count
    send_head[token_idx * R + dst_rank] = running_count  // for combine compatibility

    // NVLink direct write to receiver's IPC buffer Region B
    dst_recv_x = buffer_ptrs[dst_rank] + recv_x_offset + dst_offset * hidden_int4
    UNROLLED_WARP_COPY(src_x[token_idx], dst_recv_x)

    // Same for topk_idx, topk_weights, src_idx, x_scales
    ...
    running_count++

// Write recv_channel_offset for combine
buffer_ptrs[dst_rank][recv_channel_offset_region + rank * C + channel] = channel_offset

// Final barrier to ensure all NVLink writes are visible
barrier_block(...)
```

**Return tensors:**
```cpp
auto recv_x = torch::from_blob(buffer_ptrs[rank] + recv_x_offset, {num_worst_tokens, hidden}, options);
```

### Relevant References
- `csrc/kernels/intranode.cu` — Current dispatch/combine kernels, `notify_dispatch`, barrier primitives
- `csrc/deep_ep.cpp` — `intranode_dispatch()` C++ binding, buffer size assertions, handle construction
- `csrc/deep_ep.hpp` — Buffer class definition, member variables for IPC buffer management
- `csrc/kernels/buffer.cuh` — Device-side Buffer abstraction (may not be needed if writing raw pointers)
- `csrc/kernels/utils.cuh` — `barrier_block`, `st_na_global`, `ld_nc_global`, `UNROLLED_WARP_COPY`
- `deep_ep/buffer.py` — Python dispatch/combine API, handle tuple structure
- `tests/test_intranode.py` — Test structure for `num_worst_tokens`, correctness validation pattern

## Dependencies and Sequence

### Milestones

1. **Milestone 1: Registration and IPC layout**
   - Design IPC buffer Region B layout with deterministic offset computation
   - Add `register_direct_write_layout()` C++ method and Python binding
   - Add byte-budget validation (Region B + Region C ≤ num_nvl_bytes)

2. **Milestone 2: Direct-write dispatch kernel**
   - Implement `dispatch_direct_write` kernel in `intranode.cu`
   - Sender-only SMs: compute `dst_offset`, write `recv_x`, `recv_topk_idx`, `recv_topk_weights`, `recv_src_idx`, `recv_x_scales` directly to `buffer_ptrs[dst_rank]`
   - Produce `send_head` with combine-compatible semantics
   - Write `recv_channel_offset` to receiver's IPC buffer
   - Final `barrier_block` for completion
   - Tail padding: fill `recv_topk_idx[actual_recv:]` with -1

3. **Milestone 3: C++ binding integration**
   - Add `intranode_dispatch_direct_write()` in `deep_ep.cpp`
   - Route from existing `intranode_dispatch()` when registered + `num_worst_tokens > 0`
   - Return `torch::from_blob` views over local IPC buffer
   - Construct combine-compatible handle tuple

4. **Milestone 4: Python API and testing**
   - Add `register_direct_write_layout()` to `deep_ep/buffer.py`
   - Create `tests/test_direct_write_dispatch.py` matching `test_intranode.py` logic
   - Validate BF16, FP8, combine round-trip, repeated cycles, async mode

Milestone 1 → Milestone 2 → Milestone 3 → Milestone 4 (sequential dependency)

## Task Breakdown

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Design IPC buffer Region B layout, compute offsets for all output tensors | AC-5, AC-7 | coding | - |
| task2 | Implement `register_direct_write_layout()` in C++ Buffer class with byte-budget validation | AC-5, AC-7 | coding | task1 |
| task3 | Add pybind11 binding for registration method | AC-7 | coding | task2 |
| task4 | Implement `dispatch_direct_write` kernel (BF16 path) with sender-only SMs | AC-1, AC-3, AC-6 | coding | task1 |
| task5 | Add FP8/x_scales support to direct-write kernel | AC-2 | coding | task4 |
| task6 | Implement `send_head` and `recv_channel_offset` production for combine compatibility | AC-4 | coding | task4 |
| task7 | Add `intranode_dispatch_direct_write()` C++ binding with routing logic | AC-1, AC-7 | coding | task2, task4, task5, task6 |
| task8 | Integrate into Python `dispatch()` with registration check and fallback | AC-7 | coding | task3, task7 |
| task9 | Create test script `test_direct_write_dispatch.py` — BF16 correctness | AC-1, AC-3 | coding | task8 |
| task10 | Add FP8 test cases | AC-2 | coding | task9 |
| task11 | Add combine round-trip and repeated cycle tests | AC-4, AC-5 | coding | task9 |
| task12 | Verify build + all tests pass on current branch | AC-1 through AC-7 | coding | task9, task10, task11 |

## Claude-Codex Deliberation

### Agreements
- Using `buffer_ptrs[dst_rank]` (IPC buffer) as remote destination is valid — already cross-GPU mapped and used for NVLink writes today
- Destination offset formula `rank_offset + channel_offset + running_count` is correct and matches current receiver derivation
- Preserving `send_head` + `recv_channel_offset` semantics is the right combine compatibility target
- Restricting to fixed-shape registered path is reasonable for `num_worst_tokens > 0` (already trades flexibility for graph-compatibility)

### Resolved Disagreements
- **IPC buffer aliasing (Codex REQUIRED_CHANGE)**: Codex flagged that combine reuses the same IPC buffer for ring queue scratch, risking aliasing with dispatch outputs. Resolution: partition IPC buffer into disjoint Region B (direct-write outputs) and Region C (combine scratch), with explicit byte-budget validation. Both Claude and Codex agree this resolves the issue.
- **Cross-rank completion (Codex REQUIRED_CHANGE)**: Codex flagged that removing receiver SMs removes local completion detection. Resolution: `barrier_block` across all ranks after sender writes guarantees all NVLink writes are visible. This is the same mechanism used in `notify_dispatch`. Both agree this is sufficient.
- **from_blob safety (Codex DISAGREE)**: Codex flagged returning `from_blob` views may be unsafe if IPC buffer is reused. Resolution: Region B is disjoint from combine scratch (Region C), so views are safe until the next direct-write dispatch overwrites them. This matches the usage pattern (dispatch → expert → combine → next dispatch).
- **Routing tightness (Codex REQUIRED_CHANGE)**: Codex requested explicit gating beyond just `num_worst_tokens > 0`. Resolution: route to direct-write only when: (a) intranode-only, (b) `num_worst_tokens > 0`, (c) layout registered, (d) shape/dtype match registration.

### Convergence Status
- Final Status: `converged`

## Pending User Decisions

- DEC-1: Single vs multiple registered layouts per Buffer
  - Claude Position: Single layout (one registration, one dtype/hidden/topk combo). Simplest implementation. Re-register to change.
  - Codex Position: Consider keyed registration for multiple layouts.
  - Tradeoff Summary: Single is simpler and sufficient for inference workloads (fixed model config). Multiple adds complexity for marginal benefit.
  - Decision Status: `Deferred — start with single, extend if needed`

- DEC-2: IPC-backed output tensors as public API guarantee
  - Claude Position: Internal detail. Users see same return types. Document that tensors are valid until next dispatch call.
  - Codex Position: Make it explicit if aliasing constraints exist.
  - Tradeoff Summary: Keeping it internal preserves API flexibility. A note in docstring about tensor lifetime is sufficient.
  - Decision Status: `Deferred — internal detail with documented lifetime`

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead (e.g., `direct_write_dispatch`, `register_layout`, not `milestone2_kernel`)

### Branch
- Work is done on branch `direct-write-dispatch` (already created and checked out)

### Key Constraints
- Default dispatch path (`num_worst_tokens=0`) must remain completely unchanged
- Combine kernel code is not modified — only dispatch produces combine-compatible handles
- IPC buffer size (`num_nvl_bytes`) must accommodate both direct-write Region B and combine Region C

---

## Original Design Draft

### Background

In the current intranode dispatch implementation, token data follows a 3-step path:
```
x (sender local memory) → ring_buffer (receiver's IPC shared memory) → recv_x (receiver local memory)
```

The ring buffer exists for flow control (producer-consumer head/tail protocol), but when `num_worst_tokens > 0`, the output tensor `recv_x` is pre-allocated with a fixed size, making the ring buffer unnecessary.

### Proposed Change

When `num_worst_tokens > 0`, eliminate the ring buffer intermediate step. The sender writes directly to the receiver's `recv_x`:
```
x (sender local memory) → recv_x (receiver local memory, via NVLink direct write)
```

### Key Design Points

1. **Pointer exchange (one-time, during initialization)**:
   - Pre-allocate `recv_x`, `recv_topk_idx`, `recv_topk_weights`, `recv_src_idx`, `recv_x_scales` with fixed size `num_worst_tokens`
   - Exchange these pointers across all ranks via IPC buffer (write to a fixed region in the IPC buffer during Buffer.sync() or a dedicated init method)
   - Pointers are stable and reused across all subsequent dispatch calls

2. **Destination offset computation (per-dispatch)**:
   After `notify_dispatch`, the sender can compute the exact write position in the receiver's `recv_x`:
   ```
   dst_offset = rank_offset + channel_offset + token_seq_in_channel

   where:
     rank_offset    = rank_prefix_matrix[sender_rank - 1][dst_rank]
     channel_offset = channel_prefix_matrix[dst_rank][channel - 1]
     token_seq_in_channel = running counter within the channel
   ```

3. **No receiver SM needed**: Only sender SMs, SM count halved
4. **Ring buffer metadata eliminated**: No head/tail, no ring buffer data region
5. **End-of-transfer synchronization**: Barrier at end of dispatch kernel

### Scope
- Only implement dispatch (not combine) in this first iteration
- Create a new git branch for this work
- Test logic should be consistent with `tests/test_intranode.py`
- Only the `num_worst_tokens > 0` path uses direct-write; the default path remains unchanged

### Expected Benefits
- Eliminate receiver-side local read + local write (save memory bandwidth)
- Halve SM usage for dispatch kernel
- Remove ring queue synchronization overhead (head/tail spin-waiting)
- Simpler code path for the `num_worst_tokens > 0` case
