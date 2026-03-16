# Intranode Dispatch Direct-Write Optimization

## Background

In the current intranode dispatch implementation, token data follows a 3-step path:
```
x (sender local memory) → ring_buffer (receiver's IPC shared memory) → recv_x (receiver local memory)
```

The ring buffer exists for flow control (producer-consumer head/tail protocol), but when `num_worst_tokens > 0`, the output tensor `recv_x` is pre-allocated with a fixed size, making the ring buffer unnecessary.

## Proposed Change

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
                    (cumulative tokens from ranks 0..sender_rank-1 to dst_rank)
                    (read from dst_rank's IPC buffer after notify_dispatch barrier)

     channel_offset = channel_prefix_matrix[dst_rank][channel - 1]
                    (tokens from current rank to dst_rank in channels 0..channel-1)
                    (local tensor, computed by notify_dispatch)

     token_seq_in_channel = running counter within the channel
                          (incremented for each token with is_token_in_rank=true)
   ```

3. **No receiver SM needed**:
   - Current: even SMs send, odd SMs receive and copy to recv_x
   - Optimized: only sender SMs needed, receiver SMs eliminated
   - SM count halved for the dispatch kernel

4. **Ring buffer metadata eliminated**:
   - No `channel_start_offset`, `channel_end_offset`, `head_idx`, `tail_idx`
   - No ring buffer data region (`channel_x_buffers`, `channel_src_idx_buffers`, etc.)
   - IPC buffer only needs `rank_prefix_matrix` at the head

5. **End-of-transfer synchronization**:
   - A barrier at the end of the dispatch kernel ensures all NVLink writes are visible
   - Or use a simple counter/flag mechanism

## Scope

- Only implement dispatch (not combine) in this first iteration
- Create a new git branch for this work
- Test logic should be consistent with `tests/test_intranode.py`
- Only the `num_worst_tokens > 0` path uses direct-write; the default path remains unchanged

## Expected Benefits

- Eliminate receiver-side local read + local write (save memory bandwidth)
- Halve SM usage for dispatch kernel
- Remove ring queue synchronization overhead (head/tail spin-waiting)
- Simpler code path for the `num_worst_tokens > 0` case
