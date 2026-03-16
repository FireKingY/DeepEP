import argparse
import os
import sys
import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep
from utils import init_dist, calc_diff, inplace_unique, per_token_cast_to_fp8, per_token_cast_back


def test_main(num_sms: int, local_rank: int, num_ranks: int, rank: int, buffer: deep_ep.Buffer,
              group: dist.ProcessGroup, num_tokens: int = 1024, hidden: int = 7168,
              num_topk: int = 8, num_experts: int = 256):
    assert num_experts % num_ranks == 0
    if local_rank == 0:
        print(f'[config] num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}, num_experts={num_experts}', flush=True)

    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * rank
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    x_e4m3 = per_token_cast_to_fp8(x) if deep_ep.Buffer.is_sm90_compiled() else None
    x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T) if x_e4m3 is not None else None
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='cuda') * rank
    topk_weights_pure_rand = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx = rank_idx.to(torch.int64)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    # Expert meta
    num_tokens_per_expert = torch.zeros((num_experts, ), dtype=torch.int, device='cuda')
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)

    # Rank layout meta
    num_tokens_per_rank = torch.empty((num_ranks, ), dtype=torch.int, device='cuda')
    token_idx_in_rank = torch.full((num_ranks, num_tokens), -1, dtype=torch.long, device='cuda')
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        token_sel = (rank_idx == i).max(dim=-1)[0]
        count = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
        tokens[:count] = torch.sort(tokens[:count])[0]
        token_idx_in_rank[i][tokens[:count]] = torch.arange(count, dtype=torch.long, device='cuda')
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = token_idx_in_rank >= 0
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)

    # Config
    nvl_buffer_size = 256
    config = deep_ep.Config(num_sms, 8, nvl_buffer_size)
    num_worst_tokens = num_tokens * num_ranks

    # Register direct-write layout for BF16
    num_scales_bf16 = 0
    buffer.register_direct_write_layout(num_worst_tokens, hidden, num_topk, num_scales_bf16, 2, config)
    assert buffer.is_direct_write_registered()

    # Helper to check data pattern
    def check_data(check_x, rank_prefix_matrix):
        assert torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1))
        check_start = 0
        for i in range(num_ranks):
            check_end = rank_prefix_matrix[i][rank].item()
            assert (check_x[check_start:check_end, :].int() - i).sum().item() == 0
            check_start = check_end

    # ========== BF16 Tests ==========
    if local_rank == 0:
        print('\n=== BF16 Direct-Write Dispatch Tests ===', flush=True)

    for async_mode in (False, True):
        for current_x in (x, x_pure_rand):
            if local_rank == 0:
                print(f'[testing] BF16 direct-write (async={async_mode}, rand={current_x is x_pure_rand}) ...', flush=True, end='')

            # Standard dispatch (reference)
            dispatch_args = {
                'x': current_x, 'num_tokens_per_rank': num_tokens_per_rank,
                'is_token_in_rank': is_token_in_rank, 'num_tokens_per_expert': num_tokens_per_expert,
                'topk_idx': topk_idx, 'topk_weights': topk_weights if current_x is x else topk_weights_pure_rand,
                'config': config, 'async_finish': async_mode,
            }
            ref_x, ref_topk_idx, ref_topk_weights, ref_expert_list, ref_handle, ref_event = buffer.dispatch(**dispatch_args)
            ref_event.current_stream_wait() if async_mode else ()

            # Direct-write dispatch
            dispatch_args['num_worst_tokens'] = num_worst_tokens
            dw_x, dw_topk_idx, dw_topk_weights, dw_expert_list, dw_handle, dw_event = buffer.dispatch(**dispatch_args)
            dw_event.current_stream_wait() if async_mode else ()

            # Verify results match
            actual_recv = ref_x.size(0)
            assert dw_x.size(0) == num_worst_tokens
            assert dw_topk_idx.size(0) == num_worst_tokens
            assert dw_topk_weights.size(0) == num_worst_tokens
            assert len(dw_expert_list) == 0

            assert torch.equal(ref_x, dw_x[:actual_recv]), f'recv_x mismatch'
            assert torch.equal(ref_topk_idx, dw_topk_idx[:actual_recv]), f'recv_topk_idx mismatch'

            # Check topk_weights for non-(-1) entries
            ref_tw_clone = ref_topk_weights.clone() if ref_topk_weights is not None else None
            if ref_tw_clone is not None:
                assert torch.equal(ref_tw_clone, dw_topk_weights[:actual_recv]), f'recv_topk_weights mismatch'

            # Tail padding check
            assert torch.all(dw_topk_idx[actual_recv:] == -1).item(), f'Tail topk_idx not padded with -1'

            # Handle metadata check: recv_channel_prefix_matrix and send_head
            # are validated indirectly via combine round-trip test below

            if local_rank == 0:
                print(' passed', flush=True)

    # ========== Combine Compatibility Tests ==========
    if local_rank == 0:
        print('\n=== Combine Compatibility Tests ===', flush=True)

    for current_x in (x, x_pure_rand):
        if local_rank == 0:
            print(f'[testing] Direct-write dispatch + standard combine (rand={current_x is x_pure_rand}) ...', flush=True, end='')

        # Standard dispatch+combine as reference
        dispatch_args = {
            'x': current_x, 'num_tokens_per_rank': num_tokens_per_rank,
            'is_token_in_rank': is_token_in_rank, 'num_tokens_per_expert': num_tokens_per_expert,
            'topk_idx': topk_idx, 'topk_weights': topk_weights if current_x is x else topk_weights_pure_rand,
            'config': config,
        }
        ref_recv_x, ref_topk_idx, ref_topk_weights, _, ref_handle, _ = buffer.dispatch(**dispatch_args)
        ref_combined_x, ref_combined_tw, _ = buffer.combine(
            x=ref_recv_x, handle=ref_handle,
            topk_weights=ref_topk_weights, config=config)

        # Direct-write dispatch then combine
        dispatch_args['num_worst_tokens'] = num_worst_tokens
        dw_recv_x, dw_topk_idx, dw_topk_weights, _, dw_handle, _ = buffer.dispatch(**dispatch_args)
        actual_recv = ref_recv_x.size(0)

        # Region B (direct-write outputs) is disjoint from combine scratch,
        # so no clone needed — combine won't corrupt the IPC-backed tensors.
        rank_pm, chan_pm, recv_chan_pm, src_idx, is_tir, send_hd = dw_handle
        sliced_handle = (rank_pm, chan_pm, recv_chan_pm, src_idx[:actual_recv], is_tir, send_hd)
        dw_combined_x, dw_combined_tw, _ = buffer.combine(
            x=dw_recv_x[:actual_recv], handle=sliced_handle,
            topk_weights=dw_topk_weights[:actual_recv], config=config)

        # Verify combine results match
        assert calc_diff(ref_combined_x.float(), dw_combined_x.float()) < 5e-6, \
            f'Combine result mismatch: diff={calc_diff(ref_combined_x.float(), dw_combined_x.float())}'
        if ref_combined_tw is not None:
            assert calc_diff(ref_combined_tw, dw_combined_tw) < 1e-9, 'Combine topk_weights mismatch'

        if local_rank == 0:
            print(' passed', flush=True)

    # ========== Repeated Cycle Tests ==========
    if local_rank == 0:
        print('\n=== Repeated Dispatch/Combine Cycle Tests ===', flush=True)
        print('[testing] 5 consecutive dispatch+combine cycles ...', flush=True, end='')

    for cycle in range(5):
        dispatch_args = {
            'x': x, 'num_tokens_per_rank': num_tokens_per_rank,
            'is_token_in_rank': is_token_in_rank, 'num_tokens_per_expert': num_tokens_per_expert,
            'topk_idx': topk_idx, 'topk_weights': topk_weights,
            'config': config, 'num_worst_tokens': num_worst_tokens,
        }
        dw_recv_x, dw_topk_idx, dw_topk_weights, _, dw_handle, _ = buffer.dispatch(**dispatch_args)
        actual_recv = gbl_num_tokens_per_rank[rank].item()
        rank_pm, chan_pm, recv_chan_pm, src_idx, is_tir, send_hd = dw_handle
        sliced_handle = (rank_pm, chan_pm, recv_chan_pm, src_idx[:actual_recv], is_tir, send_hd)
        dw_combined_x, _, _ = buffer.combine(
            x=dw_recv_x[:actual_recv], handle=sliced_handle,
            topk_weights=dw_topk_weights[:actual_recv], config=config)
        check_x = dw_combined_x.float() / is_token_in_rank.sum(dim=1).unsqueeze(1)
        assert calc_diff(check_x, x) < 5e-6, f'Cycle {cycle}: combine result mismatch'

    if local_rank == 0:
        print(' passed', flush=True)

    # ========== FP8 Tests ==========
    if x_e4m3 is not None:
        if local_rank == 0:
            print('\n=== FP8 Direct-Write Dispatch Tests ===', flush=True)

        # Re-register with FP8 layout
        num_scales_fp8 = hidden // 128
        buffer.register_direct_write_layout(num_worst_tokens, hidden, num_topk, num_scales_fp8, 1, config=config)

        if local_rank == 0:
            print('[testing] FP8 direct-write ...', flush=True, end='')

        # Standard FP8 dispatch (reference)
        dispatch_args = {
            'x': x_e4m3, 'num_tokens_per_rank': num_tokens_per_rank,
            'is_token_in_rank': is_token_in_rank, 'num_tokens_per_expert': num_tokens_per_expert,
            'topk_idx': topk_idx, 'topk_weights': topk_weights,
            'config': config,
        }
        ref_result = buffer.dispatch(**dispatch_args)
        ref_x_tuple, ref_topk_idx, ref_topk_weights, _, _, _ = ref_result
        ref_x_data, ref_x_scales = ref_x_tuple
        ref_x_bf16 = per_token_cast_back(ref_x_data, ref_x_scales)

        # Direct-write FP8 dispatch
        dispatch_args['num_worst_tokens'] = num_worst_tokens
        dw_result = buffer.dispatch(**dispatch_args)
        dw_x_tuple, dw_topk_idx, dw_topk_weights, _, _, _ = dw_result
        dw_x_data, dw_x_scales = dw_x_tuple
        dw_x_bf16 = per_token_cast_back(dw_x_data, dw_x_scales)
        actual_recv = ref_x_bf16.size(0)

        assert torch.equal(ref_x_bf16, dw_x_bf16[:actual_recv]), 'FP8 recv_x mismatch'
        assert torch.equal(ref_x_data, dw_x_data[:actual_recv]), 'FP8 raw recv_x mismatch'
        assert torch.equal(ref_x_scales, dw_x_scales[:actual_recv]), 'FP8 recv_x_scales mismatch'
        assert torch.equal(ref_topk_idx, dw_topk_idx[:actual_recv]), 'FP8 recv_topk_idx mismatch'
        assert torch.all(dw_topk_idx[actual_recv:] == -1).item(), 'FP8 tail not padded'

        if local_rank == 0:
            print(' passed', flush=True)

    # Re-register for BF16 before negative tests (FP8 test may have changed registration)
    buffer.register_direct_write_layout(num_worst_tokens, hidden, num_topk, 0, 2, config=config)

    # ========== Negative Path Tests ==========
    if local_rank == 0:
        print('\n=== Negative Path Tests ===', flush=True)

    # Test: num_worst_tokens=0 should NOT use direct-write (falls back to standard)
    if local_rank == 0:
        print('[testing] num_worst_tokens=0 fallback ...', flush=True, end='')
    dispatch_args = {
        'x': x, 'num_tokens_per_rank': num_tokens_per_rank,
        'is_token_in_rank': is_token_in_rank, 'num_tokens_per_expert': num_tokens_per_expert,
        'topk_idx': topk_idx, 'topk_weights': topk_weights,
        'config': config, 'num_worst_tokens': 0,
    }
    ref_x, ref_topk_idx, ref_topk_weights, ref_expert_list, _, _ = buffer.dispatch(**dispatch_args)
    assert ref_x.size(0) == gbl_num_tokens_per_rank[rank].item(), 'num_worst_tokens=0 should use standard path'
    assert len(ref_expert_list) > 0, 'Standard path should return expert list'
    if local_rank == 0:
        print(' passed', flush=True)

    # Test: insufficient buffer size should fail registration
    if local_rank == 0:
        print('[testing] Insufficient buffer size at registration ...', flush=True, end='')
    tiny_buffer = deep_ep.Buffer(group, 1024)  # 1 KiB -- way too small
    try:
        tiny_buffer.register_direct_write_layout(num_worst_tokens, hidden, num_topk, 0, 2, config=config)
        assert False, 'Should have raised assertion error'
    except RuntimeError:
        pass
    if local_rank == 0:
        print(' passed', flush=True)

    # Test: mismatched hidden size falls back to standard ring-buffer dispatch
    if local_rank == 0:
        print('[testing] Mismatched hidden size falls back to standard path ...', flush=True, end='')
    wrong_hidden = hidden // 2
    x_wrong = torch.randn((num_tokens, wrong_hidden), dtype=torch.bfloat16, device='cuda')
    # With layout mismatch, Python routing falls back to standard dispatch (no crash)
    result = buffer.dispatch(x=x_wrong, num_tokens_per_rank=num_tokens_per_rank,
                             is_token_in_rank=is_token_in_rank, num_tokens_per_expert=num_tokens_per_expert,
                             topk_idx=topk_idx, topk_weights=topk_weights,
                             config=config, num_worst_tokens=num_worst_tokens)
    # Standard path with num_worst_tokens returns worst-case size
    assert result[0].size(0) == num_worst_tokens, 'Mismatched layout should use standard path'
    if local_rank == 0:
        print(' passed', flush=True)

    # Test: FP8 dispatch on unregistered buffer should use standard path (no crash)
    if local_rank == 0:
        print('[testing] FP8 dispatch on unregistered buffer uses standard path ...', flush=True, end='')
    unreg_buffer = deep_ep.Buffer(group, int(1024 * 1024 * 1024))
    if x_e4m3 is not None:
        # With num_worst_tokens > 0 but no registration, should fall back to standard ring buffer
        fp8_result = unreg_buffer.dispatch(
            x=x_e4m3, num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank, num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx, topk_weights=topk_weights,
            config=config, num_worst_tokens=num_worst_tokens)
        fp8_recv_x = fp8_result[0]
        # Should succeed via standard path and return worst-case size
        assert fp8_recv_x[0].size(0) == num_worst_tokens, 'Unregistered FP8 should use standard path'
    if local_rank == 0:
        print(' passed', flush=True)

    # Test: SM-count regression guard — direct-write uses num_sms/2 blocks, standard uses num_sms
    if local_rank == 0:
        print('[testing] SM-count: verify launch grid sizes ...', flush=True, end='')
    # Standard dispatch: launches config.num_sms blocks (sender+receiver pairs)
    buffer.dispatch(x=x, num_tokens_per_rank=num_tokens_per_rank,
                    is_token_in_rank=is_token_in_rank, num_tokens_per_expert=num_tokens_per_expert,
                    topk_idx=topk_idx, topk_weights=topk_weights,
                    config=config, num_worst_tokens=0)
    std_grid = buffer.runtime.get_last_dispatch_grid_size()
    assert std_grid == config.num_sms, f'Standard dispatch grid should be {config.num_sms}, got {std_grid}'
    # Direct-write dispatch: launches config.num_sms/2 blocks (sender-only)
    buffer.dispatch(x=x, num_tokens_per_rank=num_tokens_per_rank,
                    is_token_in_rank=is_token_in_rank, num_tokens_per_expert=num_tokens_per_expert,
                    topk_idx=topk_idx, topk_weights=topk_weights,
                    config=config, num_worst_tokens=num_worst_tokens)
    dw_grid = buffer.runtime.get_last_dispatch_grid_size()
    assert dw_grid == config.num_sms // 2, f'Direct-write grid should be {config.num_sms // 2}, got {dw_grid}'
    if local_rank == 0:
        print(' passed', flush=True)

    # Test: corrupted send_head causes deterministic failure (subprocess test)
    if local_rank == 0:
        print('[testing] Corrupted send_head causes process crash (subprocess) ...', flush=True, end='')
    if local_rank == 0:
        import subprocess
        helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_corrupt_send_head_helper.py')
        env = os.environ.copy()
        env['MASTER_PORT'] = '8399'  # Use different port to avoid conflict
        env['PYTHONPATH'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + \
            ((':' + env['PYTHONPATH']) if 'PYTHONPATH' in env else '')
        result = subprocess.run(
            [sys.executable, helper],
            env=env, timeout=180, capture_output=True, text=True)
        combined_output = result.stdout + result.stderr
        # 1. Verify the helper reached the corrupted combine call (not a bootstrap failure)
        assert 'REACHED_CORRUPTED_COMBINE' in combined_output, \
            f'Helper did not reach corrupted combine path. Bootstrap failure?\n' \
            f'returncode={result.returncode}\nstderr: {result.stderr[:1000]}'
        # 2. Verify the process crashed (non-zero exit from kernel trap/timeout)
        assert result.returncode != 0, \
            f'Corrupted send_head should crash, but process exited with 0.\nstderr: {result.stderr[:500]}'
        # 3. Verify it was not a silent success (no "ERROR: combine did not fail" message)
        assert 'ERROR: combine did not fail' not in combined_output, \
            'Helper reached past combine without crashing — send_head corruption was not detected'
        print(' passed', flush=True)
    group.barrier()

    if local_rank == 0:
        print('\n=== All Direct-Write Dispatch Tests Passed ===\n', flush=True)


def test_handle_parity(num_sms: int, local_rank: int, num_ranks: int, rank: int,
                       buffer_dw: deep_ep.Buffer, buffer_std: deep_ep.Buffer,
                       group: dist.ProcessGroup, config,
                       num_tokens: int = 1024, hidden: int = 7168,
                       num_topk: int = 8, num_experts: int = 256):
    """Compare handle metadata between direct-write and standard dispatch paths."""
    if local_rank == 0:
        print('=== Handle Parity Tests ===', flush=True)

    assert num_experts % num_ranks == 0
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1].to(deep_ep.topk_idx_t)
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')

    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx = rank_idx.to(torch.int64)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device='cuda')
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device='cuda')
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
    token_idx_in_rank = torch.full((num_ranks, num_tokens), -1, dtype=torch.long, device='cuda')
    for i in range(num_ranks):
        token_sel = (rank_idx == i).max(dim=-1)[0]
        count = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
        tokens[:count] = torch.sort(tokens[:count])[0]
        token_idx_in_rank[i][tokens[:count]] = torch.arange(count, dtype=torch.long, device='cuda')
    is_token_in_rank = (token_idx_in_rank.T.contiguous().to(torch.int) >= 0)

    num_worst_tokens = num_tokens * num_ranks

    # Standard dispatch with num_worst_tokens (no registration on buffer_std)
    if local_rank == 0:
        print('[testing] Handle parity: send_head, recv_channel_prefix_matrix, recv_src_idx ...', flush=True, end='')

    std_args = {
        'x': x, 'num_tokens_per_rank': num_tokens_per_rank,
        'is_token_in_rank': is_token_in_rank, 'num_tokens_per_expert': num_tokens_per_expert,
        'topk_idx': topk_idx, 'topk_weights': topk_weights,
        'config': config, 'num_worst_tokens': num_worst_tokens,
    }
    _, _, _, _, std_handle, _ = buffer_std.dispatch(**std_args)

    # Direct-write dispatch
    dw_args = dict(std_args)
    _, _, _, _, dw_handle, _ = buffer_dw.dispatch(**dw_args)

    # Compare handle metadata
    std_rpm, std_cpm, std_rcpm, std_src, std_itr, std_sh = std_handle
    dw_rpm, dw_cpm, dw_rcpm, dw_src, dw_itr, dw_sh = dw_handle

    assert torch.equal(std_sh, dw_sh), 'send_head mismatch'
    assert torch.equal(std_rcpm, dw_rcpm), 'recv_channel_prefix_matrix mismatch'

    # recv_src_idx: compare up to actual received count
    actual_recv = std_rpm[-1][rank].item()
    assert torch.equal(std_src[:actual_recv], dw_src[:actual_recv]), 'recv_src_idx mismatch'

    if local_rank == 0:
        print(' passed', flush=True)


# noinspection PyShadowingNames
def main(args: argparse.Namespace, local_rank: int, num_ranks: int, rank: int):
    torch.cuda.set_device(local_rank)
    rank, world_size, group = init_dist(local_rank, num_ranks)

    num_sms = args.num_sms if args.num_sms > 0 else 20

    # Use larger buffer to accommodate disjoint direct-write region
    num_nvl_bytes = int(args.buffer_size * 1024 * 1024)
    buffer = deep_ep.Buffer(group, num_nvl_bytes)

    test_main(num_sms, local_rank, num_ranks, rank, buffer, group,
              num_tokens=args.num_tokens, hidden=args.hidden,
              num_topk=args.num_topk, num_experts=args.num_experts)

    # Handle parity test: re-register for BF16 and use a separate unregistered buffer for reference
    config = deep_ep.Config(num_sms, 8, 256)
    buffer.register_direct_write_layout(args.num_tokens * args.num_processes, args.hidden, args.num_topk, 0, 2, config=config)
    buffer_std = deep_ep.Buffer(group, num_nvl_bytes)
    test_handle_parity(num_sms, local_rank, num_ranks, rank, buffer, buffer_std, group, config,
                       num_tokens=args.num_tokens, hidden=args.hidden,
                       num_topk=args.num_topk, num_experts=args.num_experts)

    if local_rank == 0:
        print('Success!', flush=True)


def spawn_main(local_rank, args):
    main(args, local_rank, args.num_processes, local_rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-tokens', type=int, default=1024)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=256)
    parser.add_argument('--num-sms', type=int, default=20)
    parser.add_argument('--buffer-size', type=int, default=1024, help='NVL buffer size in MiB')
    parser.add_argument('--num-processes', type=int, default=8)
    args = parser.parse_args()

    torch.multiprocessing.spawn(spawn_main, args=(args,), nprocs=args.num_processes)
