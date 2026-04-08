import argparse
import json
import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep
from utils import init_dist, bench


def build_dispatch_inputs(num_tokens: int, hidden: int, num_topk: int, num_experts: int):
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1].to(deep_ep.topk_idx_t)
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')
    return x, topk_idx, topk_weights


def build_summary(args, num_ranks: int, num_worst_tokens: int, std_hint: int, dw_hint: int, num_nvl_bytes: int,
                  actual_recv_tokens: int, actual_payload_bytes: int, standard_result: tuple,
                  direct_write_result: tuple, standard_grid_size: int, direct_write_grid_size: int):
    std_avg, std_min, std_max = standard_result
    dw_avg, dw_min, dw_max = direct_write_result
    std_bw = actual_payload_bytes / 1e9 / std_avg
    dw_bw = actual_payload_bytes / 1e9 / dw_avg
    return {
        'config': {
            'num_processes': args.num_processes,
            'num_ranks': num_ranks,
            'num_tokens': args.num_tokens,
            'hidden': args.hidden,
            'num_topk': args.num_topk,
            'num_experts': args.num_experts,
            'num_sms': args.num_sms,
            'nvl_chunk_size': args.nvl_chunk_size,
            'nvl_recv_tokens': args.nvl_recv_tokens,
            'num_worst_tokens': num_worst_tokens,
            'warmups': args.warmups,
            'tests': args.tests,
            'seed': args.seed,
        },
        'buffer': {
            'buffer_size_mib': args.buffer_size_mib,
            'std_hint_bytes': std_hint,
            'direct_write_hint_bytes': dw_hint,
            'allocated_nvl_bytes': num_nvl_bytes,
        },
        'payload': {
            'actual_recv_tokens': actual_recv_tokens,
            'actual_payload_bytes': actual_payload_bytes,
        },
        'standard': {
            'avg_s': std_avg,
            'min_s': std_min,
            'max_s': std_max,
            'avg_us': std_avg * 1e6,
            'min_us': std_min * 1e6,
            'max_us': std_max * 1e6,
            'effective_gbps': std_bw,
            'grid_size': standard_grid_size,
        },
        'direct_write': {
            'avg_s': dw_avg,
            'min_s': dw_min,
            'max_s': dw_max,
            'avg_us': dw_avg * 1e6,
            'min_us': dw_min * 1e6,
            'max_us': dw_max * 1e6,
            'effective_gbps': dw_bw,
            'grid_size': direct_write_grid_size,
        },
        'speedup_direct_write_vs_standard': std_avg / dw_avg,
    }


def print_summary(summary):
    config = summary['config']
    buffer = summary['buffer']
    payload = summary['payload']
    standard = summary['standard']
    direct_write = summary['direct_write']

    print(f'[config] num_tokens={config["num_tokens"]}, hidden={config["hidden"]}, '
          f'num_topk={config["num_topk"]}, num_experts={config["num_experts"]}, '
          f'num_ranks={config["num_ranks"]}', flush=True)
    print(f'[config] num_sms={config["num_sms"]}, nvl_chunk_size={config["nvl_chunk_size"]}, '
          f'nvl_recv_tokens={config["nvl_recv_tokens"]}, num_worst_tokens={config["num_worst_tokens"]}', flush=True)
    print(f'[buffer] std_hint={buffer["std_hint_bytes"]}, direct_write_hint={buffer["direct_write_hint_bytes"]}, '
          f'allocated_nvl_bytes={buffer["allocated_nvl_bytes"]}', flush=True)
    print(f'[payload] actual_recv_tokens={payload["actual_recv_tokens"]}, '
          f'actual_payload_bytes={payload["actual_payload_bytes"]}', flush=True)
    print('', flush=True)

    print('mode           avg_us      min_us      max_us   eff_GB/s  grid_size', flush=True)
    print('------------------------------------------------------------------', flush=True)
    print(f'standard     {standard["avg_us"]:9.2f} {standard["min_us"]:11.2f} {standard["max_us"]:11.2f} '
          f'{standard["effective_gbps"]:10.2f} {standard["grid_size"]:10d}', flush=True)
    print(f'direct_write {direct_write["avg_us"]:9.2f} {direct_write["min_us"]:11.2f} {direct_write["max_us"]:11.2f} '
          f'{direct_write["effective_gbps"]:10.2f} {direct_write["grid_size"]:10d}', flush=True)
    print('', flush=True)
    print(f'speedup_direct_write_vs_standard={summary["speedup_direct_write_vs_standard"]:.4f}x', flush=True)


def benchmark_main(args, local_rank: int, num_ranks: int, rank: int, group):
    assert args.num_sms % 2 == 0, 'num_sms must be even'
    assert args.num_experts % num_ranks == 0, 'num_experts must be divisible by num_ranks'

    torch.manual_seed(args.seed + rank)

    num_worst_tokens = args.num_worst_tokens if args.num_worst_tokens > 0 else args.num_tokens * num_ranks
    config = deep_ep.Config(args.num_sms, args.nvl_chunk_size, args.nvl_recv_tokens)
    std_hint = config.get_nvl_buffer_size_hint(args.hidden * 2, num_ranks)
    dw_hint = deep_ep.get_direct_write_nvl_size_hint(
        num_worst_tokens, args.hidden, args.num_topk, 0, 2, num_ranks, config)
    num_nvl_bytes = max(args.buffer_size_mib * 1024 * 1024, std_hint, dw_hint)

    buffer_std = deep_ep.Buffer(group, num_nvl_bytes, explicitly_destroy=True)
    buffer_dw = deep_ep.Buffer(group, num_nvl_bytes, explicitly_destroy=True)
    buffer_dw.register_direct_write_layout(num_worst_tokens, args.hidden, args.num_topk, 0, 2, config=config)
    assert buffer_dw.is_direct_write_registered()

    x, topk_idx, topk_weights = build_dispatch_inputs(args.num_tokens, args.hidden, args.num_topk, args.num_experts)
    num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = \
        buffer_std.get_dispatch_layout(topk_idx, args.num_experts)

    dispatch_args = {
        'x': x,
        'num_tokens_per_rank': num_tokens_per_rank,
        'is_token_in_rank': is_token_in_rank,
        'num_tokens_per_expert': num_tokens_per_expert,
        'topk_idx': topk_idx,
        'topk_weights': topk_weights,
        'config': config,
    }

    ref_x, ref_topk_idx, ref_topk_weights, _, _, _ = buffer_std.dispatch(**dispatch_args)
    actual_recv_tokens = ref_x.size(0)
    actual_payload_bytes = ref_x.numel() * ref_x.element_size()

    worst_case_args = dict(dispatch_args)
    worst_case_args['num_worst_tokens'] = num_worst_tokens

    std_x, std_topk_idx, std_topk_weights, std_expert_list, _, _ = buffer_std.dispatch(**worst_case_args)
    standard_grid_size = buffer_std.runtime.get_last_dispatch_grid_size()
    dw_x, dw_topk_idx, dw_topk_weights, dw_expert_list, _, _ = buffer_dw.dispatch(**worst_case_args)
    direct_write_grid_size = buffer_dw.runtime.get_last_dispatch_grid_size()

    assert len(std_expert_list) == 0
    assert len(dw_expert_list) == 0
    assert std_x.size(0) == num_worst_tokens
    assert dw_x.size(0) == num_worst_tokens
    assert std_topk_idx.size(0) == num_worst_tokens
    assert dw_topk_idx.size(0) == num_worst_tokens
    assert std_topk_weights.size(0) == num_worst_tokens
    assert dw_topk_weights.size(0) == num_worst_tokens
    assert torch.equal(ref_x, std_x[:actual_recv_tokens])
    assert torch.equal(ref_x, dw_x[:actual_recv_tokens])
    assert torch.equal(ref_topk_idx, std_topk_idx[:actual_recv_tokens])
    assert torch.equal(ref_topk_idx, dw_topk_idx[:actual_recv_tokens])
    assert torch.equal(ref_topk_weights, std_topk_weights[:actual_recv_tokens])
    assert torch.equal(ref_topk_weights, dw_topk_weights[:actual_recv_tokens])
    assert torch.all(std_topk_idx[actual_recv_tokens:] == -1).item()
    assert torch.all(dw_topk_idx[actual_recv_tokens:] == -1).item()
    assert standard_grid_size == args.num_sms
    assert direct_write_grid_size == args.num_sms // 2

    dist.barrier()
    standard_result = bench(lambda: buffer_std.dispatch(**worst_case_args), num_warmups=args.warmups, num_tests=args.tests)
    standard_grid_size = buffer_std.runtime.get_last_dispatch_grid_size()
    dist.barrier()
    direct_write_result = bench(lambda: buffer_dw.dispatch(**worst_case_args), num_warmups=args.warmups, num_tests=args.tests)
    direct_write_grid_size = buffer_dw.runtime.get_last_dispatch_grid_size()

    summary = build_summary(args, num_ranks, num_worst_tokens, std_hint, dw_hint, num_nvl_bytes, actual_recv_tokens,
                            actual_payload_bytes, standard_result, direct_write_result, standard_grid_size,
                            direct_write_grid_size)

    if local_rank == 0:
        if args.json:
            print(json.dumps(summary, sort_keys=True), flush=True)
        else:
            print_summary(summary)

    buffer_std.destroy()
    buffer_dw.destroy()


def benchmark_loop(local_rank: int, num_local_ranks: int, args):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    try:
        benchmark_main(args, local_rank, num_ranks, rank, group)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark direct-write dispatch vs standard dispatch')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-tokens', type=int, default=131072)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=256)
    parser.add_argument('--num-sms', type=int, default=24)
    parser.add_argument('--nvl-chunk-size', type=int, default=6)
    parser.add_argument('--nvl-recv-tokens', type=int, default=256)
    parser.add_argument('--num-worst-tokens', type=int, default=0)
    parser.add_argument('--buffer-size-mib', type=int, default=1024)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--tests', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    torch.multiprocessing.spawn(benchmark_loop, args=(args.num_processes, args), nprocs=args.num_processes)
