"""Comprehensive BF16 dispatch tuning: sweep num_sms × chunk_size."""
import argparse
import json
import torch
import torch.distributed as dist
import deep_ep
from utils import init_dist, bench


def tune_main(args, local_rank, num_ranks, rank, buffer, group):
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts

    # Random data
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1].to(deep_ep.topk_idx_t)
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')

    # Get layout tensors (num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert)
    num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)

    # Do one full dispatch to measure recv size (use default config)
    config0 = deep_ep.Config(24, 6, 256)
    recv_x, _, _, _, _, _ = buffer.dispatch(
        x=x,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        config=config0,
    )
    nvl_recv_bytes = recv_x.numel() * 2  # BF16 = 2 bytes

    if local_rank == 0:
        print(f'[config] num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}, '
              f'num_experts={num_experts}, num_ranks={num_ranks}')
        print(f'[info] recv_x shape={list(recv_x.shape)}, nvl_recv_bytes={nvl_recv_bytes/1e9:.3f} GB')
        print()

    # Sweep parameters
    sm_values = list(range(2, 85, 2))  # 2,4,...,84 (even; half of B200's ~168 SMs)
    chunk_values = [4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64]

    best_time, best_config = 1e10, None
    results = []

    for num_sms in sm_values:
        # Need fresh handle for each num_sms (channel_prefix_matrix depends on num_sms)
        try:
            config_for_layout = deep_ep.Config(num_sms, 6, 256)
            _, _, _, _, handle, _ = buffer.dispatch(
                x=x,
                num_tokens_per_rank=num_tokens_per_rank,
                is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=num_tokens_per_expert,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                config=config_for_layout,
            )
        except Exception as e:
            if local_rank == 0:
                print(f'SMs={num_sms:3d}: failed to get handle ({e})', flush=True)
            continue

        for chunk_size in chunk_values:
            recv_buf = max(256, chunk_size * 4)  # ring buffer must exceed chunk size
            try:
                config = deep_ep.Config(num_sms, chunk_size, recv_buf)
                tune_args = {'x': x, 'handle': handle, 'config': config}
                t = bench(lambda: buffer.dispatch(**tune_args), num_warmups=5, num_tests=20)[0]

                bw = nvl_recv_bytes / 1e9 / t
                if t < best_time:
                    best_time, best_config = t, (num_sms, chunk_size, recv_buf)

                results.append((num_sms, chunk_size, bw, t))
                if local_rank == 0:
                    print(f'SMs={num_sms:3d} chunk={chunk_size:3d} recv_buf={recv_buf:4d}: '
                          f'{bw:7.2f} GB/s, {t*1e6:8.2f} us', flush=True)
            except Exception as e:
                if local_rank == 0:
                    print(f'SMs={num_sms:3d} chunk={chunk_size:3d}: FAILED ({e})', flush=True)

    if local_rank == 0:
        print()
        print(f'=== BEST: SMs={best_config[0]}, chunk={best_config[1]}, recv_buf={best_config[2]}, '
              f'{nvl_recv_bytes/1e9/best_time:.2f} GB/s, {best_time*1e6:.2f} us ===')

        # Print top 10
        results.sort(key=lambda r: r[3])
        print('\nTop 10 configs:')
        for i, (sms, chunk, bw, t) in enumerate(results[:10]):
            print(f'  {i+1}. SMs={sms:3d} chunk={chunk:3d}: {bw:7.2f} GB/s, {t*1e6:8.2f} us')

        # Best BW per num_sms (max over all chunk sizes)
        from collections import defaultdict
        best_per_sms = defaultdict(float)
        for sms, chunk, bw, t in results:
            if bw > best_per_sms[sms]:
                best_per_sms[sms] = bw
        sms_sorted = sorted(best_per_sms.keys())

        print('\n=== BEST BW PER NUM_SMS ===')
        print('CSV: num_sms,best_bw_GBs')
        for sms in sms_sorted:
            print(f'CSV: {sms},{best_per_sms[sms]:.2f}')

        # Save JSON for plotting
        data = {'sm_values': sms_sorted,
                'best_bw': [best_per_sms[s] for s in sms_sorted],
                'nvl_peak_GBs': 900.0}
        with open('/tmp/tune_bf16_results.json', 'w') as f:
            json.dump(data, f)


def tune_loop(local_rank, num_local_ranks, args):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    buffer = deep_ep.Buffer(group, int(4e9), 0, explicitly_destroy=True)
    torch.manual_seed(rank)
    tune_main(args, local_rank, num_ranks, rank, buffer, group)
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-tokens', type=int, default=131072)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=256)
    args = parser.parse_args()
    torch.multiprocessing.spawn(tune_loop, args=(args.num_processes, args), nprocs=args.num_processes)
