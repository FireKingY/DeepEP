"""FP8 dispatch sweep: num_tokens × hidden × num_sms, plot best BW per num_sms per config."""
import argparse
import json
import torch
import torch.distributed as dist
import deep_ep
from utils import init_dist, bench, per_token_cast_to_fp8

RESULTS_FILE = '/tmp/tune_fp8_sweep_results.json'

SM_VALUES    = list(range(10, 149, 2))        # 10,12,...,148
CHUNK_VALUES = [4, 8, 16, 32, 64, 128]


def sweep_one_config(args_tuple, local_rank, num_ranks, rank, buffer):
    num_tokens, hidden, num_topk, num_experts = args_tuple

    if hidden % 128 != 0:
        if local_rank == 0:
            print(f'  Skipping H={hidden}: not divisible by 128 (required for FP8 scales)', flush=True)
        return None, 0

    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    x_fp8 = per_token_cast_to_fp8(x)
    # ensure scales are contiguous in the expected layout
    x_fp8 = (x_fp8[0], x_fp8[1].T.contiguous().T)

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1].to(deep_ep.topk_idx_t)
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')

    num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)

    # Measure recv bytes once with default config
    cfg0 = deep_ep.Config(24, 6, 256)
    result = buffer.dispatch(
        x=x_fp8, num_tokens_per_rank=num_tokens_per_rank, is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert, topk_idx=topk_idx,
        topk_weights=topk_weights, config=cfg0)
    recv_x_fp8, recv_x_scales = result[0]  # tuple for FP8
    nvl_recv_bytes = recv_x_fp8.numel() * 1 + recv_x_scales.numel() * 4  # fp8=1B, scale=4B

    if local_rank == 0:
        label = f'T={num_tokens//1024}k H={hidden}'
        print(f'\n--- {label}  recv_fp8={recv_x_fp8.shape}  recv_bytes={nvl_recv_bytes/1e9:.3f} GB ---',
              flush=True)

    best_bw_per_sms = {}

    for num_sms in SM_VALUES:
        try:
            cfg_layout = deep_ep.Config(num_sms, 4, 256)
            _, _, _, _, handle, _ = buffer.dispatch(
                x=x_fp8, num_tokens_per_rank=num_tokens_per_rank, is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=num_tokens_per_expert, topk_idx=topk_idx,
                topk_weights=topk_weights, config=cfg_layout)
        except Exception as e:
            if local_rank == 0:
                print(f'  SMs={num_sms}: handle failed ({e})', flush=True)
            continue

        best_bw, best_chunk = 0.0, None
        for chunk_size in CHUNK_VALUES:
            recv_buf = max(256, chunk_size * 4)
            if recv_buf <= chunk_size:
                continue
            try:
                cfg = deep_ep.Config(num_sms, chunk_size, recv_buf)
                tune_args = {'x': x_fp8, 'handle': handle, 'config': cfg}
                t = bench(lambda: buffer.dispatch(**tune_args), num_warmups=3, num_tests=10)[0]
                bw = nvl_recv_bytes / 1e9 / t
                if bw > best_bw:
                    best_bw, best_chunk = bw, chunk_size
            except Exception:
                pass

        best_bw_per_sms[num_sms] = (best_bw, best_chunk)
        if local_rank == 0:
            print(f'  SMs={num_sms:3d}: best={best_bw:7.2f} GB/s  chunk={best_chunk}', flush=True)

    del x, x_fp8, scores, topk_idx, topk_weights
    del num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank
    return best_bw_per_sms, nvl_recv_bytes


def tune_main(args, local_rank, num_ranks, rank, buffer, group):
    combos = [(t, h) for t in args.num_tokens_list for h in args.hidden_list]
    all_results = {}

    for num_tokens, hidden in combos:
        bw_per_sms, _ = sweep_one_config(
            (num_tokens, hidden, args.num_topk, args.num_experts),
            local_rank, num_ranks, rank, buffer)
        torch.cuda.empty_cache()
        if bw_per_sms is None:
            continue
        key = f'T={num_tokens//1024}k H={hidden}'
        all_results[key] = {str(k): list(v) for k, v in bw_per_sms.items()}

    if local_rank == 0:
        with open(RESULTS_FILE, 'w') as f:
            json.dump({'sm_values': SM_VALUES, 'results': all_results,
                       'nvl_peak': 900.0}, f, indent=2)
        print(f'\nResults saved to {RESULTS_FILE}', flush=True)


def tune_loop(local_rank, num_local_ranks, args):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    buffer = deep_ep.Buffer(group, int(4e9), 0, explicitly_destroy=True)
    torch.manual_seed(rank)
    tune_main(args, local_rank, num_ranks, rank, buffer, group)
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


def plot_all(results_file):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    with open(results_file) as f:
        data = json.load(f)
    sm_values = data['sm_values']
    results   = data['results']
    nvl_peak  = data['nvl_peak']

    def plot_subset(keys, title, outfile):
        keys = [k for k in keys if k in results]
        if not keys:
            print(f'No data for {outfile}, skipping')
            return
        fig, ax = plt.subplots(figsize=(11, 6))
        colors = cm.tab10(np.linspace(0, 1, len(keys)))
        for key, color in zip(keys, colors):
            bw_dict = results[key]
            bws    = [bw_dict[str(s)][0] if str(s) in bw_dict else float('nan') for s in sm_values]
            chunks = [bw_dict[str(s)][1] if str(s) in bw_dict else None          for s in sm_values]
            ax.plot(sm_values, bws, marker='o', markersize=4, linewidth=2, label=key, color=color)
            valid = [(i, b) for i, b in enumerate(bws) if b == b and b > 0]
            if valid:
                idx, peak = max(valid, key=lambda x: x[1])
                ax.annotate(f'{peak:.0f} GB/s\n(c={chunks[idx]})',
                            xy=(sm_values[idx], peak), xytext=(6, 4),
                            textcoords='offset points', fontsize=8, color=color,
                            arrowprops=dict(arrowstyle='-', color=color, lw=0.8))
        ax.axhline(nvl_peak,        color='red',  linestyle='--', linewidth=1.2,
                   label=f'NVLink peak ({nvl_peak:.0f} GB/s)')
        ax.axhline(nvl_peak * 0.84, color='gray', linestyle=':',  linewidth=1.0,
                   label=f'~84% ({nvl_peak*0.84:.0f} GB/s)')
        ax.set_xlabel('num_sms', fontsize=12)
        ax.set_ylabel('Best Bandwidth (GB/s)', fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(sm_values) + 4)
        ax.set_ylim(0, nvl_peak * 1.05)
        ax.set_xticks(range(0, max(sm_values) + 1, 16))
        fig.tight_layout()
        fig.savefig(outfile, dpi=150)
        print(f'Saved {outfile}')

    token_sizes = [4, 8, 16, 32, 64]
    hidden_sizes = [7168, 3584, 1024]

    # All combos overview
    all_keys = [f'T={t}k H={h}' for t in token_sizes for h in hidden_sizes]
    plot_subset(all_keys,
                'FP8 Dispatch: Best BW vs num_sms (all configs)\n(B200, NVLink 5.0, 8 GPUs)',
                '/tmp/fp8_graph_all.png')

    # Fixed token, vary hidden
    for t in token_sizes:
        keys = [f'T={t}k H={h}' for h in hidden_sizes]
        plot_subset(keys,
                    f'FP8 Dispatch: T={t}k — effect of hidden size\n(B200, NVLink 5.0, 8 GPUs)',
                    f'/tmp/fp8_graph_{t}k_hidden.png')

    # Fixed hidden, vary tokens
    for h in hidden_sizes:
        keys = [f'T={t}k H={h}' for t in token_sizes]
        plot_subset(keys,
                    f'FP8 Dispatch: H={h} — effect of num_tokens\n(B200, NVLink 5.0, 8 GPUs)',
                    f'/tmp/fp8_graph_{h}_tokens.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-tokens-list', type=int, nargs='+',
                        default=[4096, 8192, 16384, 32768, 65536])
    parser.add_argument('--hidden-list', type=int, nargs='+',
                        default=[7168, 3584, 1024])
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=256)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()

    if not args.plot_only:
        torch.multiprocessing.spawn(tune_loop, args=(args.num_processes, args),
                                    nprocs=args.num_processes)
    plot_all(RESULTS_FILE)
