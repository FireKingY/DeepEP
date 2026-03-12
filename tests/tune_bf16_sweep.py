"""BF16 dispatch sweep: num_tokens × hidden × num_sms, plot best BW per num_sms per config."""
import argparse
import json
import os
import torch
import torch.distributed as dist
import deep_ep
from utils import init_dist, bench

RESULTS_FILE = '/tmp/tune_bf16_sweep_results.json'

SM_VALUES    = list(range(10, 149, 2))        # 10,12,...,148
CHUNK_VALUES = [4, 8, 16, 32, 64, 128]


def sweep_one_config(args_tuple, local_rank, num_ranks, rank, buffer):
    num_tokens, hidden, num_topk, num_experts = args_tuple

    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1].to(deep_ep.topk_idx_t)
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')

    num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)

    # Measure recv bytes once
    cfg0 = deep_ep.Config(24, 6, 256)
    recv_x, _, _, _, _, _ = buffer.dispatch(
        x=x, num_tokens_per_rank=num_tokens_per_rank, is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert, topk_idx=topk_idx,
        topk_weights=topk_weights, config=cfg0)
    nvl_recv_bytes = recv_x.numel() * 2

    if local_rank == 0:
        label = f'T={num_tokens//1024}k H={hidden}'
        print(f'\n--- {label}  recv={nvl_recv_bytes/1e9:.3f} GB ---', flush=True)

    # best_bw_per_sms[num_sms] = best BW over all chunk sizes
    best_bw_per_sms = {}

    for num_sms in SM_VALUES:
        try:
            cfg_layout = deep_ep.Config(num_sms, 4, 256)
            _, _, _, _, handle, _ = buffer.dispatch(
                x=x, num_tokens_per_rank=num_tokens_per_rank, is_token_in_rank=is_token_in_rank,
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
                tune_args = {'x': x, 'handle': handle, 'config': cfg}
                t = bench(lambda: buffer.dispatch(**tune_args), num_warmups=3, num_tests=10)[0]
                bw = nvl_recv_bytes / 1e9 / t
                if bw > best_bw:
                    best_bw, best_chunk = bw, chunk_size
            except Exception:
                pass

        best_bw_per_sms[num_sms] = (best_bw, best_chunk)
        if local_rank == 0:
            print(f'  SMs={num_sms:3d}: best={best_bw:7.2f} GB/s  chunk={best_chunk}', flush=True)

    return best_bw_per_sms, nvl_recv_bytes


def tune_main(args, local_rank, num_ranks, rank, buffer, group):
    combos = [(t, h) for t in args.num_tokens_list for h in args.hidden_list]

    all_results = {}  # key: "T=Xk H=Y" -> {sms: bw}

    for num_tokens, hidden in combos:
        bw_per_sms, _ = sweep_one_config(
            (num_tokens, hidden, args.num_topk, args.num_experts),
            local_rank, num_ranks, rank, buffer)
        key = f'T={num_tokens//1024}k H={hidden}'
        # store as {sms: [bw, best_chunk]}
        all_results[key] = {str(k): list(v) for k, v in bw_per_sms.items()}

    if local_rank == 0:
        import subprocess
        git_commit = 'unknown'
        try:
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            pass

        metadata = {
            'implementation': args.implementation,
            'git_commit': git_commit,
            'build_flags': ' '.join(
                f'{k}={v}' for k, v in [
                    ('FORCE_DISABLE_NVSHMEM', os.environ.get('FORCE_DISABLE_NVSHMEM')),
                    ('TORCH_CUDA_ARCH_LIST', os.environ.get('TORCH_CUDA_ARCH_LIST')),
                    ('NVSHMEM_DIR', os.environ.get('NVSHMEM_DIR')),
                    ('DISABLE_SM90_FEATURES', os.environ.get('DISABLE_SM90_FEATURES')),
                ] if v is not None
            ) or 'default',
            'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda,
            'num_tokens_list': args.num_tokens_list,
            'hidden_list': args.hidden_list,
            'num_topk': args.num_topk,
            'num_experts': args.num_experts,
            'sm_values': SM_VALUES,
            'chunk_values': CHUNK_VALUES,
            'num_warmups': 3,
            'num_tests': 10,
            'nvl_peak': 900.0,
        }

        with open(RESULTS_FILE, 'w') as f:
            json.dump({**metadata, 'results': all_results}, f, indent=2)
        print(f'\nResults saved to {RESULTS_FILE}', flush=True)


def tune_loop(local_rank, num_local_ranks, args):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    buffer = deep_ep.Buffer(group, int(8e9), 0, explicitly_destroy=True)
    torch.manual_seed(rank)
    tune_main(args, local_rank, num_ranks, rank, buffer, group)
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


def plot_results():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    with open(RESULTS_FILE) as f:
        data = json.load(f)

    sm_values = data['sm_values']
    results = data['results']
    nvl_peak = data['nvl_peak']

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = cm.tab20(np.linspace(0, 1, len(results)))
    for (label, bw_dict), color in zip(sorted(results.items()), colors):
        bws    = [bw_dict[str(s)][0] if str(s) in bw_dict else float('nan') for s in sm_values]
        chunks = [bw_dict[str(s)][1] if str(s) in bw_dict else None          for s in sm_values]
        line, = ax.plot(sm_values, bws, marker='o', markersize=3, linewidth=1.5,
                        label=label, color=color)
        # annotate best point with chunk size
        peak_bw = max((b for b in bws if not (b != b)), default=0)
        if peak_bw > 0:
            idx = bws.index(peak_bw)
            ax.annotate(f'c={chunks[idx]}',
                        xy=(sm_values[idx], peak_bw),
                        xytext=(4, 4), textcoords='offset points',
                        fontsize=6, color=color)

    ax.axhline(nvl_peak, color='red', linestyle='--', linewidth=1.2,
               label=f'NVLink peak ({nvl_peak:.0f} GB/s)')
    ax.axhline(nvl_peak * 0.84, color='gray', linestyle=':', linewidth=1.0,
               label=f'~84% NVLink ({nvl_peak*0.84:.0f} GB/s)')

    ax.set_xlabel('num_sms', fontsize=13)
    ax.set_ylabel('Best Bandwidth (GB/s)', fontsize=13)
    ax.set_title('BF16 Dispatch: Best BW vs num_sms\n(B200, NVLink 5.0, 8 GPUs, best over all chunk sizes)',
                 fontsize=13)
    ax.legend(fontsize=9, ncol=2, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(sm_values) + 2)
    ax.set_ylim(0, nvl_peak * 1.05)
    ax.set_xticks(range(0, max(sm_values) + 1, 8))

    out = '/tmp/tune_bf16_sweep.png'
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f'Graph saved to {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-tokens-list', type=int, nargs='+',
                        default=[4096, 8192, 16384, 32768, 65536])
    parser.add_argument('--hidden-list', type=int, nargs='+',
                        default=[7168, 3584, 1024])
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=256)
    parser.add_argument('--implementation', type=str, default='unknown',
                        help='Label for the sender implementation (e.g., ldst, tma)')
    args = parser.parse_args()

    torch.multiprocessing.spawn(tune_loop, args=(args.num_processes, args),
                                nprocs=args.num_processes)
    plot_results()
