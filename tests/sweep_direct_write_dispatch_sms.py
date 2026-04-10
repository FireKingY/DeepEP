"""Sweep origin vs direct-write dispatch bandwidth across num_sms values.

This script runs one 8-GPU distributed session, sweeps:

- seq_len in [1024, 4096, 16384]
- num_sms in [10, 12, ..., 60]

and renders one overlay plot with 6 lines:

- origin x [1k, 4k, 16k]
- direct_write x [1k, 4k, 16k]
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import deep_ep
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from matplotlib.lines import Line2D

from utils import bench, init_dist


def build_dispatch_inputs(num_tokens: int, hidden: int, num_topk: int, num_experts: int):
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1].to(deep_ep.topk_idx_t)
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')
    return x, topk_idx, topk_weights


def collect_gpu_snapshot():
    try:
        output = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-gpu=index,memory.used,memory.free,utilization.gpu',
                '--format=csv,noheader,nounits',
            ],
            text=True,
        )
        return [line.strip() for line in output.splitlines() if line.strip()]
    except Exception:
        return []


def save_json(path: str, payload):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2) + '\n')


def plot_results(json_path: str, png_path: str):
    with open(json_path) as f:
        payload = json.load(f)

    sm_values = payload['sm_values']
    seq_lens = payload['seq_lens']
    results = payload['results']

    fig, ax = plt.subplots(figsize=(12.5, 7))
    colors = cm.tab10(np.linspace(0, 1, len(seq_lens)))

    for color, seq_len in zip(colors, seq_lens):
        key = str(seq_len)
        seq_label = f'{seq_len // 1024}k'
        std_bws = [results[key][str(sm)]['standard_gbps'] for sm in sm_values]
        dw_bws = [results[key][str(sm)]['direct_write_gbps'] for sm in sm_values]

        ax.plot(
            sm_values,
            std_bws,
            marker='o',
            markersize=4,
            linewidth=2.0,
            linestyle='-',
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
        )
        ax.plot(
            sm_values,
            dw_bws,
            marker='o',
            markersize=4,
            linewidth=2.2,
            linestyle=(0, (7, 3)),
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=1.0,
        )

    ax.set_xlabel('num_sms', fontsize=12)
    ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax.set_title(
        'Direct-Write Dispatch vs Origin: Bandwidth vs num_sms\n'
        f'(8 GPUs, hidden={payload["hidden"]}, topk={payload["num_topk"]}, experts={payload["num_experts"]})',
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min(sm_values) - 1, max(sm_values) + 1)
    ax.set_xticks(sm_values[::2])

    legend_handles = [
        Line2D([0], [0], color=color, linewidth=2.2, marker='o', markersize=5, label=f'{seq_len // 1024}k')
        for color, seq_len in zip(colors, seq_lens)
    ]
    legend_handles.extend([
        Line2D([0], [0], color='black', linewidth=2.0, linestyle='-', label='origin'),
        Line2D([0], [0], color='black', linewidth=2.2, linestyle=(0, (7, 3)), label='direct_write'),
    ])
    ax.legend(
        handles=legend_handles,
        title='Color = seq_len, line style = implementation',
        fontsize=9,
        title_fontsize=10,
        ncol=1,
        loc='lower right',
    )

    out = Path(png_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def print_peak_summary(json_path: str):
    with open(json_path) as f:
        payload = json.load(f)

    sm_values = payload['sm_values']
    results = payload['results']

    print('', flush=True)
    print('Peak bandwidth per seq_len:', flush=True)
    for seq_len in payload['seq_lens']:
        key = str(seq_len)
        best_std_sm = max(sm_values, key=lambda sm: results[key][str(sm)]['standard_gbps'])
        best_dw_sm = max(sm_values, key=lambda sm: results[key][str(sm)]['direct_write_gbps'])
        best_std_bw = results[key][str(best_std_sm)]['standard_gbps']
        best_dw_bw = results[key][str(best_dw_sm)]['direct_write_gbps']
        print(
            f'  seq_len={seq_len:<5d} '
            f'origin_peak={best_std_bw:8.2f} GB/s @ SM={best_std_sm:<2d}   '
            f'direct_write_peak={best_dw_bw:8.2f} GB/s @ SM={best_dw_sm:<2d}',
            flush=True,
        )


def run_one_point(seq_len: int, config: deep_ep.Config, num_ranks: int, args,
                  buffer_std: deep_ep.Buffer, buffer_dw: deep_ep.Buffer,
                  x, topk_idx, topk_weights,
                  num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank,
                  ref_x, ref_topk_idx, ref_topk_weights):
    num_worst_tokens = seq_len * num_ranks

    buffer_dw.register_direct_write_layout(
        num_worst_tokens, args.hidden, args.num_topk, 0, 2, config=config)

    dispatch_args = {
        'x': x,
        'num_tokens_per_rank': num_tokens_per_rank,
        'is_token_in_rank': is_token_in_rank,
        'num_tokens_per_expert': num_tokens_per_expert,
        'topk_idx': topk_idx,
        'topk_weights': topk_weights,
        'config': config,
        'num_worst_tokens': num_worst_tokens,
    }

    actual_recv_tokens = ref_x.size(0)
    actual_payload_bytes = ref_x.numel() * ref_x.element_size()

    std_x, std_topk_idx, std_topk_weights, std_expert_list, _, _ = buffer_std.dispatch(**dispatch_args)
    standard_grid_size = buffer_std.runtime.get_last_dispatch_grid_size()
    dw_x, dw_topk_idx, dw_topk_weights, dw_expert_list, _, _ = buffer_dw.dispatch(**dispatch_args)
    direct_write_grid_size = buffer_dw.runtime.get_last_dispatch_grid_size()

    assert len(std_expert_list) == 0
    assert len(dw_expert_list) == 0
    assert std_x.size(0) == num_worst_tokens
    assert dw_x.size(0) == num_worst_tokens
    assert torch.equal(ref_x, std_x[:actual_recv_tokens])
    assert torch.equal(ref_x, dw_x[:actual_recv_tokens])
    assert torch.equal(ref_topk_idx, std_topk_idx[:actual_recv_tokens])
    assert torch.equal(ref_topk_idx, dw_topk_idx[:actual_recv_tokens])
    assert torch.equal(ref_topk_weights, std_topk_weights[:actual_recv_tokens])
    assert torch.equal(ref_topk_weights, dw_topk_weights[:actual_recv_tokens])
    assert torch.all(std_topk_idx[actual_recv_tokens:] == -1).item()
    assert torch.all(dw_topk_idx[actual_recv_tokens:] == -1).item()
    assert standard_grid_size == config.num_sms
    assert direct_write_grid_size == config.num_sms // 2

    dist.barrier()
    standard_result = bench(
        lambda: buffer_std.dispatch(**dispatch_args),
        num_warmups=args.warmups,
        num_tests=args.tests,
    )
    standard_grid_size = buffer_std.runtime.get_last_dispatch_grid_size()

    dist.barrier()
    direct_write_result = bench(
        lambda: buffer_dw.dispatch(**dispatch_args),
        num_warmups=args.warmups,
        num_tests=args.tests,
    )
    direct_write_grid_size = buffer_dw.runtime.get_last_dispatch_grid_size()

    std_avg = standard_result[0]
    dw_avg = direct_write_result[0]
    return {
        'actual_recv_tokens': actual_recv_tokens,
        'actual_payload_bytes': actual_payload_bytes,
        'standard_us': std_avg * 1e6,
        'standard_gbps': actual_payload_bytes / 1e9 / std_avg,
        'standard_grid_size': standard_grid_size,
        'direct_write_us': dw_avg * 1e6,
        'direct_write_gbps': actual_payload_bytes / 1e9 / dw_avg,
        'direct_write_grid_size': direct_write_grid_size,
        'speedup': std_avg / dw_avg,
    }


def sweep_main(args, local_rank: int, num_ranks: int, rank: int, group):
    sm_values = list(range(args.sm_start, args.sm_end + 1, args.sm_step))
    total_runs = len(args.seq_lens) * len(sm_values)
    run_idx = 0
    all_results = {str(seq_len): {} for seq_len in args.seq_lens} if local_rank == 0 else None

    for seq_idx, seq_len in enumerate(args.seq_lens):
        torch.manual_seed(args.seed + rank + seq_idx * 1000)

        num_worst_tokens = seq_len * num_ranks
        max_nvl_bytes = args.buffer_size_mib * 1024 * 1024
        for num_sms in sm_values:
            config = deep_ep.Config(num_sms, args.nvl_chunk_size, args.nvl_recv_tokens)
            std_hint = config.get_nvl_buffer_size_hint(args.hidden * 2, num_ranks)
            dw_hint = deep_ep.get_direct_write_nvl_size_hint(
                num_worst_tokens, args.hidden, args.num_topk, 0, 2, num_ranks, config)
            max_nvl_bytes = max(max_nvl_bytes, std_hint, dw_hint)

        buffer_std = deep_ep.Buffer(group, max_nvl_bytes, explicitly_destroy=True)
        buffer_dw = deep_ep.Buffer(group, max_nvl_bytes, explicitly_destroy=True)

        x, topk_idx, topk_weights = build_dispatch_inputs(seq_len, args.hidden, args.num_topk, args.num_experts)
        num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = \
            buffer_std.get_dispatch_layout(topk_idx, args.num_experts)

        ref_config = deep_ep.Config(sm_values[0], args.nvl_chunk_size, args.nvl_recv_tokens)
        ref_args = {
            'x': x,
            'num_tokens_per_rank': num_tokens_per_rank,
            'is_token_in_rank': is_token_in_rank,
            'num_tokens_per_expert': num_tokens_per_expert,
            'topk_idx': topk_idx,
            'topk_weights': topk_weights,
            'config': ref_config,
        }
        ref_x, ref_topk_idx, ref_topk_weights, _, _, _ = buffer_std.dispatch(**ref_args)

        for num_sms in sm_values:
            run_idx += 1
            if local_rank == 0:
                print(f'[{run_idx:02d}/{total_runs:02d}] seq_len={seq_len}, num_sms={num_sms}', flush=True)
            config = deep_ep.Config(num_sms, args.nvl_chunk_size, args.nvl_recv_tokens)
            point = run_one_point(
                seq_len, config, num_ranks, args,
                buffer_std, buffer_dw,
                x, topk_idx, topk_weights,
                num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank,
                ref_x, ref_topk_idx, ref_topk_weights)
            if local_rank == 0:
                all_results[str(seq_len)][str(num_sms)] = point
                print(
                    f'  origin={point["standard_gbps"]:.2f} GB/s, '
                    f'direct_write={point["direct_write_gbps"]:.2f} GB/s, '
                    f'speedup={point["speedup"]:.4f}x',
                    flush=True,
                )

        buffer_std.destroy()
        buffer_dw.destroy()
        dist.barrier()

    if local_rank == 0:
        payload = {
            'seq_lens': args.seq_lens,
            'sm_values': sm_values,
            'num_processes': args.num_processes,
            'hidden': args.hidden,
            'num_topk': args.num_topk,
            'num_experts': args.num_experts,
            'nvl_chunk_size': args.nvl_chunk_size,
            'nvl_recv_tokens': args.nvl_recv_tokens,
            'buffer_size_mib': args.buffer_size_mib,
            'warmups': args.warmups,
            'tests': args.tests,
            'seed': args.seed,
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'gpu_snapshot_before_run': args.gpu_snapshot_before_run,
            'results': all_results,
        }
        save_json(args.output_json, payload)


def sweep_loop(local_rank, num_local_ranks, args):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    try:
        sweep_main(args, local_rank, num_ranks, rank, group)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='Sweep direct-write dispatch bandwidth vs num_sms')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--seq-lens', type=int, nargs='+', default=[1024, 4096, 16384])
    parser.add_argument('--sm-start', type=int, default=10)
    parser.add_argument('--sm-end', type=int, default=60)
    parser.add_argument('--sm-step', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=256)
    parser.add_argument('--nvl-chunk-size', type=int, default=6)
    parser.add_argument('--nvl-recv-tokens', type=int, default=256)
    parser.add_argument('--buffer-size-mib', type=int, default=1024)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--tests', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-json', default='figures/direct_write_dispatch_sms_sweep_8gpu.json')
    parser.add_argument('--output-png', default='figures/direct_write_dispatch_sms_sweep_8gpu.png')
    args = parser.parse_args()

    assert args.num_processes == 8, 'this sweep is intended for 8-GPU intranode runs'
    assert args.num_experts % args.num_processes == 0
    assert args.sm_step > 0

    sm_values = list(range(args.sm_start, args.sm_end + 1, args.sm_step))
    assert sm_values, 'empty SM sweep'
    assert all(sm % 2 == 0 for sm in sm_values), 'all num_sms values must be even'

    repo_root = Path(__file__).resolve().parents[1]
    args.output_json = str((repo_root / args.output_json).resolve())
    args.output_png = str((repo_root / args.output_png).resolve())
    args.gpu_snapshot_before_run = collect_gpu_snapshot()

    torch.multiprocessing.spawn(sweep_loop, args=(args.num_processes, args), nprocs=args.num_processes)
    plot_results(args.output_json, args.output_png)
    print_peak_summary(args.output_json)
    print('', flush=True)
    print(f'Results JSON saved to {args.output_json}', flush=True)
    print(f'Plot saved to {args.output_png}', flush=True)


if __name__ == '__main__':
    main()
