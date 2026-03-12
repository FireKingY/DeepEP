"""Compare ld/st vs TMA sender BF16 dispatch performance.

Loads two JSON result files (one per implementation), generates overlay
comparison plots and a summary delta table. Reads metadata from the
archived JSON files rather than querying the current environment.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def load_results(path):
    with open(path) as f:
        return json.load(f)


def get_peak_for_config(bw_dict, sm_values):
    """Return (peak_bw, best_sms, best_chunk) for a config's bw_dict."""
    peak_bw, best_sms, best_chunk = 0.0, None, None
    for s in sm_values:
        key = str(s)
        if key in bw_dict:
            bw, chunk = bw_dict[key]
            if bw > peak_bw:
                peak_bw, best_sms, best_chunk = bw, s, chunk
    return peak_bw, best_sms, best_chunk


def plot_overlay(ldst_data, tma_data, keys, title, outfile, nvl_peak=900.0):
    """Plot overlay of ld/st (solid) and TMA (dashed) on same axes."""
    sm_values = ldst_data['sm_values']
    ldst_results = ldst_data['results']
    tma_results = tma_data['results']

    keys = [k for k in keys if k in ldst_results and k in tma_results]
    if not keys:
        print(f'No matching data for {outfile}, skipping')
        return

    fig, ax = plt.subplots(figsize=(13, 7))
    colors = cm.tab10(np.linspace(0, 1, max(len(keys), 1)))

    for key, color in zip(keys, colors):
        # ld/st: solid line
        ldst_bw = ldst_results[key]
        bws_ldst = [ldst_bw[str(s)][0] if str(s) in ldst_bw else float('nan') for s in sm_values]
        ax.plot(sm_values, bws_ldst, marker='o', markersize=3, linewidth=1.8,
                label=f'{key} (ld/st)', color=color, linestyle='-')

        # TMA: dashed line
        tma_bw = tma_results[key]
        bws_tma = [tma_bw[str(s)][0] if str(s) in tma_bw else float('nan') for s in sm_values]
        ax.plot(sm_values, bws_tma, marker='s', markersize=3, linewidth=1.8,
                label=f'{key} (TMA)', color=color, linestyle='--')

    ax.axhline(nvl_peak, color='red', linestyle='--', linewidth=1.2,
               label=f'NVLink peak ({nvl_peak:.0f} GB/s)')
    ax.axhline(nvl_peak * 0.84, color='gray', linestyle=':', linewidth=1.0,
               label=f'~84% ({nvl_peak * 0.84:.0f} GB/s)')

    ax.set_xlabel('num_sms', fontsize=12)
    ax.set_ylabel('Best Bandwidth (GB/s)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8, ncol=2, loc='lower right', handlelength=3.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(sm_values) + 4)
    ax.set_ylim(0, nvl_peak * 1.05)
    ax.set_xticks(range(0, max(sm_values) + 1, 16))

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f'Saved {outfile}')


def generate_summary_table(ldst_data, tma_data, outfile):
    """Generate a text summary table comparing peak BW per config."""
    sm_values = ldst_data['sm_values']
    ldst_results = ldst_data['results']
    tma_results = tma_data['results']

    all_keys = sorted(set(ldst_results.keys()) & set(tma_results.keys()))

    lines = []
    lines.append('=' * 100)
    lines.append('BF16 Dispatch Sender Comparison: ld/st vs TMA')
    lines.append('=' * 100)

    # Build metadata (read from archived JSON, not current environment)
    lines.append('')
    lines.append('Build Configuration (from archived run metadata):')
    for label, data in [('ld/st', ldst_data), ('TMA', tma_data)]:
        impl = data.get('implementation', 'unknown')
        lines.append(f'  [{label}] implementation: {impl}')
        lines.append(f'    git_commit: {data.get("git_commit", "unknown")}')
        lines.append(f'    torch: {data.get("torch_version", "unknown")}')
        lines.append(f'    CUDA: {data.get("cuda_version", "unknown")}')
        lines.append(f'    build_flags: {data.get("build_flags", "unknown")}')
        lines.append(f'    num_experts: {data.get("num_experts", "unknown")}')
        lines.append(f'    num_topk: {data.get("num_topk", "unknown")}')
        lines.append(f'    num_warmups: {data.get("num_warmups", "unknown")}')
        lines.append(f'    num_tests: {data.get("num_tests", "unknown")}')
    lines.append(f'  NVLink peak: {ldst_data.get("nvl_peak", 900.0):.0f} GB/s')
    lines.append('')

    # Table header
    header = f'{"Config":<20} {"ld/st Peak":>12} {"@SMs":>6} {"@chunk":>7} {"TMA Peak":>12} {"@SMs":>6} {"@chunk":>7} {"Delta":>8}'
    lines.append(header)
    lines.append('-' * len(header))

    for key in all_keys:
        ldst_peak, ldst_sms, ldst_chunk = get_peak_for_config(ldst_results[key], sm_values)
        tma_peak, tma_sms, tma_chunk = get_peak_for_config(tma_results[key], sm_values)

        if ldst_peak > 0:
            delta_pct = (tma_peak - ldst_peak) / ldst_peak * 100
        else:
            delta_pct = 0.0

        ldst_sms_s = f'{ldst_sms:>5}' if ldst_sms is not None else '  N/A'
        ldst_chunk_s = f'{ldst_chunk:>6}' if ldst_chunk is not None else '   N/A'
        tma_sms_s = f'{tma_sms:>5}' if tma_sms is not None else '  N/A'
        tma_chunk_s = f'{tma_chunk:>6}' if tma_chunk is not None else '   N/A'
        delta_s = f'{delta_pct:>+7.2f}%' if ldst_peak > 0 and tma_peak > 0 else '    N/A'

        lines.append(
            f'{key:<20} {ldst_peak:>10.2f}  {ldst_sms_s} {ldst_chunk_s}  '
            f'{tma_peak:>10.2f}  {tma_sms_s} {tma_chunk_s}  {delta_s}'
        )

    lines.append('')

    with open(outfile, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Summary saved to {outfile}')
    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description='Compare ld/st vs TMA sender performance')
    parser.add_argument('--ldst-file', required=True, help='Path to ld/st results JSON')
    parser.add_argument('--tma-file', required=True, help='Path to TMA results JSON')
    parser.add_argument('--output-dir', default='/tmp', help='Output directory for plots')
    args = parser.parse_args()

    ldst_data = load_results(args.ldst_file)
    tma_data = load_results(args.tma_file)

    # Validate that both files use the same SM sweep
    if ldst_data['sm_values'] != tma_data['sm_values']:
        print(f'ERROR: SM sweep mismatch between input files.')
        print(f'  ld/st: {ldst_data["sm_values"][:5]}...{ldst_data["sm_values"][-3:]} '
              f'({len(ldst_data["sm_values"])} values)')
        print(f'  TMA:   {tma_data["sm_values"][:5]}...{tma_data["sm_values"][-3:]} '
              f'({len(tma_data["sm_values"])} values)')
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Derive token/hidden sizes from archived metadata (fall back to result keys)
    num_tokens_list = ldst_data.get('num_tokens_list')
    hidden_list = ldst_data.get('hidden_list')
    if num_tokens_list and hidden_list:
        token_sizes = [t // 1024 for t in num_tokens_list]
        hidden_sizes = hidden_list
    else:
        # Parse from result keys like "T=4k H=7168"
        token_sizes = sorted({int(k.split('k')[0].split('=')[1]) for k in ldst_data['results']})
        hidden_sizes = sorted({int(k.split('H=')[1]) for k in ldst_data['results']}, reverse=True)

    nvl_peak = ldst_data.get('nvl_peak', 900.0)

    # Overview: all combos
    all_keys = [f'T={t}k H={h}' for t in token_sizes for h in hidden_sizes]
    plot_overlay(ldst_data, tma_data, all_keys,
                 'BF16 Dispatch: ld/st vs TMA Sender (all configs)\n(B200, NVLink 5.0, 8 GPUs)',
                 os.path.join(args.output_dir, 'sender_cmp_all.png'), nvl_peak=nvl_peak)

    # Per-token focused plots
    for t in token_sizes:
        keys = [f'T={t}k H={h}' for h in hidden_sizes]
        plot_overlay(ldst_data, tma_data, keys,
                     f'BF16 Dispatch: ld/st vs TMA — T={t}k\n(B200, NVLink 5.0, 8 GPUs)',
                     os.path.join(args.output_dir, f'sender_cmp_{t}k_hidden.png'),
                     nvl_peak=nvl_peak)

    # Per-hidden focused plots
    for h in hidden_sizes:
        keys = [f'T={t}k H={h}' for t in token_sizes]
        plot_overlay(ldst_data, tma_data, keys,
                     f'BF16 Dispatch: ld/st vs TMA — H={h}\n(B200, NVLink 5.0, 8 GPUs)',
                     os.path.join(args.output_dir, f'sender_cmp_{h}_tokens.png'),
                     nvl_peak=nvl_peak)

    # Summary table
    generate_summary_table(ldst_data, tma_data,
                           os.path.join(args.output_dir, 'sender_comparison_summary.txt'))


if __name__ == '__main__':
    main()
