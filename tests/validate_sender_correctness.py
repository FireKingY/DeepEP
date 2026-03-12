"""Validate BF16 dispatch correctness across sender implementations.

Two-phase workflow:
  Phase 1 (--save-reference): Run dispatch with current build, save outputs.
  Phase 2 (--compare-reference): Run dispatch with current build, compare against saved reference.

Usage:
  # With ld/st build:
  python validate_sender_correctness.py --save-reference --output-dir /tmp/sender_ref
  # With TMA build:
  python validate_sender_correctness.py --compare-reference --output-dir /tmp/sender_ref --results-file /tmp/validation.json
"""
import argparse
import json
import os
import torch
import torch.distributed as dist
import deep_ep
from utils import init_dist

CONFIGS = [
    {'label': 'default',       'num_sms': 24,  'chunk': 8},
    {'label': 'sms40_chunk16', 'num_sms': 40,  'chunk': 16},
    {'label': 'sms100_chunk32','num_sms': 100, 'chunk': 32},
]


def run_dispatch(buffer, x, topk_idx, topk_weights, num_tokens_per_rank,
                 is_token_in_rank, num_tokens_per_expert, num_sms, chunk):
    """Run BF16 dispatch and return (recv_x, recv_topk_idx, recv_topk_weights)."""
    recv_buf = max(256, chunk * 4)
    cfg = deep_ep.Config(num_sms, chunk, recv_buf)
    recv_x, recv_topk_idx, recv_topk_weights, _, _, _ = buffer.dispatch(
        x=x, num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        topk_idx=topk_idx, topk_weights=topk_weights, config=cfg)
    return recv_x, recv_topk_idx, recv_topk_weights


def validate_main(args, local_rank, num_ranks, rank, buffer, group):
    num_tokens = 4096
    hidden = 7168
    num_topk = 8
    num_experts = 64

    torch.manual_seed(42 + rank)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1].to(deep_ep.topk_idx_t)
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')

    num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.save_reference:
        if local_rank == 0:
            print('Saving ld/st reference outputs...', flush=True)
        for cfg in CONFIGS:
            recv_x, recv_topk_idx, recv_topk_weights = run_dispatch(
                buffer, x, topk_idx, topk_weights,
                num_tokens_per_rank, is_token_in_rank,
                num_tokens_per_expert, cfg['num_sms'], cfg['chunk'])
            ref_path = os.path.join(args.output_dir, f'ref_{cfg["label"]}_rank{rank}.pt')
            torch.save({
                'recv_x': recv_x.cpu(),
                'recv_topk_idx': recv_topk_idx.cpu(),
                'recv_topk_weights': recv_topk_weights.cpu(),
            }, ref_path)
            if local_rank == 0:
                print(f'  {cfg["label"]}: saved recv_x{list(recv_x.shape)} '
                      f'recv_topk_idx{list(recv_topk_idx.shape)} '
                      f'recv_topk_weights{list(recv_topk_weights.shape)}', flush=True)

    elif args.compare_reference:
        if local_rank == 0:
            print('Comparing TMA outputs against ld/st reference (all ranks)...', flush=True)
        results = []
        all_pass = True
        for cfg in CONFIGS:
            recv_x, recv_topk_idx, recv_topk_weights = run_dispatch(
                buffer, x, topk_idx, topk_weights,
                num_tokens_per_rank, is_token_in_rank,
                num_tokens_per_expert, cfg['num_sms'], cfg['chunk'])
            ref_path = os.path.join(args.output_dir, f'ref_{cfg["label"]}_rank{rank}.pt')
            ref = torch.load(ref_path, map_location='cuda', weights_only=True)

            # Compare all dispatch outputs
            x_diff = (recv_x.float() - ref['recv_x'].float()).abs().max().item()
            idx_diff = (recv_topk_idx.long() - ref['recv_topk_idx'].long()).abs().max().item()
            weights_diff = (recv_topk_weights.float() - ref['recv_topk_weights'].float()).abs().max().item()
            local_max_diff = max(x_diff, float(idx_diff), weights_diff)

            # Aggregate max diff across all ranks
            global_max_diff_t = torch.tensor([local_max_diff], device='cuda')
            dist.all_reduce(global_max_diff_t, op=dist.ReduceOp.MAX, group=group)
            global_max_diff = global_max_diff_t.item()

            passed = global_max_diff == 0.0
            if not passed:
                all_pass = False

            result = {
                'config': cfg['label'],
                'num_sms': cfg['num_sms'],
                'chunk': cfg['chunk'],
                'passed': passed,
                'max_diff_recv_x': x_diff,
                'max_diff_topk_idx': idx_diff,
                'max_diff_topk_weights': weights_diff,
                'max_elementwise_diff_global': global_max_diff,
                'num_ranks_validated': num_ranks,
                'recv_x_shape': list(recv_x.shape),
            }
            results.append(result)
            if local_rank == 0:
                status = 'PASS' if passed else 'FAIL'
                print(f'  {cfg["label"]} (sms={cfg["num_sms"]}, chunk={cfg["chunk"]}): '
                      f'{status}  x_diff={x_diff} idx_diff={idx_diff} '
                      f'weights_diff={weights_diff} global={global_max_diff}', flush=True)

            assert passed, (f'Validation FAILED on config {cfg["label"]}: '
                            f'global_max_diff={global_max_diff} (rank {rank} '
                            f'x={x_diff} idx={idx_diff} weights={weights_diff})')

        if rank == 0 and args.results_file:
            validation = {
                'test': 'BF16 dispatch sender correctness (TMA vs ld/st)',
                'num_tokens': num_tokens,
                'hidden': hidden,
                'num_topk': num_topk,
                'num_experts': num_experts,
                'num_ranks': num_ranks,
                'torch_version': torch.__version__,
                'cuda_version': torch.version.cuda,
                'all_passed': all_pass,
                'configs': results,
            }
            with open(args.results_file, 'w') as f:
                json.dump(validation, f, indent=2)
            print(f'\nValidation results saved to {args.results_file}', flush=True)
            print(f'Overall: {"ALL PASSED" if all_pass else "SOME FAILED"}', flush=True)


def validate_loop(local_rank, num_local_ranks, args):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    buffer = deep_ep.Buffer(group, int(8e9), 0, explicitly_destroy=True)
    validate_main(args, local_rank, num_ranks, rank, buffer, group)
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--output-dir', required=True, help='Dir for reference tensors')
    parser.add_argument('--results-file', default=None, help='Path to write validation JSON')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--save-reference', action='store_true')
    mode.add_argument('--compare-reference', action='store_true')
    args = parser.parse_args()

    torch.multiprocessing.spawn(validate_loop, args=(args.num_processes, args),
                                nprocs=args.num_processes)
