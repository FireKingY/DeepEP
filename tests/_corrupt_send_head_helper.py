"""Helper script invoked by test_direct_write_dispatch.py as a subprocess.
Corrupts send_head in a direct-write dispatch handle, then calls combine.
Expected to crash via kernel trap (non-zero exit).

The marker 'REACHED_CORRUPTED_COMBINE' is printed to stderr immediately
before the corrupted combine call, so the parent test can verify the
helper actually reached the critical code path before failing."""
import sys
import os

# Add both the tests/ dir (for utils) and the repo root (for deep_ep)
_tests_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_tests_dir)
sys.path.insert(0, _tests_dir)
sys.path.insert(0, _repo_root)

import torch

# noinspection PyUnresolvedReferences
import deep_ep
from utils import init_dist, inplace_unique

_NUM_PROCESSES = 8


def run(local_rank):
    num_ranks = _NUM_PROCESSES
    torch.cuda.set_device(local_rank)
    rank, _, group = init_dist(local_rank, num_ranks)

    num_tokens, hidden, num_topk, num_experts = 256, 7168, 8, 256
    num_sms = 20
    config = deep_ep.Config(num_sms, 8, 256)
    num_worst_tokens = num_tokens * num_ranks

    buffer = deep_ep.Buffer(group, 1024 * 1024 * 1024)
    buffer.register_direct_write_layout(num_worst_tokens, hidden, num_topk, 0, 2, config=config)

    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1].to(deep_ep.topk_idx_t)
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')

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

    # Dispatch
    recv_x, _, topk_w, _, handle, _ = buffer.dispatch(
        x=x, num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank, num_tokens_per_expert=num_tokens_per_expert,
        topk_idx=topk_idx, topk_weights=topk_weights,
        config=config, num_worst_tokens=num_worst_tokens)

    actual_recv = handle[0][-1][rank].item()
    rank_pm, chan_pm, recv_chan_pm, src_idx, is_tir, send_hd = handle

    # Corrupt send_head: set all valid entries to a huge value
    # This makes combine wait for a tail position that never arrives → timeout → trap
    corrupted_sh = send_hd.clone()
    corrupted_sh[corrupted_sh >= 0] = 999999
    corrupted_handle = (rank_pm, chan_pm, recv_chan_pm, src_idx[:actual_recv], is_tir, corrupted_sh)

    # Emit marker so the parent test can verify we reached this point
    print("REACHED_CORRUPTED_COMBINE", file=sys.stderr, flush=True)

    # This should trap/timeout and kill the process
    buffer.combine(x=recv_x[:actual_recv], handle=corrupted_handle,
                   topk_weights=topk_w[:actual_recv], config=config)
    torch.cuda.synchronize()

    # If we get here, the test failed — combine should have trapped
    print("ERROR: combine did not fail with corrupted send_head", file=sys.stderr, flush=True)
    sys.exit(0)  # Return 0 = test failure (we expected non-zero)


if __name__ == '__main__':
    torch.multiprocessing.spawn(run, nprocs=_NUM_PROCESSES)
