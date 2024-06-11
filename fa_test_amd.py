import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
loss = nn.MSELoss()
from typing import Dict, List, Optional, Tuple, Type
from argparse import ArgumentParser
from triton_flash_attention import triton_attention

parser = ArgumentParser(description="LLM Inference Benchmark Example")
parser.add_argument(
    "--b",
    type=int,
    default=1,
    help="batch"
)

parser.add_argument(
    "--s",
    type=int,
    default=1,
    help="seqlen"
)

parser.add_argument(
    "--nh",
    type=int,
    default=1,
    help="nheads"
)

parser.add_argument(
    "--hs",
    type=int,
    default=1,
    help="headsize"
)

args = parser.parse_args()

seqlen    = args.s
batch     = args.b
num_heads = args.nh
headsize    = args.hs

qkv_ratio = 8
warmups = 5
iters = 5

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    tokens, n_kv_heads, head_dim = x.shape
    return (x[:, :,
              None, :].expand(tokens, n_kv_heads, n_rep,
                              head_dim).reshape(tokens, n_kv_heads * n_rep,
                                                head_dim))

def _naive_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    prompt_lens: List[int],
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    scale: float,
) -> torch.Tensor:
    query = query.reshape(-1, num_heads, head_size)
    #key = key.view(-1, num_kv_heads, head_size)
    #value = value.view(-1, num_kv_heads, head_size)
    key = key.reshape(-1, num_heads, head_size)
    value = value.reshape(-1, num_heads, head_size)
    num_tokens = query.shape[0]
    output = torch.empty_like(query)
    start = 0
    for _, prompt_len in enumerate(prompt_lens):
        end = start + prompt_len
        out = _naive_masked_attention(
            #query[None, start:end],
            #key[None, start:end],
            #value[None, start:end],
            query[start:end],
            key[start:end],
            value[start:end],
            scale,
        )
        # TODO(woosuk): Unnecessary copy. Optimize.
        output[start:end].copy_(out)
        start += prompt_len

    # Using view got RuntimeError: view size is not compatible
    # with input tensor's size and stride (at least one
    # dimension spans across two contiguous subspaces).
    # Use reshape instead.
    #return output.reshape(num_tokens, -1)
    return output


def _naive_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:

    seq_len, _, _ = query.shape
    attn_mask = torch.triu(torch.ones(seq_len,
                                      seq_len,
                                      dtype=query.dtype,
                                      device=query.device),
                           diagonal=1)
    attn_mask = attn_mask * torch.finfo(query.dtype).min

    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out

causal = True

print("batch" + "," + "seqlen" + "," + "num_heads" + "," + "num_kv_heads" + "," + "latency_fav2" + "," + "latency_naive")
num_kv_heads = int(num_heads/qkv_ratio)
max_prompt_len = seqlen

q_list = [0]
length_q = 0
prompt_lens = [seqlen]
for b in range(batch):
    length_q = length_q + seqlen
    q_list.append(length_q)
    prompt_lens.append(seqlen)

seq_start_loc = torch.as_tensor(q_list, device="cuda").int()

scale = 1 / math.sqrt(headsize)

#print("causal: ", causal)
dtype=torch.float16

latency_set = []
latency_set_triton = []
for itr in range(warmups + iters):

    query = torch.randn((batch * seqlen, num_heads, headsize), dtype=dtype, device="cuda", requires_grad=False)
    key   = torch.randn((batch * seqlen, num_kv_heads, headsize), dtype=dtype, device="cuda", requires_grad=False)
    value = torch.randn((batch * seqlen, num_kv_heads, headsize), dtype=dtype, device="cuda", requires_grad=False)
    if 0:
        print("qkv shapes")
        print(query.shape)
        print(key.shape)
        print(value.shape)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()

    start_event.record()
    out_triton_fav2, _ = triton_attention(
                            query,
                            key,
                            value,
                            None,
                            seq_start_loc,
                            seq_start_loc,
                            max_prompt_len,
                            max_prompt_len,
                            True,
                            scale,
                        )
    #del triton_attention

    end_event.record()

    torch.cuda.synchronize()
    latency_set_triton.append(start_event.elapsed_time(end_event))


    if num_heads != num_kv_heads:
        num_queries_per_kv = num_heads // num_kv_heads
        key = repeat_kv(key, num_queries_per_kv)
        value = repeat_kv(value, num_queries_per_kv)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()

    start_event.record()

    out_ref = _naive_attention(
                            query,
                            key,
                            value,
                            prompt_lens,
                            num_heads,
                            num_kv_heads,
                            headsize,
                            scale,
                        )
    end_event.record()

    torch.cuda.synchronize()
    latency_set.append(start_event.elapsed_time(end_event))
    if 0:
        print("itr: ", str(itr), ", loss: ", loss(out_ref, out_triton_fav2))

latency_set_triton.sort()
latency_set_triton = latency_set_triton[:iters]
count = len(latency_set_triton)
latency_avg_triton = sum(latency_set_triton) / count

latency_set.sort()
latency_set = latency_set[:iters]
count = len(latency_set)
latency_avg = sum(latency_set) / count
print(str(batch) + "," + str(seqlen) + "," + str(num_heads) + "," + str(num_kv_heads) + "," + "{0:8.2f}".format(latency_avg_triton) + "," + "{0:8.2f}".format(latency_avg))

if 0:
    print(out_ref.shape)
    print(out_ref)
    print(out_triton_fav2.shape)
    print(out_triton_fav2)
