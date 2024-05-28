import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
loss = nn.MSELoss()
from typing import Dict, List, Optional, Tuple, Type

from flash_attn import flash_attn_varlen_func

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

causal_list = [True]

#llama3 70b config
batch_size_list = [
        1,
        8,
        16,
        32,
        ]
seqlen_list = [
        1,
        512,
        1024,
        2048,
        4096,
        ]
nheads_list = [
        64,
        ]
qkv_ratio = 8
head_size = 128
warmups = 5
iters = 5

print("batch" + "," + "seqlen" + "," + "num_heads" + "," + "num_kv_heads" + "," + "latency_fav2" + "," + "latency_naive")
for seqlen in seqlen_list:
    for batch in batch_size_list:
        for num_heads in nheads_list:
            for causal in causal_list:
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

                scale = 1 / math.sqrt(head_size)

                #print("causal: ", causal)
                dtype=torch.float16

                latency_set = []
                latency_set_fav2 = []
                for itr in range(warmups + iters):

                    query = torch.randn((batch * seqlen, num_heads, head_size), dtype=dtype, device="cuda", requires_grad=False)
                    key   = torch.randn((batch * seqlen, num_kv_heads, head_size), dtype=dtype, device="cuda", requires_grad=False)
                    value = torch.randn((batch * seqlen, num_kv_heads, head_size), dtype=dtype, device="cuda", requires_grad=False)
                    if 0:
                        print("qkv shapes")
                        print(query.shape)
                        print(key.shape)
                        print(value.shape)
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()

                    start_event.record()
                    out_fav2 = flash_attn_varlen_func(
                                            query,
                                            key,
                                            value,
                                            seq_start_loc,
                                            seq_start_loc,
                                            max_prompt_len,
                                            max_prompt_len,
                                            0.0,
                                            scale,
                                            True,
                                            (-1, -1),
                                            None,
                                        )

                    end_event.record()

                    torch.cuda.synchronize()
                    latency_set_fav2.append(start_event.elapsed_time(end_event))


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
                                            head_size,
                                            scale,
                                        )
                    end_event.record()

                    torch.cuda.synchronize()
                    latency_set.append(start_event.elapsed_time(end_event))
                    if 0:
                        print("itr: ", str(itr), ", loss: ", loss(out_ref, out_fav2))
                latency_set_fav2.sort()
                latency_set_fav2 = latency_set_fav2[:iters]
                count = len(latency_set_fav2)
                latency_avg_fav2 = sum(latency_set_fav2) / count

                latency_set.sort()
                latency_set = latency_set[:iters]
                count = len(latency_set)
                latency_avg = sum(latency_set) / count
                print(str(batch) + "," + str(seqlen) + "," + str(num_heads) + "," + str(num_kv_heads) + "," + "{0:8.2f}".format(latency_avg_fav2) + "," + "{0:8.2f}".format(latency_avg))

                if 0:
                    print(out_ref.shape)
                    print(out_ref)
                    print(out_fav2.shape)
                    print(out_fav2)
