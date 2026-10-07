from wan.modules.attention import attention, flash_attention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d,
    WanT2VCrossAttention,
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import copy
import torch.distributed as dist

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention,
    dynamic=False,
    mode="default"
)


class DualFullExpertLinear(nn.Linear):
    """Route memory and generation tokens to separate linear weights."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.memory_weight = nn.Parameter(torch.empty_like(self.weight))
        self.memory_bias = (
            nn.Parameter(torch.empty_like(self.bias)) if self.bias is not None else None
        )

    @classmethod
    def from_linear(cls, linear):
        routed = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)
        with torch.no_grad():
            routed.weight.copy_(linear.weight)
            routed.memory_weight.copy_(linear.weight)
            if linear.bias is not None:
                routed.bias.copy_(linear.bias)
                routed.memory_bias.copy_(linear.bias)
        return routed


    def forward(self, x, memory_token_count=0):
        if memory_token_count == 0:
            return F.linear(x, self.weight, self.bias)
        if not 0 <= memory_token_count <= x.shape[1]:
            raise ValueError(
                f"memory_token_count={memory_token_count} is incompatible with input {x.shape}"
            )
        memory_output = F.linear(
            x[:, :memory_token_count], self.memory_weight, self.memory_bias
        )
        if memory_token_count == x.shape[1]:
            return memory_output
        target_output = F.linear(x[:, memory_token_count:], self.weight, self.bias)
        return torch.cat([memory_output, target_output], dim=1)


class MemoryRoutedFFN(nn.Sequential):
    """Token-routed FFN compatible with FSDP wrapping."""

    def forward(self, x, memory_token_count=0):
        x = _memory_routed_linear(self[0], x, memory_token_count)
        x = self[1](x)
        return _memory_routed_linear(self[2], x, memory_token_count)


def _memory_routed_linear(linear, x, memory_token_count):
    if isinstance(linear, DualFullExpertLinear):
        return linear(x, memory_token_count)
    return linear(x)


def _memory_routed_norm(norm, x, memory_token_count):
    if isinstance(norm, nn.Identity) or memory_token_count == 0:
        return norm(x)
    if isinstance(norm, nn.LayerNorm) and norm.weight is None and norm.bias is None:
        return norm(x)

    memory_output = _memory_only_norm(norm, x[:, :memory_token_count])

    if memory_token_count == x.shape[1]:
        return memory_output
    return torch.cat([memory_output, norm(x[:, memory_token_count:])], dim=1)


def _memory_only_linear(linear, x):
    return F.linear(x, linear.memory_weight, linear.memory_bias)


def _memory_only_norm(norm, x):
    if isinstance(norm, nn.Identity):
        return x
    memory_weight = getattr(norm, "memory_weight", None)
    memory_bias = getattr(norm, "memory_bias", None)
    if isinstance(norm, WanRMSNorm):
        weight = memory_weight if memory_weight is not None else norm.weight.detach()
        return norm._norm(x.float()).type_as(x) * weight
    if isinstance(norm, nn.LayerNorm):
        weight = (
            memory_weight
            if memory_weight is not None
            else (norm.weight.detach() if norm.weight is not None else None)
        )
        bias = (
            memory_bias
            if memory_bias is not None
            else (norm.bias.detach() if norm.bias is not None else None)
        )
        return F.layer_norm(x, norm.normalized_shape, weight, bias, norm.eps).type_as(x)
    raise TypeError(f"Unsupported memory-only norm: {type(norm)}")


def _memory_t2v_cross_attention(
    module, x, context, context_lens, crossattn_cache=None
):
    b, n, d = x.size(0), module.num_heads, module.head_dim
    q = _memory_only_norm(module.norm_q, _memory_only_linear(module.q, x)).view(
        b, -1, n, d
    )
    if crossattn_cache is not None and crossattn_cache.get("is_init", False):
        k = crossattn_cache["k"]
        v = crossattn_cache["v"]
    else:
        k = _memory_only_norm(
            module.norm_k, _memory_only_linear(module.k, context)
        ).view(b, -1, n, d)
        v = _memory_only_linear(module.v, context).view(b, -1, n, d)
        if crossattn_cache is not None:
            crossattn_cache["is_init"] = True
            crossattn_cache["k"] = k
            crossattn_cache["v"] = v
    output = flash_attention(q, k, v, k_lens=context_lens).flatten(2)
    return _memory_only_linear(module.o, output)


def _memory_routed_cross_attention(
    module, x, context, context_lens, memory_token_count, crossattn_cache=None,
    memory_context=None,
):
    if memory_token_count == 0:
        return module(x, context, context_lens, crossattn_cache=crossattn_cache)
    if crossattn_cache is not None and memory_token_count != x.shape[1]:
        raise ValueError("Differentiable writer routing does not support a cross-attention cache")
    if not isinstance(module, WanT2VCrossAttention):
        raise TypeError(f"Writer routing currently supports T2V cross-attention, got {type(module)}")

    memory_output = _memory_t2v_cross_attention(
        module,
        x[:, :memory_token_count],
        memory_context if memory_context is not None else context.detach(),
        context_lens,
        crossattn_cache=crossattn_cache,
    )
    if memory_token_count == x.shape[1]:
        return memory_output
    target_output = module(
        x[:, memory_token_count:], context, context_lens, crossattn_cache=None
    )
    return torch.cat([memory_output, target_output], dim=1)


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


def causal_rope_apply_frames(x, grid_sizes, freqs, frame_indices):
    """RoPE variant where every latent frame is given an *explicit* temporal
    index (instead of a contiguous ``start_frame:start_frame+f`` range).

    This is used by the streaming long-video KV cache: keys/values are stored
    *before* RoPE is applied, and on every step we re-apply RoPE with relative
    indices ``frame_indices`` so the temporal position never leaves the model's
    trained range (e.g. 0-20).

    Args:
        x (Tensor): Shape [B, L, n, d], with ``L = f*h*w`` for each sample.
        grid_sizes (Tensor): Shape [B, 3] with (f, h, w) per sample.
        frame_indices (LongTensor): Shape [f], the temporal RoPE index for each
            of the ``f`` latent frames.
    """
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    frame_indices = frame_indices.to(freqs[0].device)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][frame_indices].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560

        # ── Streaming long-video KV cache (relative RoPE) ──────────────────
        # When `kv_rope_relative` is True the self-attention stores *un-roped*
        # K/V in the cache and re-applies RoPE every step with contiguous
        # relative frame indices, so the temporal position stays in
        # [0, kv_cache_max_frames-1] (never exceeding the trained range).
        # `kv_cache_max_frames` is the size of the attention window *including*
        # the frame currently being generated: with the defaults below the
        # window is the sink frame (idx 0) + 19 FIFO context frames (idx 1-19)
        # + the current frame (idx 20). The first `kv_cache_sink` frame(s) are
        # never evicted; the rest roll FIFO.
        # Default off → ordinary (short-video) inference is unchanged.
        self.kv_rope_relative = False
        self.kv_cache_max_frames = 21
        self.kv_cache_sink = 1
        # Trained temporal range (position-encoding ceiling). RoPE positions are
        # placed to match training: sink at {0..sink-1}, recent window pinned to
        # the top of this range, gap preserved; beyond it the geometry freezes.
        self.kv_cache_train_frames = 21
        self.kv_cache_position_mode = "top_aligned"

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        memory_token_count=0,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = _memory_routed_linear(self.q, x, memory_token_count)
            k = _memory_routed_linear(self.k, x, memory_token_count)
            q = _memory_routed_norm(self.norm_q, q, memory_token_count).view(b, s, n, d)
            k = _memory_routed_norm(self.norm_k, k, memory_token_count).view(b, s, n, d)
            v = _memory_routed_linear(self.v, x, memory_token_count).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)
        elif self.kv_rope_relative:
            # ── Streaming long-video path ──────────────────────────────────
            # Store raw (pre-RoPE) K/V; re-apply RoPE with relative indices so
            # the temporal position always lives in [0, kv_cache_max_frames-1].
            fs = math.prod(grid_sizes[0][1:]).item()
            cur_frames = q.shape[1] // fs
            cur_abs = current_start // fs
            max_frames = self.kv_cache_max_frames
            sink = self.kv_cache_sink

            # Per-cache bookkeeping: chronological list of the absolute frame
            # index occupying each slot (sink frames first, then recent window).
            abs_list = kv_cache.get("stream_abs", None)
            if abs_list is None:
                abs_list = []
                kv_cache["stream_abs"] = abs_list

            new_block_abs = [cur_abs + i for i in range(cur_frames)]
            # Re-running the same frame (extra denoising / context-update step)
            # overwrites the tail slot instead of appending a new frame.
            is_rewrite = (len(abs_list) >= cur_frames
                          and abs_list[-cur_frames:] == new_block_abs)

            if is_rewrite:
                tail_slot = len(abs_list) - cur_frames
            else:
                n_after = len(abs_list) + cur_frames
                if n_after > max_frames:
                    # Evict the oldest non-sink frame(s); shift the kept window
                    # left, leaving the first `sink` frame(s) pinned in place.
                    num_evict = n_after - max_frames
                    src_lo = (sink + num_evict) * fs
                    src_hi = len(abs_list) * fs
                    dst_lo = sink * fs
                    dst_hi = dst_lo + (src_hi - src_lo)
                    if src_hi > src_lo:
                        kv_cache["k"][:, dst_lo:dst_hi] = kv_cache["k"][:, src_lo:src_hi].clone()
                        kv_cache["v"][:, dst_lo:dst_hi] = kv_cache["v"][:, src_lo:src_hi].clone()
                    del abs_list[sink:sink + num_evict]
                abs_list.extend(new_block_abs)
                tail_slot = len(abs_list) - cur_frames

            # Write the current block's *un-roped* K/V into its tail slot.
            tail_lo = tail_slot * fs
            tail_hi = tail_lo + cur_frames * fs
            kv_cache["k"][:, tail_lo:tail_hi] = k
            kv_cache["v"][:, tail_lo:tail_hi] = v

            num_occ = len(abs_list)
            if self.kv_cache_position_mode == "contiguous":
                # Compress the retained cache into the training prefix. With
                # sink=0, max_frames=12 this yields {0..11} once full and then
                # keeps that exact RoPE geometry forever.
                pos = torch.arange(num_occ, device=q.device, dtype=torch.long)
            else:
                # RoPE positions that MATCH the teacher-forcing training geometry:
                #   sink frames pinned at {0 .. sink-1};
                #   recent window pinned to the TOP of the trained range so the
                #     query/window relative distances equal those seen in training;
                #   the gap (dropped middle frames) is preserved.
                # Query is clamped to train_frames-1, so beyond the trained range
                # the whole geometry freezes (e.g. sink={0,1,2}, window={9..20}
                # forever) instead of being compressed into {0..num_occ-1}.
                n_win = num_occ - sink
                if n_win <= 0:
                    pos = torch.arange(num_occ, device=q.device, dtype=torch.long)
                else:
                    q_abs = cur_abs + cur_frames - 1
                    Q = min(int(q_abs), self.kv_cache_train_frames - 1)
                    pos = torch.cat([
                        torch.arange(sink, device=q.device, dtype=torch.long),
                        torch.arange(Q - (n_win - 1), Q + 1, device=q.device, dtype=torch.long),
                    ])

            window_grid = grid_sizes.clone()
            window_grid[:, 0] = num_occ
            roped_key = causal_rope_apply_frames(
                kv_cache["k"][:, :num_occ * fs], window_grid, freqs, pos).type_as(v)

            q_pos = pos[tail_slot:tail_slot + cur_frames]
            roped_query = causal_rope_apply_frames(
                q, grid_sizes, freqs, q_pos).type_as(v)

            x = attention(
                roped_query,
                roped_key,
                kv_cache["v"][:, :num_occ * fs]
            )
            # Keep the legacy counters roughly in sync (unused by this path).
            kv_cache["global_end_index"].fill_(current_start + q.shape[1])
            kv_cache["local_end_index"].fill_(num_occ * fs)
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # Insert the new keys/values at the end
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            win_lo = max(0, local_end_index - self.max_attention_size)
            if self.local_attn_size != -1 and sink_tokens > 0 and win_lo > sink_tokens:
                # Disjoint sink + recent window (a real gap separates them).
                # Eviction above keeps [0, sink_tokens) un-rolled, so this slice is
                # always the original first `sink_size` frames.
                k_read = torch.cat(
                    [kv_cache["k"][:, :sink_tokens], kv_cache["k"][:, win_lo:local_end_index]], dim=1)
                v_read = torch.cat(
                    [kv_cache["v"][:, :sink_tokens], kv_cache["v"][:, win_lo:local_end_index]], dim=1)
            else:
                # Contiguous: when a sink exists and the window already reaches back
                # into/before it, read from 0 so the sink is not skipped (e.g. frame
                # 12 must see frame 0). Otherwise the plain recent slice (pure window,
                # or full attention when local_attn_size == -1).
                lo = 0 if (self.local_attn_size != -1 and sink_tokens > 0) else win_lo
                k_read = kv_cache["k"][:, lo:local_end_index]
                v_read = kv_cache["v"][:, lo:local_end_index]
            x = attention(roped_query, k_read, v_read)
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        # output
        x = x.flatten(2)
        # x.shape is [1, 65520, 1536]
        x = _memory_routed_linear(self.o, x, memory_token_count)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        memory_token_count=0,
        memory_context=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        if memory_token_count % frame_seqlen != 0:
            raise ValueError(
                f"memory_token_count={memory_token_count} is not frame-aligned "
                f"for frame_seqlen={frame_seqlen}"
            )
        memory_frame_count = memory_token_count // frame_seqlen
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        if memory_frame_count == 0:
            modulated_e = self.modulation.unsqueeze(1) + e
        else:
            memory_modulation = getattr(
                self, "memory_modulation", self.modulation.detach()
            )
            memory_e = memory_modulation.unsqueeze(1) + e[:, :memory_frame_count]
            if memory_frame_count == num_frames:
                modulated_e = memory_e
            else:
                target_e = self.modulation.unsqueeze(1) + e[:, memory_frame_count:]
                modulated_e = torch.cat([memory_e, target_e], dim=1)
        e = modulated_e.chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            (_memory_routed_norm(self.norm1, x, memory_token_count)
             .unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start,
            memory_token_count=memory_token_count)

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            normalized_x = _memory_routed_norm(self.norm3, x, memory_token_count)
            x = x + _memory_routed_cross_attention(
                self.cross_attn,
                normalized_x,
                context,
                context_lens,
                memory_token_count,
                crossattn_cache=crossattn_cache,
                memory_context=memory_context,
            )
            ffn_input = (
                _memory_routed_norm(self.norm2, x, memory_token_count)
                .unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                * (1 + e[4]) + e[3]
            ).flatten(1, 2)
            if getattr(self, "dual_full_expert_enabled", False):
                y = self.ffn(
                    ffn_input, memory_token_count=memory_token_count
                )
            else:
                y = self.ffn(ffn_input)
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False
        self.dual_full_expert_enabled = False


    @staticmethod
    def _add_memory_norm_parameters(norm):
        if isinstance(norm, nn.Identity):
            return 0
        count = 0
        if getattr(norm, "weight", None) is not None:
            norm.register_parameter(
                "memory_weight", nn.Parameter(norm.weight.detach().clone())
            )
            count += norm.weight.numel()
        if getattr(norm, "bias", None) is not None:
            norm.register_parameter(
                "memory_bias", nn.Parameter(norm.bias.detach().clone())
            )
            count += norm.bias.numel()
        return count

    def enable_dual_full_expert(self):
        if self.dual_full_expert_enabled:
            raise RuntimeError("Dual full experts are already enabled")
        if self.model_type != 't2v':
            raise NotImplementedError("Dual full experts currently support T2V only")

        memory_params = 0
        for block in self.blocks:
            for name in ("q", "k", "v", "o"):
                linear = getattr(block.self_attn, name)
                routed = DualFullExpertLinear.from_linear(linear)
                setattr(block.self_attn, name, routed)
                memory_params += routed.memory_weight.numel()
                if routed.memory_bias is not None:
                    memory_params += routed.memory_bias.numel()
            for name in ("q", "k", "v", "o"):
                linear = getattr(block.cross_attn, name)
                routed = DualFullExpertLinear.from_linear(linear)
                setattr(block.cross_attn, name, routed)
                memory_params += routed.memory_weight.numel()
                if routed.memory_bias is not None:
                    memory_params += routed.memory_bias.numel()
            for index in (0, 2):
                routed = DualFullExpertLinear.from_linear(block.ffn[index])
                block.ffn[index] = routed
                memory_params += routed.memory_weight.numel()
                if routed.memory_bias is not None:
                    memory_params += routed.memory_bias.numel()
            block.ffn = MemoryRoutedFFN(*block.ffn.children())
            block.dual_full_expert_enabled = True
            for norm in (
                block.norm1,
                block.norm2,
                block.norm3,
                block.self_attn.norm_q,
                block.self_attn.norm_k,
                block.cross_attn.norm_q,
                block.cross_attn.norm_k,
            ):
                memory_params += self._add_memory_norm_parameters(norm)
            block.memory_modulation = nn.Parameter(block.modulation.detach().clone())
            memory_params += block.memory_modulation.numel()

        self.memory_patch_embedding = copy.deepcopy(self.patch_embedding)
        self.memory_text_embedding = copy.deepcopy(self.text_embedding)
        self.memory_time_embedding = copy.deepcopy(self.time_embedding)
        self.memory_time_projection = copy.deepcopy(self.time_projection)
        self.memory_head = copy.deepcopy(self.head)
        memory_params += sum(
            parameter.numel()
            for module in (
                self.memory_patch_embedding,
                self.memory_text_embedding,
                self.memory_time_embedding,
                self.memory_time_projection,
                self.memory_head,
            )
            for parameter in module.parameters()
        )

        self.dual_full_expert_enabled = True
        return memory_params



    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1,
        local_attn_size: int = -1, sink_size: int = 0
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        # debug
        DEBUG = False
        if DEBUG:
            num_frames = 9
            frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # Block-wise causal mask attends to elements before the end of the current chunk.
        attention_block_size = frame_seqlen * num_frame_per_block

        # Streaming attention = sink + sliding window (StreamingLLM-style), token units.
        #   local_attn_size = -1  -> unbounded recent window (original full behaviour)
        #   local_attn_size  = W  -> attend to the last W frames INCLUDING the current one
        #   sink_size        = S  -> the first S frames are ALWAYS attended (pinned)
        # When local_attn_size == -1 and sink_size == 0 this reduces exactly to the
        # original blockwise-causal teacher-forcing mask.
        # sink/window are measured in LATENTS (frame_seqlen), independent of the
        # chunk size, so chunk=1 and chunk=3 share the same temporal extent. The
        # intra-chunk bidirectional + cross-chunk causal structure still comes from
        # attention_block_size below. sink/window must be multiples of the chunk so
        # the window stays chunk-aligned (matches the chunk-granular cache eviction).
        assert sink_size % num_frame_per_block == 0, \
            f"sink_size ({sink_size}) must be a multiple of num_frame_per_block ({num_frame_per_block})"
        if local_attn_size != -1:
            assert local_attn_size % num_frame_per_block == 0, \
                f"local_attn_size ({local_attn_size}) must be a multiple of num_frame_per_block ({num_frame_per_block})"
        win_tokens = local_attn_size * frame_seqlen if local_attn_size != -1 else None
        sink_tokens = sink_size * frame_seqlen

        # clean context frames: blockwise-causal self-attention in [context_starts, context_ends)
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # noisy frames: recent clean-context window [noise_context_starts, noise_context_ends) + own block
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames (self-attention building their K/V representation)
        for start in frame_indices:
            end = int(start) + attention_block_size
            context_ends[start:start + attention_block_size] = end
            # a clean frame looks back at most `local_attn_size` frames (incl. its own block)
            context_starts[start:start + attention_block_size] = \
                max(0, end - win_tokens) if win_tokens is not None else 0

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # clean context = all frames strictly before this block ...
            ctx_end = block_index * attention_block_size
            noise_context_ends[start:end] = ctx_end
            # ... limited to the most recent (local_attn_size - 1) clean frames
            if win_tokens is not None:
                noise_context_starts[start:end] = max(0, ctx_end - (win_tokens - attention_block_size))
            else:
                noise_context_starts[start:end] = 0

        def attention_mask(b, h, q_idx, kv_idx):
            # clean self-attention: CAUSAL (kv block <= q block) AND in the recent
            # window OR the sink. The causal bound is applied to BOTH terms, so the
            # sink frames themselves stay causal (clean frame 0 -> {0}, 1 -> {0,1}).
            clean_causal = kv_idx < context_ends[q_idx]
            clean_keep = (kv_idx >= context_starts[q_idx]) | (kv_idx < sink_tokens)
            clean_mask = (q_idx < clean_ends) & clean_causal & clean_keep
            # noisy frames: own block (C1) + clean context STRICTLY BEFORE this block
            # (causal), restricted to the recent window OR the sink.
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            ctx_causal = kv_idx < noise_context_ends[q_idx]
            ctx_keep = (kv_idx >= noise_context_starts[q_idx]) | (kv_idx < sink_tokens)
            noise_mask = (q_idx >= clean_ends) & (C1 | (ctx_causal & ctx_keep))

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                                padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        use_memory_expert: bool = False,
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        use_memory_expert = (
            self.dual_full_expert_enabled and use_memory_expert
        )

        # embeddings
        patch_embedding = (
            self.memory_patch_embedding if use_memory_expert
            else self.patch_embedding
        )
        x = [patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        memory_token_count = 0
        if self.dual_full_expert_enabled and use_memory_expert:
            memory_token_count = x.shape[1]


        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        time_embedding = (
            self.memory_time_embedding if use_memory_expert
            else self.time_embedding
        )
        time_projection = (
            self.memory_time_projection if use_memory_expert
            else self.time_projection
        )
        e = time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        padded_context = torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ])
        text_embedding = (
            self.memory_text_embedding if use_memory_expert
            else self.text_embedding
        )
        context = text_embedding(padded_context)

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            memory_token_count=memory_token_count,
            memory_context=context if use_memory_expert else None,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            block_crossattn_cache = crossattn_cache[block_index]
            if self.dual_full_expert_enabled:
                expert_name = (
                    "memory_expert" if use_memory_expert
                    else "generation_expert"
                )
                block_crossattn_cache = block_crossattn_cache.setdefault(
                    expert_name, {"is_init": False}
                )
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": block_crossattn_cache,
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": block_crossattn_cache,
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = block(x, **kwargs)

        # head
        head = self.memory_head if use_memory_expert else self.head
        x = head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
        use_memory_expert: bool = False,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None: # TF
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size,
                        sink_size=self.sink_size,
                    )
            else: # DF?
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        memory_token_count = 0

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        time_embedding = self.time_embedding
        time_projection = self.time_projection
        e = time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        padded_context = torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ])
        context = self.text_embedding(padded_context)
        memory_context = None
        if self.dual_full_expert_enabled and (
                clean_x is not None or use_memory_expert):
            memory_context = self.memory_text_embedding(padded_context)

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            # clean_x.detach()
            if self.dual_full_expert_enabled:
                clean_x = [
                    self.memory_patch_embedding(u.unsqueeze(0)) for u in clean_x
                ]
            else:
                clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            if self.dual_full_expert_enabled:
                memory_token_count = clean_x.shape[1]
            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            if self.dual_full_expert_enabled:
                e_clean = self.memory_time_embedding(
                    sinusoidal_embedding_1d(
                        self.freq_dim, aug_t.flatten()
                    ).type_as(x)
                )
                e0_clean = self.memory_time_projection(e_clean).unflatten(
                    1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            else:
                e_clean = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
                e0_clean = self.time_projection(e_clean).unflatten(
                    1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)
        elif self.dual_full_expert_enabled and use_memory_expert:
            memory_token_count = x.shape[1]

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            memory_token_count=memory_token_count,
            memory_context=memory_context)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]
            # [1,32760,1536]
        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            # TF or DF
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
