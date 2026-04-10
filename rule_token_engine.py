import math
from typing import Dict, List, NamedTuple, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from tokenizer_utils import rms_normalize


class CellState(NamedTuple):
    cell_matrix: torch.Tensor
    cell_counts: torch.Tensor
    token_to_cell: torch.Tensor
    token_pos: torch.Tensor


class RuleSortContext(NamedTuple):
    """Precomputed sort/group indices for rule-based grouped operations.

    All projections and expert FFNs within a single TransformerBlock share
    the same rule_ids.  Computing the sort context once and reusing it
    eliminates O(N log N) redundant work per projection call.
    """
    unique_rules: torch.Tensor
    inverse: torch.Tensor
    sort_idx: torch.Tensor
    sorted_groups: torch.Tensor
    counts: torch.Tensor
    max_g: int
    pos_in_group: torch.Tensor


def compute_rule_sort_context(rule_ids: torch.Tensor) -> RuleSortContext:
    unique_rules, inverse = torch.unique(rule_ids, return_inverse=True)
    U = unique_rules.size(0)
    N = rule_ids.size(0)
    sort_idx = torch.argsort(inverse, stable=True)
    sorted_groups = inverse[sort_idx]
    counts = torch.bincount(inverse, minlength=U)
    max_g = int(counts.max().item())
    offsets = torch.zeros(U + 1, dtype=torch.long, device=rule_ids.device)
    offsets[1:] = counts.cumsum(0)
    pos_in_group = torch.arange(N, device=rule_ids.device) - offsets[sorted_groups]
    return RuleSortContext(unique_rules, inverse, sort_idx, sorted_groups, counts, max_g, pos_in_group)


class FiniteScalarQuantizer(nn.Module):
    def __init__(self, levels=None, embedding_dim=256):
        super().__init__()
        if levels is None:
            d = max(3, int(math.ceil(math.log(embedding_dim + 1) / math.log(6))))
            levels = self._greedy_factorize(max(2, int(math.sqrt(embedding_dim) * math.log(embedding_dim + 1))), d)
        self.levels = levels
        self.num_rules = math.prod(levels)
        self.d = len(levels)
        self.project_in = nn.Linear(embedding_dim, self.d)
        self.project_out = nn.Linear(self.d, embedding_dim)
        basis = [1]
        for i in range(len(levels) - 1):
            basis.append(basis[-1] * levels[i])
        self.register_buffer("basis", torch.tensor(basis, dtype=torch.long))
        self.register_buffer("levels_t", torch.tensor(levels, dtype=torch.float32))

    @staticmethod
    def _greedy_factorize(n, d):
        levels = []
        rem = n
        for i in range(d - 1):
            remaining_dims = d - i
            if rem <= 1:
                levels.append(1)
                continue
            ideal = max(2, int(round(rem ** (1.0 / remaining_dims))))
            best = 1
            for delta in range(ideal + 1):
                for cand in [ideal - delta, ideal + delta]:
                    if cand >= 2 and rem % cand == 0:
                        best = cand
                        break
                if best >= 2:
                    break
            levels.append(best)
            rem = rem // best if best >= 2 else rem
        levels.append(max(1, rem))
        return levels

    def forward(self, x):
        z = self.project_in(x)
        bound = torch.tanh(z)
        scaled = (bound + 1.0) / 2.0 * (self.levels_t - 1.0)
        quantized = torch.round(scaled)
        z_q = scaled + (quantized - scaled).detach()
        rule_ids = torch.sum(quantized.long() * self.basis, dim=-1)
        out = self.project_out(z_q)
        return out, rule_ids


class CustomRMSNorm(nn.Module):
    def __init__(self, dim, eps=None):
        super().__init__()
        self.eps = eps if eps is not None else torch.finfo(torch.float32).eps * dim
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normalized = rms_normalize(x, eps=self.eps)
        return normalized * self.weight.to(x.dtype)


def apply_rope(q, k, start_pos=0, rope_base=None):
    seq_len = q.size(2)
    head_dim = q.size(3)
    device = q.device
    if rope_base is None:
        rope_base = max(2.0, head_dim * math.exp(math.log(max(head_dim, 2))))
    position = torch.arange(start_pos, start_pos + seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, head_dim, 2, device=device) * (-math.log(rope_base) / head_dim))
    freqs = position * div_term
    freqs = freqs.unsqueeze(0).unsqueeze(0)
    sin = torch.sin(freqs)
    cos = torch.cos(freqs)
    q1, q2 = q[..., 0::2], q[..., 1::2]
    k1, k2 = k[..., 0::2], k[..., 1::2]
    q_rotated = torch.stack([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1).flatten(-2)
    k_rotated = torch.stack([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1).flatten(-2)
    return q_rotated, k_rotated


class RuleAwareProjection(nn.Module):
    def __init__(self, num_rules, in_features, out_features, rank, block_size):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.block_size = max(1, block_size)
        self.num_blocks = max(1, min(in_features, out_features) // self.block_size)
        self.shared_rank = max(1, int(round(math.sqrt(max(2, min(in_features, out_features))))))

        self.shared_in = nn.Parameter(torch.empty(in_features, self.shared_rank))
        self.shared_out = nn.Parameter(torch.empty(self.shared_rank, out_features))
        self.rule_in = nn.Parameter(torch.empty(num_rules, self.block_size, rank))
        self.rule_out = nn.Parameter(torch.empty(num_rules, rank, self.block_size))
        self.rule_block_logits = nn.Parameter(torch.empty(num_rules, self.num_blocks))

        nn.init.kaiming_uniform_(self.shared_in, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.shared_out, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.rule_in, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.rule_out, a=math.sqrt(5))
        nn.init.normal_(self.rule_block_logits, std=1.0 / math.sqrt(self.num_blocks))

    def _shared_forward(self, x):
        shared_hidden = torch.matmul(x, self.shared_in)
        return torch.matmul(shared_hidden, self.shared_out)

    def forward(self, x):
        return self._shared_forward(x)

    def forward_rules_batched(self, x, rule_ids, sort_ctx=None):
        """Grouped-padded vectorization: rule params O(U) instead of O(N)."""
        base_out = self._shared_forward(x)
        N = x.size(0)
        total_block_dim = self.num_blocks * self.block_size
        x_blocks = x[..., :total_block_dim].view(N, self.num_blocks, self.block_size)

        if sort_ctx is None:
            sort_ctx = compute_rule_sort_context(rule_ids)

        U = sort_ctx.unique_rules.size(0)
        sorted_x_blocks = x_blocks[sort_ctx.sort_idx]

        padded = x_blocks.new_zeros(U, sort_ctx.max_g * self.num_blocks, self.block_size)
        flat_pos = sort_ctx.pos_in_group * self.num_blocks
        block_offsets = torch.arange(self.num_blocks, device=x.device)
        all_rows = sort_ctx.sorted_groups.unsqueeze(1).expand(-1, self.num_blocks).reshape(-1)
        all_cols = (flat_pos.unsqueeze(1) + block_offsets.unsqueeze(0)).reshape(-1)
        padded[all_rows, all_cols] = sorted_x_blocks.reshape(-1, self.block_size)

        r_in = self.rule_in[sort_ctx.unique_rules]
        r_out = self.rule_out[sort_ctx.unique_rules]

        adapter_hidden = torch.bmm(padded, r_in)
        adapter_out = torch.bmm(adapter_hidden, r_out)

        adapter_out = adapter_out.view(U, sort_ctx.max_g, self.num_blocks, self.block_size)

        logits = self.rule_block_logits[sort_ctx.unique_rules]
        temperature = 1.0 / math.sqrt(self.num_blocks)
        selector = F.softmax(logits / temperature, dim=-1).view(U, 1, self.num_blocks, 1)
        weighted = (adapter_out * selector).reshape(U, sort_ctx.max_g, total_block_dim)

        sorted_weighted = weighted[sort_ctx.sorted_groups, sort_ctx.pos_in_group]
        result = torch.empty(N, total_block_dim, device=x.device, dtype=sorted_weighted.dtype)
        result[sort_ctx.sort_idx] = sorted_weighted

        adapter_full = torch.zeros_like(base_out)
        adapter_full[..., :total_block_dim] = result.to(base_out.dtype)
        return base_out + adapter_full


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout, rope_base, num_rules):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.proj_rank = max(1, int(round(math.sqrt(self.head_dim))))

        self.q_proj = RuleAwareProjection(num_rules, d_model, d_model, self.proj_rank, self.head_dim)
        self.k_proj = RuleAwareProjection(num_rules, d_model, d_model, self.proj_rank, self.head_dim)
        self.v_proj = RuleAwareProjection(num_rules, d_model, d_model, self.proj_rank, self.head_dim)
        self.out_proj = RuleAwareProjection(num_rules, d_model, d_model, self.proj_rank, self.head_dim)

        self.router_rank = self.proj_rank
        self.rule_router_q = nn.Parameter(torch.randn(num_rules, self.router_rank) * 0.02)
        self.rule_router_k = nn.Parameter(torch.randn(num_rules, self.router_rank) * 0.02)

        self.dropout = nn.Dropout(dropout)
        self.rope_base = rope_base

    def _dense_attention(self, q, k, v, seq_len, start_pos, rule_mask=None):
        is_causal = (seq_len > 1 and start_pos == 0)
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=rule_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal if rule_mask is None else False,
        )

    def _rule_causal_attention(self, x, target_rule_ids, start_pos, sort_ctx=None):
        B, Seq, C = x.size()
        flat_x = x.view(-1, C)
        flat_rules = target_rule_ids.view(-1)

        if sort_ctx is None:
            sort_ctx = compute_rule_sort_context(flat_rules)

        q = self.q_proj.forward_rules_batched(flat_x, flat_rules, sort_ctx=sort_ctx)
        k = self.k_proj.forward_rules_batched(flat_x, flat_rules, sort_ctx=sort_ctx)
        v = self.v_proj.forward_rules_batched(flat_x, flat_rules, sort_ctx=sort_ctx)

        q = q.view(B, Seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Seq, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, start_pos, rope_base=self.rope_base)

        rq = self.rule_router_q[target_rule_ids]  # [B, Seq, rank]
        rk = self.rule_router_k[target_rule_ids]  # [B, Seq, rank]
        rq_norm = F.normalize(rq, p=2, dim=-1)
        rk_norm = F.normalize(rk, p=2, dim=-1)
        tau = 1.0 / math.log(max(math.e, self.router_rank))
        rule_affinity = torch.matmul(rq_norm, rk_norm.transpose(-1, -2)) / tau  # [B, Seq, Seq]

        causal_mask = torch.tril(torch.ones(Seq, Seq, device=q.device, dtype=torch.bool))
        masked_affinity = rule_affinity.masked_fill(~causal_mask.unsqueeze(0), float('-inf'))

        top_k = max(2, int(math.sqrt(Seq)))
        if Seq > top_k:
            threshold = torch.topk(masked_affinity, top_k, dim=-1).values[..., -1:]
            rule_mask = (masked_affinity >= threshold) & causal_mask.unsqueeze(0)
        else:
            rule_mask = causal_mask.unsqueeze(0)
            
        final_mask = rule_mask.unsqueeze(1) # [B, 1, Seq, Seq]

        attn_out = self._dense_attention(q, k, v, Seq, start_pos, rule_mask=final_mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(-1, C)

        out = self.out_proj.forward_rules_batched(attn_out, flat_rules, sort_ctx=sort_ctx)
        return out.view(B, Seq, C)

    def forward(self, x, start_pos=0, kv_cache=None, target_rule_ids=None, sort_ctx=None):
        B, Seq, C = x.size()
        if target_rule_ids is not None and kv_cache is None and Seq > 1:
            out = self._rule_causal_attention(x, target_rule_ids, start_pos, sort_ctx=sort_ctx)
            new_kv_cache = None
        else:
            q = self.q_proj(x).view(B, Seq, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, Seq, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, Seq, self.num_heads, self.head_dim).transpose(1, 2)
            q, k = apply_rope(q, k, start_pos, rope_base=self.rope_base)
            if kv_cache is not None:
                k_cache, v_cache = kv_cache
                k = torch.cat([k_cache, k], dim=2)
                v = torch.cat([v_cache, v], dim=2)
                new_kv_cache = (k, v)
            else:
                new_kv_cache = (k, v)
            out = self._dense_attention(q, k, v, Seq, start_pos)
            out = out.transpose(1, 2).contiguous().view(B, Seq, C)
            out = self.out_proj(out)
        return out, new_kv_cache


class BatchedRuleExperts(nn.Module):
    def __init__(self, num_rules, d_model, expert_dim, dropout):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(num_rules, d_model, expert_dim))
        self.b1 = nn.Parameter(torch.empty(num_rules, expert_dim))
        self.w2 = nn.Parameter(torch.empty(num_rules, expert_dim, d_model))
        self.b2 = nn.Parameter(torch.empty(num_rules, d_model))
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w1)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b1, -bound, bound)

        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w2)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b2, -bound, bound)

    def forward(self, x, rules):
        if x.numel() == 0:
            return x
        w1 = self.w1[rules]
        b1 = self.b1[rules]
        h = torch.bmm(x.unsqueeze(1), w1).squeeze(1) + b1
        h = F.gelu(h)
        h = self.dropout(h)
        w2 = self.w2[rules]
        b2 = self.b2[rules]
        out = torch.bmm(h.unsqueeze(1), w2).squeeze(1) + b2
        return self.dropout(out)

    @staticmethod
    def forward_with_runtime_pages(x, rules, runtime_pages, dropout_p, training, sort_ctx=None):
        if x.numel() == 0:
            return x
        N, D = x.shape

        if sort_ctx is None:
            sort_ctx = compute_rule_sort_context(rules)

        unique_list = sort_ctx.unique_rules.tolist()
        U = len(unique_list)

        stacked_w1 = torch.stack([runtime_pages[int(r)]["w1"] for r in unique_list])
        stacked_b1 = torch.stack([runtime_pages[int(r)]["b1"] for r in unique_list])
        stacked_w2 = torch.stack([runtime_pages[int(r)]["w2"] for r in unique_list])
        stacked_b2 = torch.stack([runtime_pages[int(r)]["b2"] for r in unique_list])

        sorted_x = x[sort_ctx.sort_idx]
        padded = x.new_zeros(U, sort_ctx.max_g, D)
        padded[sort_ctx.sorted_groups, sort_ctx.pos_in_group] = sorted_x

        h = torch.bmm(padded, stacked_w1) + stacked_b1.unsqueeze(1)
        h = F.gelu(h)
        h = F.dropout(h, p=dropout_p, training=training)
        out_padded = torch.bmm(h, stacked_w2) + stacked_b2.unsqueeze(1)
        out_padded = F.dropout(out_padded, p=dropout_p, training=training)

        sorted_out = out_padded[sort_ctx.sorted_groups, sort_ctx.pos_in_group]
        out = torch.empty_like(x)
        out[sort_ctx.sort_idx] = sorted_out
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout, rope_base, num_rules, expert_dim=None):
        super().__init__()
        self.norm1 = CustomRMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, dropout, rope_base, num_rules)
        self.norm2 = CustomRMSNorm(d_model)
        if expert_dim is None:
            expert_dim = max(16, dim_feedforward // max(num_rules, 1))
        self.experts = BatchedRuleExperts(num_rules, d_model, expert_dim, dropout)
        self.num_rules = num_rules
        self.expert_dim = expert_dim

    def forward(self, x, target_rule_ids=None, start_pos=0, kv_cache=None, runtime_expert_pages=None):
        B, Seq, C = x.size()
        flat_rules = target_rule_ids.view(-1) if target_rule_ids is not None else None
        sort_ctx = compute_rule_sort_context(flat_rules) if flat_rules is not None else None

        attn_out, new_kv_cache = self.attn(
            self.norm1(x), start_pos=start_pos, kv_cache=kv_cache,
            target_rule_ids=target_rule_ids, sort_ctx=sort_ctx,
        )
        x = x + attn_out
        normed_x = self.norm2(x)
        flat_x = normed_x.view(-1, C)
        if flat_rules is None:
            flat_rules = torch.zeros(B * Seq, dtype=torch.long, device=x.device)
        if runtime_expert_pages is not None:
            moe_out_flat = BatchedRuleExperts.forward_with_runtime_pages(
                flat_x, flat_rules, runtime_expert_pages,
                self.experts.dropout.p if self.experts is not None else 0.0,
                self.training, sort_ctx=sort_ctx,
            )
        else:
            if self.experts is None:
                raise RuntimeError("专家参数已卸载，但未提供 runtime_expert_pages。")
            moe_out_flat = self.experts(flat_x, flat_rules)
        moe_out = moe_out_flat.view(B, Seq, C)
        x = x + moe_out
        return x, new_kv_cache


class RuleTokenCausalModel(nn.Module):
    def __init__(self, vocab_size, num_rules=None, embed_size=None, hidden_size=None, num_heads=None, num_layers=None, expert_dim=None):
        super().__init__()
        base_entropy = math.log2(max(vocab_size, 2))
        if embed_size is None:
            embed_size = 1 << math.ceil(math.log2(base_entropy * base_entropy))
        if num_rules is None:
            num_rules = 1 << round(math.log2(math.sqrt(embed_size) * math.log(max(vocab_size, 2))))
        if hidden_size is None:
            hidden_size = 1 << math.ceil(math.log2(embed_size * math.e))
        if num_layers is None:
            num_layers = max(4, int(math.ceil(base_entropy / math.e)))
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        current_dim = embed_size

        if num_heads is None:
            divisors = [head for head in range(1, current_dim + 1) if current_dim % head == 0]
            target_head_dim = max(1, int(math.sqrt(current_dim)))
            num_heads = min(divisors, key=lambda head: (abs((current_dim // head) - target_head_dim), -head))
        self.num_heads = num_heads
        head_dim = current_dim // num_heads
        estimated_context_span = embed_size * max(math.log(num_rules + num_layers + 1), 1.0)
        self.rope_base = max(2.0, estimated_context_span * max(math.log(head_dim + math.e), 1.0))
        dropout = 1.0 / math.sqrt(max(2, num_layers * num_heads))

        fsq_levels = self._compute_fsq_levels(num_rules)
        actual_num_rules = math.prod(fsq_levels)
        if actual_num_rules != num_rules:
            num_rules = actual_num_rules
        self.num_rules = num_rules

        self.layers = nn.ModuleList([
            TransformerBlock(
                current_dim,
                num_heads,
                hidden_size,
                dropout=dropout,
                rope_base=self.rope_base,
                num_rules=self.num_rules,
                expert_dim=expert_dim,
            )
            for _ in range(num_layers)
        ])
        self.expert_dim = self.layers[0].expert_dim if self.layers else (
            int(expert_dim) if expert_dim is not None else max(16, hidden_size // max(num_rules, 1))
        )
        self.final_norm = CustomRMSNorm(current_dim)
        self.vq_layer = FiniteScalarQuantizer(fsq_levels, current_dim)
        self.rule_boundary_predictor = nn.Linear(current_dim, self.num_rules)
        self.truth_semantic_field = nn.Sequential(
            nn.Linear(current_dim, current_dim),
            nn.GELU(),
            nn.Linear(current_dim, embed_size),
        )

    def _compute_fsq_levels(self, num_rules):
        if num_rules <= 1:
            return [2, 2, 2]
        d = max(3, int(math.ceil(math.log(num_rules + 1) / math.log(6))))
        levels = FiniteScalarQuantizer._greedy_factorize(num_rules, d)
        if min(levels) >= 2:
            return sorted(levels, reverse=True)
        base = max(2, int(math.ceil(num_rules ** (1.0 / d))))
        while base ** d < num_rules:
            base += 1
        uniform = [base] * d
        for i in range(d):
            while uniform[i] > 2:
                trial = list(uniform)
                trial[i] -= 1
                if math.prod(trial) >= num_rules:
                    uniform[i] -= 1
                else:
                    break
        return sorted(uniform, reverse=True)

    def offload_experts_to_runtime(self):
        for layer in self.layers:
            layer.experts = None

    def forward(self, embeds, target_rule_ids=None, start_pos=0, past_key_values=None, runtime_expert_pages=None):
        features = rms_normalize(embeds) * math.sqrt(self.embed_size)
        h = features
        new_past_key_values = []
        for i, layer in enumerate(self.layers):
            kv_cache = past_key_values[i] if past_key_values is not None else None
            layer_pages = runtime_expert_pages[i] if runtime_expert_pages is not None else None
            if kv_cache is None and self.training and torch.is_grad_enabled() and h.requires_grad:
                def checkpointed_layer(inp):
                    layer_out, _ = layer(inp, target_rule_ids=target_rule_ids, start_pos=start_pos, kv_cache=None, runtime_expert_pages=layer_pages)
                    return layer_out
                h = checkpoint(checkpointed_layer, h, use_reentrant=False)
                new_kv_cache = None
            else:
                h, new_kv_cache = layer(h, target_rule_ids=target_rule_ids, start_pos=start_pos, kv_cache=kv_cache, runtime_expert_pages=layer_pages)
            new_past_key_values.append(new_kv_cache)
        memory = self.final_norm(h)
        rule_logits = self.rule_boundary_predictor(memory)
        field_state = self.truth_semantic_field(memory)
        return rule_logits, field_state, memory, new_past_key_values

    def compute_rule_vocab_mask(self, token_chunk_embeds=None, token_chunk_ids=None, cell_state=None) -> CellState:
        device = next(self.parameters()).device
        if cell_state is None:
            max_tokens_per_rule = max(256, (self.vocab_size // self.num_rules) * 10)
            cell_matrix = torch.zeros((self.num_rules, max_tokens_per_rule), dtype=torch.long, device=device)
            cell_counts = torch.zeros(self.num_rules, dtype=torch.long, device=device)
            token_pos = torch.zeros(self.vocab_size, dtype=torch.long, device=device)
            with torch.no_grad():
                chunk_sz = max(1, 100000000 // self.embed_size)
                global_rules_list = []
                for start_idx in range(0, self.vocab_size, chunk_sz):
                    end_idx = min(start_idx + chunk_sz, self.vocab_size)
                    sub_embeds = rms_normalize(token_chunk_embeds[start_idx:end_idx]) * math.sqrt(self.embed_size)
                    _, sub_rules = self.vq_layer(sub_embeds.to(device))
                    global_rules_list.append(sub_rules)
                global_rules = torch.cat(global_rules_list)
            token_to_cell = global_rules.clone()
            counts = torch.bincount(global_rules, minlength=self.num_rules)
            cell_counts.copy_(counts)
            _, sorted_tokens = torch.sort(global_rules)
            offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(counts[:-1], dim=0)])
            offsets_list = offsets.tolist()
            counts_list = counts.tolist()
            sorted_tokens_list = sorted_tokens.tolist()
            max_tokens_per_rule = cell_matrix.size(1)
            for r in range(self.num_rules):
                c = counts_list[r]
                if c > 0:
                    start = offsets_list[r]
                    c_clamped = min(c, max_tokens_per_rule)
                    rule_tokens = torch.tensor(sorted_tokens_list[start:start + c_clamped], dtype=torch.long, device=device)
                    cell_matrix[r, :c_clamped] = rule_tokens
                    token_pos[rule_tokens] = torch.arange(c_clamped, dtype=torch.long, device=device)
                    cell_counts[r] = c_clamped
            empty_mask = (cell_counts == 0)
            if empty_mask.any():
                cell_counts[empty_mask] = 1
            return CellState(cell_matrix, cell_counts, token_to_cell, token_pos)

        cell_matrix, cell_counts, token_to_cell, token_pos = cell_state
        max_tokens_per_rule = cell_matrix.size(1)
        with torch.no_grad():
            embeds = rms_normalize(token_chunk_embeds) * math.sqrt(self.embed_size)
            z = self.vq_layer.project_in(embeds.to(device))
            bound = torch.tanh(z)
            scaled = (bound + 1.0) / 2.0 * (self.vq_layer.levels_t - 1.0)
            quantized = torch.round(scaled)
            new_primary_rules = torch.sum(quantized.long() * self.vq_layer.basis, dim=-1)

        old_primary_rules = token_to_cell[token_chunk_ids]
        changed_mask = old_primary_rules != new_primary_rules
        changed_tokens = token_chunk_ids[changed_mask].tolist()
        old_r_list = old_primary_rules[changed_mask].tolist()
        new_r_list = new_primary_rules[changed_mask].tolist()

        for t, o_r, n_r in zip(changed_tokens, old_r_list, new_r_list):
            old_pos = int(token_pos[t].item())
            old_count = int(cell_counts[o_r].item())
            if old_pos < old_count:
                last_idx = old_count - 1
                if old_pos < last_idx:
                    last_token = int(cell_matrix[o_r, last_idx].item())
                    cell_matrix[o_r, old_pos] = last_token
                    token_pos[last_token] = old_pos
                cell_counts[o_r] = max(1, old_count - 1)

            new_pos = int(cell_counts[n_r].item())
            if new_pos < max_tokens_per_rule:
                cell_matrix[n_r, new_pos] = t
                token_pos[t] = new_pos
                cell_counts[n_r] = new_pos + 1
            token_to_cell[t] = n_r

        return CellState(cell_matrix, cell_counts, token_to_cell, token_pos)


class _ExpertLayerState(nn.Module):
    def __init__(self, num_rules, d_model, expert_dim):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(num_rules, d_model, expert_dim, device="cpu"))
        self.b1 = nn.Parameter(torch.empty(num_rules, expert_dim, device="cpu"))
        self.w2 = nn.Parameter(torch.empty(num_rules, expert_dim, d_model, device="cpu"))
        self.b2 = nn.Parameter(torch.empty(num_rules, d_model, device="cpu"))


class RulePagedExpertStore(nn.Module):
    def __init__(self, num_layers, num_rules, d_model, expert_dim):
        super().__init__()
        self.num_layers = num_layers
        self.num_rules = num_rules
        self.d_model = d_model
        self.expert_dim = expert_dim
        self.layers = nn.ModuleList([_ExpertLayerState(num_rules, d_model, expert_dim) for _ in range(num_layers)])
        self.cached_pages: List[Dict] = [{} for _ in range(num_layers)]
        self.register_buffer("rule_energy", torch.zeros(num_rules, dtype=torch.float32, device="cpu"))

    @classmethod
    def from_model(cls, model: RuleTokenCausalModel):
        store = cls(model.num_layers, model.num_rules, model.embed_size, model.expert_dim)
        for layer_idx, block in enumerate(model.layers):
            expert = block.experts
            if expert is None:
                continue
            dst = store.layers[layer_idx]
            dst.w1.data.copy_(expert.w1.detach().cpu())
            dst.b1.data.copy_(expert.b1.detach().cpu())
            dst.w2.data.copy_(expert.w2.detach().cpu())
            dst.b2.data.copy_(expert.b2.detach().cpu())
        return store

    def build_runtime(self, active_rules: List[int], device: torch.device, training: bool):
        active_rules = [int(r) for r in active_rules]

        for r in active_rules:
            self.rule_energy[r] += 1.0
        self.rule_energy *= 0.99

        if device.type == "cuda":
            total_mem = torch.cuda.get_device_properties(device).total_memory
            alloc_mem = torch.cuda.memory_allocated(device)
            if alloc_mem > 0.85 * total_mem:
                for layer_cache in self.cached_pages:
                    if not layer_cache:
                        continue
                    keys = list(layer_cache.keys())
                    keys.sort(key=lambda k: self.rule_energy[k].item())
                    to_evict = [k for k in keys if k not in active_rules][:max(1, len(keys) // 2)]
                    for k in to_evict:
                        del layer_cache[k]
                torch.cuda.empty_cache()

        runtime_pages = []
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = self.cached_pages[layer_idx]
            uncached = [r for r in active_rules if r not in layer_cache]
            if uncached:
                idx_t = torch.tensor(uncached, dtype=torch.long)
                w1_batch = layer.w1[idx_t].detach().to(device, non_blocking=True)
                b1_batch = layer.b1[idx_t].detach().to(device, non_blocking=True)
                w2_batch = layer.w2[idx_t].detach().to(device, non_blocking=True)
                b2_batch = layer.b2[idx_t].detach().to(device, non_blocking=True)
                for i, r in enumerate(uncached):
                    layer_cache[r] = {
                        "w1": w1_batch[i], "b1": b1_batch[i],
                        "w2": w2_batch[i], "b2": b2_batch[i],
                    }

            pages = {}
            for rule_idx in active_rules:
                cached_page = layer_cache[rule_idx]
                pages[rule_idx] = {
                    "w1": cached_page["w1"].requires_grad_(training),
                    "b1": cached_page["b1"].requires_grad_(training),
                    "w2": cached_page["w2"].requires_grad_(training),
                    "b2": cached_page["b2"].requires_grad_(training),
                }
            runtime_pages.append(pages)
        return runtime_pages

    def scatter_runtime_grads(self, runtime_pages):
        for layer_idx, pages in enumerate(runtime_pages):
            layer = self.layers[layer_idx]
            if layer.w1.grad is None:
                layer.w1.grad = torch.zeros_like(layer.w1, memory_format=torch.contiguous_format)
                layer.b1.grad = torch.zeros_like(layer.b1, memory_format=torch.contiguous_format)
                layer.w2.grad = torch.zeros_like(layer.w2, memory_format=torch.contiguous_format)
                layer.b2.grad = torch.zeros_like(layer.b2, memory_format=torch.contiguous_format)
            indices = []
            w1_grads, b1_grads, w2_grads, b2_grads = [], [], [], []
            for rule_idx, page in pages.items():
                has_grad = page["w1"].grad is not None
                if has_grad:
                    indices.append(rule_idx)
                    w1_grads.append(page["w1"].grad.detach())
                    b1_grads.append(page["b1"].grad.detach())
                    w2_grads.append(page["w2"].grad.detach())
                    b2_grads.append(page["b2"].grad.detach())
            if indices:
                idx_t = torch.tensor(indices, dtype=torch.long)
                layer.w1.grad[idx_t] = torch.stack(w1_grads).cpu()
                layer.b1.grad[idx_t] = torch.stack(b1_grads).cpu()
                layer.w2.grad[idx_t] = torch.stack(w2_grads).cpu()
                layer.b2.grad[idx_t] = torch.stack(b2_grads).cpu()

    def zero_grad(self, set_to_none: bool = True):
        for param in self.parameters():
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()
        for layer_cache in self.cached_pages:
            for page in layer_cache.values():
                for tensor in page.values():
                    if set_to_none:
                        tensor.grad = None
                    elif tensor.grad is not None:
                        tensor.grad.zero_()
