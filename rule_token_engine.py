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
        # Saturation-safe bounding.  Plain tanh has gradient tanh'(z) that
        # vanishes once |z|≫1, which in turn kills gradient flow into
        # project_in and freezes the rule codebook.  We add a 1/d-budget
        # self-normalising leak so the derivative d bound / d z is lower
        # bounded by ~1/d everywhere, keeping FSQ learnable.
        leak = 1.0 / max(self.d, 1)
        bound = torch.tanh(z) + leak * z / (1.0 + z.abs().detach())
        bound = bound.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        scaled = (bound + 1.0) / 2.0 * (self.levels_t - 1.0)
        quantized = torch.round(scaled)
        z_q = scaled + (quantized - scaled).detach()
        rule_ids = torch.sum(quantized.long() * self.basis, dim=-1)
        out = self.project_out(z_q)
        return out, rule_ids


class HolographicRuleBinding(nn.Module):
    """Plate-style Holographic Reduced Representation binding of rule ⊛ key.

    A rule-aware circular convolution: for every position, the rule identity
    is encoded as a learnable unit-norm "role" vector and is bound with a
    context-driven "filler" vector produced from the residual stream.

    Mathematical properties (Plate, 1995; Kanerva, 2009):
      - binding: a ⊛ b = IFFT(FFT(a) ⊙ FFT(b))
      - the role vectors stay approximately unit-norm, so binding is a
        near-isometry and the total energy of the bound signal matches the
        filler — i.e. no arbitrary scale is injected into the residual stream.
      - unbinding is approximately b ≈ a* ⊛ (a ⊛ b), where a* is the
        involution of a (time-reversed on the circular axis).  This gives
        the field_state a *retrievable* rule-typed component, unlike a plain
        linear projection.

    Complexity: O(B·S·D log D) per forward, dominated by two rFFTs.
    """

    def __init__(self, num_rules: int, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_rules = num_rules
        # Initialise each rule's role vector with unit RMS so binding is
        # variance-preserving from the start.
        role_init = torch.randn(num_rules, feature_dim) / math.sqrt(feature_dim)
        self.role = nn.Parameter(role_init)
        self.filler_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.out_scale = nn.Parameter(torch.full((feature_dim,), 1.0 / math.sqrt(feature_dim)))

    def forward(self, memory: torch.Tensor, rule_ids: torch.Tensor) -> torch.Tensor:
        role = F.normalize(self.role[rule_ids], p=2, dim=-1)  # [B,S,D]
        filler = self.filler_proj(memory)                      # [B,S,D]
        # Use rFFT: real-valued inputs → half-spectrum multiplication → iRFFT.
        with torch.amp.autocast(device_type=role.device.type, enabled=False):
            X = torch.fft.rfft(role.float(), n=self.feature_dim, dim=-1)
            Y = torch.fft.rfft(filler.float(), n=self.feature_dim, dim=-1)
            bound = torch.fft.irfft(X * Y, n=self.feature_dim, dim=-1)
        return (bound.to(memory.dtype) * self.out_scale)

    def unbind(self, bound: torch.Tensor, rule_ids: torch.Tensor) -> torch.Tensor:
        """Approximate inverse: involution(role) ⊛ bound ≈ filler.

        Useful for downstream probes / analyses that want to recover the
        context-driven component from a rule-typed representation.
        """
        role = F.normalize(self.role[rule_ids], p=2, dim=-1)
        role_star = torch.roll(role.flip(-1), shifts=1, dims=-1)  # circular involution
        with torch.amp.autocast(device_type=role.device.type, enabled=False):
            X = torch.fft.rfft(role_star.float(), n=self.feature_dim, dim=-1)
            Y = torch.fft.rfft(bound.float(), n=self.feature_dim, dim=-1)
            recovered = torch.fft.irfft(X * Y, n=self.feature_dim, dim=-1)
        return recovered.to(bound.dtype)


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
        # Default base matches the Llama-family convention 10000 but scales
        # with head_dim so small heads keep enough phase dynamic range.
        rope_base = max(10000.0, float(head_dim) ** 2)
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


def _factor_near_sqrt(n: int) -> Tuple[int, int]:
    """Return the (a, b) with a*b == n whose geometric ratio is closest to 1.

    Used to shape the Kronecker factors so each factor is ~sqrt(n); this
    maximises the effective rank of the Kronecker product for a given
    parameter budget.
    """
    n = max(1, int(n))
    target = max(1, int(round(math.sqrt(n))))
    for delta in range(0, n):
        for cand in (target - delta, target + delta):
            if 1 <= cand <= n and n % cand == 0:
                other = n // cand
                a, b = min(cand, other), max(cand, other)
                return a, b
    return 1, n


class RuleAwareProjection(nn.Module):
    """Kronecker-factor rule adapter: W_r = W_0 + alpha · (U_r ⊗ V_r).

    Old implementation used a block-diagonal LoRA, where the rule-specific
    part only touched diagonal blocks of the weight matrix and the effective
    rank per output channel was capped at the inner LoRA rank `k`.

    Here we replace the block-diagonal pattern by a true Kronecker product.
    For input dimension in = a·b and output dimension out = c·d we store
      U_r ∈ R^{c × a}   V_r ∈ R^{d × b}
    so that the per-rule adapter matrix (U_r ⊗ V_r) has:
      - parameter count  c·a + d·b   (i.e. O(sqrt(in·out)))
      - representable rank  min(c·d, a·b) = min(out, in)
    That is, the same parameter budget as block-LoRA buys us linear-in-d rank
    instead of constant-in-d rank.  Application uses the identity
      (U ⊗ V) vec(X) = vec(V X U^T)
    so the forward kernel is two small batched matmuls of shape (b×a)·(a×c)
    and (d×b)·(b×c).  No hand-chosen block partitioning remains.
    """

    def __init__(self, num_rules, in_features, out_features):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.shared_rank = max(1, int(round(math.sqrt(max(2, min(in_features, out_features))))))

        # Factorise in / out near the geometric centre for dense Kronecker.
        self.a, self.b = _factor_near_sqrt(self.in_features)    # in = a * b
        self.c, self.d = _factor_near_sqrt(self.out_features)   # out = c * d

        self.shared_in = nn.Parameter(torch.empty(self.in_features, self.shared_rank))
        self.shared_out = nn.Parameter(torch.empty(self.shared_rank, self.out_features))
        self.rule_U = nn.Parameter(torch.empty(num_rules, self.c, self.a))
        self.rule_V = nn.Parameter(torch.empty(num_rules, self.d, self.b))
        # Per-rule learnable gain.  Note: Kronecker adapters can't use the
        # classical LoRA zero-init trick — if gain == 0 the gradient w.r.t.
        # both U and V is zero (multiplicative symmetry), so the factors
        # would never move.  Instead we seed with a small positive scale:
        # the rule path starts at ~1% of the shared projection magnitude,
        # but every factor receives gradient from step 1.
        init_gain = 1.0 / math.sqrt(max(num_rules, 1))
        self.rule_gain = nn.Parameter(torch.full((num_rules,), init_gain))

        nn.init.kaiming_uniform_(self.shared_in, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.shared_out, a=math.sqrt(5))
        # Scale rule factors so Var[(U ⊗ V) x] ≈ Var[x] at init.
        kron_std = 1.0 / math.sqrt(max(self.a * self.b, 1))
        nn.init.normal_(self.rule_U, std=kron_std)
        nn.init.normal_(self.rule_V, std=kron_std)

    def _shared_forward(self, x):
        shared_hidden = torch.matmul(x, self.shared_in)
        return torch.matmul(shared_hidden, self.shared_out)

    def forward(self, x):
        # Shared-only path.  We deliberately do *not* apply the rule adapter
        # here; the rule-aware kernel requires rule ids and is called through
        # forward_rules_batched on the training path.
        return self._shared_forward(x)

    def forward_rules_batched(self, x, rule_ids, sort_ctx=None):
        """Kronecker-batched forward:  y_r = W_0 · x + gain_r · (U_r ⊗ V_r) · x.

        Grouped by rule so that each unique rule only performs a small
        (d×b)·(b×c) and (b×a)·(a×c) batched matmul.
        """
        base_out = self._shared_forward(x)
        N = x.size(0)
        if sort_ctx is None:
            sort_ctx = compute_rule_sort_context(rule_ids)
        U = sort_ctx.unique_rules.size(0)

        a, b, c, d = self.a, self.b, self.c, self.d
        # Reshape input  x ∈ R^{N × (a·b)}  →  [N, b, a] (column-major vec).
        x_mat = x.view(N, b, a)

        sorted_x = x_mat[sort_ctx.sort_idx]                          # [N, b, a]
        padded = sorted_x.new_zeros(U, sort_ctx.max_g, b, a)
        padded[sort_ctx.sorted_groups, sort_ctx.pos_in_group] = sorted_x

        U_rules = self.rule_U[sort_ctx.unique_rules]                 # [U, c, a]
        V_rules = self.rule_V[sort_ctx.unique_rules]                 # [U, d, b]

        # (V X U^T)  ≡  Kronecker-vector product  (U ⊗ V) vec(X).
        # Use einsum; on modern PyTorch this compiles down to two bmm's.
        xu = torch.einsum('ugba,uca->ugbc', padded, U_rules)         # [U, max_g, b, c]
        vxu = torch.einsum('udb,ugbc->ugdc', V_rules, xu)            # [U, max_g, d, c]
        adapter_padded = vxu.reshape(U, sort_ctx.max_g, d * c)       # [U, max_g, out]

        gain = self.rule_gain[sort_ctx.unique_rules].view(U, 1, 1)
        adapter_padded = adapter_padded * gain

        sorted_adapter = adapter_padded[sort_ctx.sorted_groups, sort_ctx.pos_in_group]
        adapter_flat = torch.empty(N, d * c, device=x.device, dtype=sorted_adapter.dtype)
        adapter_flat[sort_ctx.sort_idx] = sorted_adapter
        return base_out + adapter_flat.to(base_out.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout, rope_base, num_rules):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.proj_rank = max(1, int(round(math.sqrt(self.head_dim))))

        self.q_proj = RuleAwareProjection(num_rules, d_model, d_model)
        self.k_proj = RuleAwareProjection(num_rules, d_model, d_model)
        self.v_proj = RuleAwareProjection(num_rules, d_model, d_model)
        self.out_proj = RuleAwareProjection(num_rules, d_model, d_model)

        self.router_rank = self.proj_rank
        self.rule_router_q = nn.Parameter(torch.randn(num_rules, self.router_rank) * 0.02)
        self.rule_router_k = nn.Parameter(torch.randn(num_rules, self.router_rank) * 0.02)

        # Hierarchical rule-level attention path. At each position t, a small
        # query head looks up the causal running mean of every active rule
        # (prefix average of values so far conditioned on rule == r).  This
        # complements the token-level attention by giving every token an
        # O(1)-per-rule global summary without forming an L×L matrix.
        # Complexity: O(B·S·U·C) where U is the number of rules active in
        # the batch, typically U ≪ R ≪ S.
        self.rule_level_q = nn.Linear(d_model, d_model, bias=False)
        self.rule_level_kv = nn.Linear(d_model, 2 * d_model, bias=False)
        # Gate starts slightly > 0 so the hierarchical path contributes ~1%
        # of the token-level output at initialisation.  A strict zero init
        # would multiplicatively kill gradient to rule_level_q / rule_level_kv
        # on the first step (gate·tanh = 0 → dL/d{q,kv} = 0), leading to a
        # slow cold-start.
        self.rule_level_gate = nn.Parameter(torch.full((d_model,), 1e-2))

        self.dropout = nn.Dropout(dropout)
        self.rope_base = rope_base

    def _dense_attention(self, q, k, v, seq_len, start_pos, attn_bias=None):
        is_causal = (seq_len > 1 and start_pos == 0)
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal if attn_bias is None else False,
        )

    def _hierarchical_rule_attention(self, x: torch.Tensor, rule_ids: torch.Tensor) -> torch.Tensor:
        """Rule-level causal attention: every token attends to the prefix
        mean of *each active rule* up to its own position.

        Derivation.  Let 1_{r}(s) = [rule(s) == r].  The causal rule-conditional
        running mean at time t for rule r is
            μ_t^r = (Σ_{s≤t} 1_{r}(s) · V(x_s)) / (Σ_{s≤t} 1_{r}(s)),
        which we obtain in O(S·U) with two cumulative sums.  Each token then
        queries this U-sized memory via a scaled-dot-product attention over
        the U axis, giving a causal O(U) memory per position instead of O(S).

        Shapes:
          x        : [B, S, C]
          rule_ids : [B, S]
          returns  : [B, S, C]   (already scaled by the residual gate)
        """
        B, S, C = x.shape
        device = x.device
        unique_rules, inverse = torch.unique(rule_ids, return_inverse=True)  # inverse: [B,S]
        U = unique_rules.numel()
        if U == 0:
            return torch.zeros_like(x)

        # Build K,V and Q for the rule-level path.
        kv = self.rule_level_kv(x)            # [B, S, 2C]
        k_val, v_val = kv.chunk(2, dim=-1)    # each [B, S, C]
        q_val = self.rule_level_q(x)          # [B, S, C]

        # One-hot masks for the U rules actually present in this batch.
        # [B, S, U]; dtype float for cumsum.
        membership = F.one_hot(inverse, num_classes=U).to(k_val.dtype)

        # Causal prefix sums per rule.
        # kv_per_rule[b, t, u, c] = Σ_{s≤t} 1[rule(s)==u] · kv[b, s, c]
        # count_per_rule[b, t, u] = Σ_{s≤t} 1[rule(s)==u]
        k_sum = torch.einsum('bsu,bsc->bsuc', membership, k_val).cumsum(dim=1)   # [B,S,U,C]
        v_sum = torch.einsum('bsu,bsc->bsuc', membership, v_val).cumsum(dim=1)   # [B,S,U,C]
        count = membership.cumsum(dim=1).clamp_min(1.0)                           # [B,S,U]
        k_mean = k_sum / count.unsqueeze(-1)
        v_mean = v_sum / count.unsqueeze(-1)

        # Per-position attention over the U rule summaries.
        scale = 1.0 / math.sqrt(max(C, 1))
        logits = torch.einsum('bsc,bsuc->bsu', q_val, k_mean) * scale             # [B,S,U]

        # Mask rules that have never appeared in the causal prefix so they
        # cannot leak information from future tokens.
        seen_mask = (count > 0.5)                                                 # [B,S,U]
        logits = logits.masked_fill(~seen_mask, torch.finfo(logits.dtype).min)
        weights = F.softmax(logits, dim=-1)
        out = torch.einsum('bsu,bsuc->bsc', weights, v_mean)                      # [B,S,C]

        # Residual gate (per-channel), starts at zero so the hierarchical
        # path is OFF at initialisation and grows only if it improves loss.
        return out * torch.tanh(self.rule_level_gate)

    def _compute_rule_level_state(self, x: torch.Tensor, rule_ids: torch.Tensor):
        B, _, C = x.shape
        kv = self.rule_level_kv(x)
        k_val, v_val = kv.chunk(2, dim=-1)
        membership = F.one_hot(rule_ids.to(torch.long), num_classes=self.rule_router_q.size(0)).to(k_val.dtype)
        rule_k_sum = torch.einsum('bsr,bsc->brc', membership, k_val)
        rule_v_sum = torch.einsum('bsr,bsc->brc', membership, v_val)
        rule_count = membership.sum(dim=1)
        return {
            "rule_level_k": k_val,
            "rule_level_v": v_val,
            "rule_k_sum": rule_k_sum.view(B, self.rule_router_q.size(0), C),
            "rule_v_sum": rule_v_sum.view(B, self.rule_router_q.size(0), C),
            "rule_count": rule_count.view(B, self.rule_router_q.size(0)),
        }

    def _build_rule_kv_cache(self, x: torch.Tensor, target_rule_ids: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        cache = {
            "k": k,
            "v": v,
            "rule_ids": target_rule_ids.to(torch.long),
        }
        cache.update(self._compute_rule_level_state(x, target_rule_ids))
        return cache

    def _append_rule_kv_cache(
        self,
        kv_cache,
        x_step: torch.Tensor,
        rule_ids_step: torch.Tensor,
        k_step: torch.Tensor,
        v_step: torch.Tensor,
    ):
        if kv_cache is None:
            return self._build_rule_kv_cache(x_step, rule_ids_step, k_step, v_step)

        updated = {
            "k": torch.cat([kv_cache["k"], k_step], dim=2),
            "v": torch.cat([kv_cache["v"], v_step], dim=2),
            "rule_ids": torch.cat([kv_cache["rule_ids"], rule_ids_step.to(torch.long)], dim=1),
        }
        step_state = self._compute_rule_level_state(x_step, rule_ids_step)
        updated["rule_level_k"] = torch.cat([kv_cache["rule_level_k"], step_state["rule_level_k"]], dim=1)
        updated["rule_level_v"] = torch.cat([kv_cache["rule_level_v"], step_state["rule_level_v"]], dim=1)
        updated["rule_k_sum"] = kv_cache["rule_k_sum"] + step_state["rule_k_sum"]
        updated["rule_v_sum"] = kv_cache["rule_v_sum"] + step_state["rule_v_sum"]
        updated["rule_count"] = kv_cache["rule_count"] + step_state["rule_count"]
        return updated

    def _hierarchical_rule_attention_incremental(self, x_step: torch.Tensor, kv_cache) -> torch.Tensor:
        q_val = self.rule_level_q(x_step)
        rule_count = kv_cache["rule_count"].clamp_min(1.0)
        k_mean = kv_cache["rule_k_sum"] / rule_count.unsqueeze(-1)
        v_mean = kv_cache["rule_v_sum"] / rule_count.unsqueeze(-1)
        scale = 1.0 / math.sqrt(max(x_step.size(-1), 1))
        logits = torch.einsum('bqc,brc->bqr', q_val, k_mean) * scale
        seen_mask = kv_cache["rule_count"] > 0.5
        logits = logits.masked_fill(~seen_mask.unsqueeze(1), torch.finfo(logits.dtype).min)
        weights = F.softmax(logits, dim=-1)
        out = torch.einsum('bqr,brc->bqc', weights, v_mean)
        return out * torch.tanh(self.rule_level_gate)

    def _rule_attention_bias_incremental(self, query_rule_ids: torch.Tensor, all_rule_ids: torch.Tensor, dtype: torch.dtype):
        rq = self.rule_router_q[query_rule_ids]
        rk = self.rule_router_k[all_rule_ids]
        rq_norm = F.normalize(rq, p=2, dim=-1)
        rk_norm = F.normalize(rk, p=2, dim=-1)
        rule_bias = torch.matmul(rq_norm, rk_norm.transpose(-1, -2)) * math.sqrt(self.head_dim)
        return rule_bias.to(dtype).unsqueeze(1)

    def _rule_incremental_attention(self, x: torch.Tensor, target_rule_ids: torch.Tensor, start_pos: int, kv_cache=None):
        B, Seq, C = x.size()
        outputs = []
        cache = kv_cache
        for offset in range(Seq):
            x_step = x[:, offset:offset + 1, :]
            rule_step = target_rule_ids[:, offset:offset + 1]
            flat_x = x_step.reshape(-1, C)
            flat_rules = rule_step.reshape(-1)
            sort_ctx = compute_rule_sort_context(flat_rules)

            q_step = self.q_proj.forward_rules_batched(flat_x, flat_rules, sort_ctx=sort_ctx)
            k_step = self.k_proj.forward_rules_batched(flat_x, flat_rules, sort_ctx=sort_ctx)
            v_step = self.v_proj.forward_rules_batched(flat_x, flat_rules, sort_ctx=sort_ctx)

            q_step = q_step.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
            k_step = k_step.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
            v_step = v_step.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
            q_step, k_step = apply_rope(q_step, k_step, start_pos + offset, rope_base=self.rope_base)

            cache = self._append_rule_kv_cache(cache, x_step, rule_step, k_step, v_step)
            attn_bias = self._rule_attention_bias_incremental(rule_step, cache["rule_ids"], q_step.dtype)
            attn_out = self._dense_attention(q_step, cache["k"], cache["v"], 1, start_pos + offset, attn_bias=attn_bias)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, C)
            attn_out = attn_out + self._hierarchical_rule_attention_incremental(x_step, cache)

            out_step = self.out_proj.forward_rules_batched(attn_out.view(-1, C), flat_rules, sort_ctx=sort_ctx)
            outputs.append(out_step.view(B, 1, C))

        return torch.cat(outputs, dim=1), cache

    def _rule_causal_attention(self, x, target_rule_ids, start_pos, sort_ctx=None, build_cache=False):
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

        # Differentiable rule router: compute a soft additive bias to be injected
        # into scaled-dot-product attention.  Gradients flow naturally through
        # the softmax of SDPA, so rule_router_q/k learn on every backward pass.
        rq = self.rule_router_q[target_rule_ids]                # [B, Seq, rank]
        rk = self.rule_router_k[target_rule_ids]                # [B, Seq, rank]
        rq_norm = F.normalize(rq, p=2, dim=-1)
        rk_norm = F.normalize(rk, p=2, dim=-1)
        rule_affinity = torch.matmul(rq_norm, rk_norm.transpose(-1, -2))  # [B, Seq, Seq] in [-1,1]

        # Amplitude scales with head capacity so the bias is commensurate with
        # QK^T/sqrt(d) magnitudes.  No hand-tuned constants.
        bias_scale = math.sqrt(self.head_dim)
        rule_bias = rule_affinity * bias_scale                   # [B, Seq, Seq]

        # Promote to q's dtype *before* masking so the fill value is guaranteed
        # representable: under AMP autocast, q may be upcast to fp32 by
        # apply_rope while rule_bias stays in fp16.  Using finfo(fp32).min on a
        # fp16 tensor overflows (-3.4e38 → Inf on Half).  Casting first ensures
        # neg_inf == finfo(bias.dtype).min and avoids any silent NaN injection.
        rule_bias = rule_bias.to(q.dtype)
        causal_mask = torch.tril(torch.ones(Seq, Seq, device=q.device, dtype=torch.bool))
        neg_inf = torch.finfo(rule_bias.dtype).min
        rule_bias = rule_bias.masked_fill(~causal_mask.unsqueeze(0), neg_inf)
        attn_bias = rule_bias.unsqueeze(1)                       # [B, 1, Seq, Seq]

        attn_out = self._dense_attention(q, k, v, Seq, start_pos, attn_bias=attn_bias)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Seq, C)

        # Hierarchical rule-level path added *before* the output projection so
        # the per-rule out_proj adapter can still shape the combined signal.
        attn_out = attn_out + self._hierarchical_rule_attention(x, target_rule_ids)

        attn_out_flat = attn_out.view(-1, C)
        out = self.out_proj.forward_rules_batched(attn_out_flat, flat_rules, sort_ctx=sort_ctx)
        new_kv_cache = self._build_rule_kv_cache(x, target_rule_ids, k, v) if build_cache else None
        return out.view(B, Seq, C), new_kv_cache

    def forward(self, x, start_pos=0, kv_cache=None, target_rule_ids=None, sort_ctx=None):
        B, Seq, C = x.size()
        if target_rule_ids is not None:
            if kv_cache is None and Seq > 1:
                build_cache = (not self.training) or (not torch.is_grad_enabled())
                out, new_kv_cache = self._rule_causal_attention(
                    x, target_rule_ids, start_pos, sort_ctx=sort_ctx, build_cache=build_cache,
                )
            else:
                out, new_kv_cache = self._rule_incremental_attention(x, target_rule_ids, start_pos, kv_cache=kv_cache)
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
            # Old formula dim_feedforward // num_rules collapsed to ~4-16,
            # which is below the degrees of freedom needed for a useful FFN.
            # Lower bound by sqrt(dim_feedforward) so each expert keeps an
            # intrinsic capacity that scales with the residual stream width.
            capacity_floor = max(16, int(round(math.sqrt(max(dim_feedforward, 1)))))
            expert_dim = max(capacity_floor, dim_feedforward // max(num_rules, 1))
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
            int(expert_dim) if expert_dim is not None
            else max(max(16, int(round(math.sqrt(max(hidden_size, 1))))), hidden_size // max(num_rules, 1))
        )
        self.final_norm = CustomRMSNorm(current_dim)
        self.vq_layer = FiniteScalarQuantizer(fsq_levels, current_dim)
        self.rule_boundary_predictor = nn.Linear(current_dim, self.num_rules)
        self.truth_semantic_field = nn.Sequential(
            nn.Linear(current_dim, current_dim),
            nn.GELU(),
            nn.Linear(current_dim, embed_size),
        )
        # HRR path: a rule-typed, retrievable composition layer that runs in
        # parallel with the plain linear field.  Adds ~O(R·D + D²) parameters
        # and O(D log D) FLOPs per step — negligible compared with attention.
        self.holographic_bind = HolographicRuleBinding(self.num_rules, embed_size)

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
        linear_field = self.truth_semantic_field(memory)
        if target_rule_ids is not None:
            hrr_field = self.holographic_bind(linear_field, target_rule_ids)
            # Superpose: linear path carries magnitude, HRR path injects
            # rule-typed structure.  The HRR output already has per-channel
            # scale absorbed into out_scale, no extra gain needed.
            field_state = linear_field + hrr_field
        else:
            field_state = linear_field
        return rule_logits, field_state, memory, new_past_key_values

    def compute_rule_vocab_mask(self, token_chunk_embeds=None, token_chunk_ids=None, cell_state=None) -> CellState:
        device = next(self.parameters()).device
        if cell_state is None:
            return self._build_cell_state_vectorized(token_chunk_embeds, device)
        return self._rebuild_cell_state_vectorized(cell_state, token_chunk_embeds, token_chunk_ids, device)

    def _vq_rule_ids(self, embeds: torch.Tensor) -> torch.Tensor:
        """Single VQ forward pass that returns rule ids only (no STE grad)."""
        z = self.vq_layer.project_in(embeds)
        leak = 1.0 / max(self.vq_layer.d, 1)
        bound = torch.tanh(z) + leak * z / (1.0 + z.abs())
        bound = bound.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        scaled = (bound + 1.0) / 2.0 * (self.vq_layer.levels_t - 1.0)
        quantized = torch.round(scaled)
        return torch.sum(quantized.long() * self.vq_layer.basis, dim=-1)

    def _build_cell_state_vectorized(self, token_chunk_embeds: torch.Tensor, device: torch.device) -> CellState:
        """Fully vectorised initial cell state construction.

        Complexity: a single argsort (O(V log V)) + a handful of scatter ops.
        No Python-level iteration over rules, no .item() sync points.
        """
        max_tokens_per_rule = max(256, (self.vocab_size // max(self.num_rules, 1)) * 10)
        with torch.no_grad():
            chunk_sz = max(1, 100_000_000 // max(self.embed_size, 1))
            global_rules_list = []
            for start_idx in range(0, self.vocab_size, chunk_sz):
                end_idx = min(start_idx + chunk_sz, self.vocab_size)
                sub_embeds = rms_normalize(token_chunk_embeds[start_idx:end_idx]) * math.sqrt(self.embed_size)
                global_rules_list.append(self._vq_rule_ids(sub_embeds.to(device)))
            global_rules = torch.cat(global_rules_list)  # [V] rule id per token

        token_to_cell = global_rules.clone()
        counts = torch.bincount(global_rules, minlength=self.num_rules)  # [R]
        sort_idx = torch.argsort(global_rules, stable=True)              # tokens sorted by rule
        sorted_rules = global_rules[sort_idx]                            # [V] monotone-increasing (within each rule group)

        # Build per-rule offsets, then the local position of each sorted token within its rule.
        offsets = torch.zeros(self.num_rules + 1, dtype=torch.long, device=device)
        offsets[1:] = torch.cumsum(counts, dim=0)
        pos_within_rule_sorted = torch.arange(self.vocab_size, device=device) - offsets[sorted_rules]

        # Truncate overflow in a single vectorised gate (preserves the first
        # max_tokens_per_rule occurrences of each rule by sort order).
        keep_mask = pos_within_rule_sorted < max_tokens_per_rule
        kept_sort_idx = sort_idx[keep_mask]
        kept_rules = sorted_rules[keep_mask]
        kept_pos = pos_within_rule_sorted[keep_mask]

        cell_matrix = torch.zeros((self.num_rules, max_tokens_per_rule), dtype=torch.long, device=device)
        cell_matrix[kept_rules, kept_pos] = kept_sort_idx

        token_pos = torch.zeros(self.vocab_size, dtype=torch.long, device=device)
        token_pos[kept_sort_idx] = kept_pos

        cell_counts = counts.clamp_max(max_tokens_per_rule).clone()
        # Empty rules get 1 slot of fallback (pointing to token 0) to avoid
        # zero-size softmax / sampling collapses downstream.
        cell_counts = torch.where(cell_counts == 0, torch.ones_like(cell_counts), cell_counts)
        return CellState(cell_matrix, cell_counts, token_to_cell, token_pos)

    def _rebuild_cell_state_vectorized(
        self,
        cell_state: CellState,
        token_chunk_embeds: torch.Tensor,
        token_chunk_ids: torch.Tensor,
        device: torch.device,
    ) -> CellState:
        """Vectorised incremental rebuild.

        The old implementation used a Python for-loop over every changed token
        with four .item() syncs each, which blew O(K) GPU<->CPU round trips.
        We instead rebuild the affected rule rows from scratch using segmented
        scatter, which completes in O(V + R) on-device with zero per-token
        branching.
        """
        cell_matrix, cell_counts, token_to_cell, token_pos = cell_state
        max_tokens_per_rule = cell_matrix.size(1)
        with torch.no_grad():
            embeds = rms_normalize(token_chunk_embeds) * math.sqrt(self.embed_size)
            new_primary_rules = self._vq_rule_ids(embeds.to(device))

        # Apply the partial update into a copy of token_to_cell, then rebuild
        # the per-rule index from the full vocabulary.  Cost is O(V) regardless
        # of the chunk size, which amortises over many optimisation steps.
        updated_token_to_cell = token_to_cell.clone()
        updated_token_to_cell[token_chunk_ids] = new_primary_rules

        counts = torch.bincount(updated_token_to_cell, minlength=self.num_rules)
        sort_idx = torch.argsort(updated_token_to_cell, stable=True)
        sorted_rules = updated_token_to_cell[sort_idx]
        offsets = torch.zeros(self.num_rules + 1, dtype=torch.long, device=device)
        offsets[1:] = torch.cumsum(counts, dim=0)
        pos_within_rule_sorted = torch.arange(self.vocab_size, device=device) - offsets[sorted_rules]
        keep_mask = pos_within_rule_sorted < max_tokens_per_rule
        kept_sort_idx = sort_idx[keep_mask]
        kept_rules = sorted_rules[keep_mask]
        kept_pos = pos_within_rule_sorted[keep_mask]

        new_cell_matrix = torch.zeros_like(cell_matrix)
        new_cell_matrix[kept_rules, kept_pos] = kept_sort_idx
        new_token_pos = torch.zeros_like(token_pos)
        new_token_pos[kept_sort_idx] = kept_pos
        new_cell_counts = counts.clamp_max(max_tokens_per_rule).clone()
        new_cell_counts = torch.where(new_cell_counts == 0, torch.ones_like(new_cell_counts), new_cell_counts)

        return CellState(new_cell_matrix, new_cell_counts, updated_token_to_cell, new_token_pos)


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

