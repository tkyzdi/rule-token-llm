import glob
import math
import sys
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from rule_token_engine import RulePagedExpertStore, RuleTokenCausalModel
from tokenizer_utils import (
    discover_model_path,
    get_custom_tokenizer,
    get_logger,
    infer_model_config,
    load_checkpoint_payload,
    load_vocab_info,
    resolve_project_path,
    resolve_protocol_tokens,
)

logger = get_logger()


def build_rule_token_index(model, embedding_weight_cpu):
    cell_state = model.compute_rule_vocab_mask(token_chunk_embeds=embedding_weight_cpu)
    cell_matrix, cell_counts, token_to_cell, _ = cell_state
    cell_matrix_cpu = cell_matrix.cpu().tolist()
    cell_counts_cpu = cell_counts.cpu().tolist()
    rule_to_tokens = {}
    for rule_id in range(model.num_rules):
        c = cell_counts_cpu[rule_id]
        rule_to_tokens[rule_id] = cell_matrix_cpu[rule_id][:c]
    return rule_to_tokens, token_to_cell.detach().cpu()


def load_contexts(vocab_size):
    chunk_files = sorted(glob.glob(resolve_project_path("*_dataset_chunk_*.pt")))
    if not chunk_files:
        return []
    desired = max(1, int(round(math.log(vocab_size))))
    contexts = []
    chunks_to_sample = chunk_files[:desired] if len(chunk_files) >= desired else chunk_files
    samples_per_chunk = max(1, desired // len(chunks_to_sample))
    for chunk_file in chunks_to_sample:
        chunk_data = torch.load(chunk_file, map_location="cpu")
        step = max(1, len(chunk_data) // samples_per_chunk)
        for index in range(0, len(chunk_data), step):
            sequence = chunk_data[index]["sequence"].tolist()
            if len(sequence) > 1:
                prefix_len = max(1, int(math.sqrt(len(sequence) * max(math.log(vocab_size), 1.0))))
                contexts.append(sequence[: min(prefix_len, len(sequence) - 1)])
            if len(contexts) >= desired:
                break
        if len(contexts) >= desired:
            break
    return contexts


def decode_sequence(enc, special_tokens_inv, protocol_tokens, token_ids):
    parts = []
    normal_ids = []
    pad_id = protocol_tokens["pad"]["id"]
    format_symbols = set()
    for slot_parts in protocol_tokens["segment_slots"].values():
        for payload in slot_parts.values():
            format_symbols.add(payload["symbol"])
    for role_payload in protocol_tokens["role_slots"]:
        format_symbols.add(role_payload["symbol"])
    format_symbols.add(protocol_tokens["eos"]["symbol"])
    for token_id in token_ids:
        if token_id in special_tokens_inv:
            if normal_ids:
                parts.append(enc.decode(normal_ids))
                normal_ids = []
            if token_id != pad_id:
                symbol = special_tokens_inv[token_id]
                if symbol in format_symbols:
                    parts.append(f"\n{symbol}\n")
                else:
                    parts.append(symbol)
        else:
            if enc.is_normal_token_id(token_id):
                normal_ids.append(token_id)
    if normal_ids:
        parts.append(enc.decode(normal_ids))
    return "".join(parts)


def load_runtime_assets(checkpoint_path, model):
    payload = load_checkpoint_payload(checkpoint_path)
    state_dict = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    if isinstance(state_dict, dict):
        filtered_state = {k: v for k, v in state_dict.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
        model.load_state_dict(filtered_state, strict=False)

    expert_store = None
    if isinstance(payload, dict) and "expert_store" in payload:
        expert_store = RulePagedExpertStore(model.num_layers, model.num_rules, model.embed_size, model.expert_dim)
        expert_state = payload["expert_store"]
        current_state = expert_store.state_dict()
        filtered_expert_state = {k: v for k, v in expert_state.items() if k in current_state and current_state[k].shape == v.shape}
        if filtered_expert_state:
            current_state.update(filtered_expert_state)
            expert_store.load_state_dict(current_state, strict=False)
        model.offload_experts_to_runtime()

    if not isinstance(payload, dict) or "cpu_token_embedding.weight" not in payload:
        raise KeyError("检查点中未找到 cpu_token_embedding.weight。")
    embedding_weight_cpu = payload["cpu_token_embedding.weight"].float().cpu()
    # Pin host memory so repeated `.to(device, non_blocking=True)` gathers in
    # the hot generation loop can overlap with compute.  Safe no-op on CPU-only
    # machines.
    if torch.cuda.is_available():
        try:
            embedding_weight_cpu = embedding_weight_cpu.pin_memory()
        except RuntimeError:
            pass

    return expert_store, embedding_weight_cpu


def load_active_tokens():
    active_tokens = set()
    chunk_files = glob.glob(resolve_project_path("*_dataset_chunk_*.pt"))
    for chunk_file in chunk_files:
        try:
            chunk_data = torch.load(chunk_file, map_location="cpu", weights_only=False)
            for item in chunk_data:
                seq = item.get("sequence", [])
                if isinstance(seq, torch.Tensor):
                    active_tokens.update(seq.tolist())
                elif isinstance(seq, list):
                    active_tokens.update(seq)
        except Exception:
            pass
    return active_tokens


def _trim_rule_kv_cache(kv_cache, keep: int):
    if kv_cache is None:
        return None
    if not isinstance(kv_cache, dict):
        k_cache, v_cache = kv_cache
        return (k_cache[:, :, -keep:, :], v_cache[:, :, -keep:, :])

    trimmed = {
        "k": kv_cache["k"][:, :, -keep:, :],
        "v": kv_cache["v"][:, :, -keep:, :],
        "rule_ids": kv_cache["rule_ids"][:, -keep:],
        "rule_level_k": kv_cache["rule_level_k"][:, -keep:, :],
        "rule_level_v": kv_cache["rule_level_v"][:, -keep:, :],
    }
    num_rules = kv_cache["rule_count"].size(1)
    membership = F.one_hot(trimmed["rule_ids"].to(torch.long), num_classes=num_rules).to(trimmed["rule_level_k"].dtype)
    trimmed["rule_count"] = membership.sum(dim=1)
    trimmed["rule_k_sum"] = torch.einsum('bsr,bsc->brc', membership, trimmed["rule_level_k"])
    trimmed["rule_v_sum"] = torch.einsum('bsr,bsc->brc', membership, trimmed["rule_level_v"])
    return trimmed

def build_rule_candidate_cache(
    rule_to_tokens: Dict[int, List[int]],
    embedding_weight_cpu: torch.Tensor,
    device: torch.device,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Pre-materialise per-rule candidate state on the inference device.

    The original hot loop rebuilt a CPU long tensor + H2D copy + L2 norm every
    step; for a sample generating N tokens with K candidates per rule this
    costs O(N·(K + H2D latency)) and dominates wall-clock.  Since rule_to_tokens
    is constant during evaluation we compute each rule's candidate ids and
    L2-normalised embeddings exactly once.  Memory cost = Σ_r |tokens_r| · D,
    i.e. at most O(V·D) — identical to the cpu_token_embedding table itself.
    """
    cache: Dict[int, Dict[str, torch.Tensor]] = {}
    for rule_id, token_ids in rule_to_tokens.items():
        if not token_ids:
            continue
        tokens_long = torch.tensor(token_ids, dtype=torch.long)
        embeds = embedding_weight_cpu[tokens_long].to(device=device, dtype=torch.float32)
        embeds = F.normalize(embeds, p=2, dim=-1)
        cache[int(rule_id)] = {
            "tokens": tokens_long.to(device),
            "norm_embeds": embeds,
        }
    return cache


@torch.inference_mode()
def generate_tokens(
    model: RuleTokenCausalModel,
    expert_store: Optional[RulePagedExpertStore],
    embedding_weight_cpu: torch.Tensor,
    token_to_cell: torch.Tensor,
    rule_to_tokens: Dict[int, List[int]],
    ctx_tokens: List[int],
    eos_id: int,
    device: torch.device,
    max_new_tokens: int,
    rule_cache: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    token_to_cell_device: Optional[torch.Tensor] = None,
) -> List[int]:
    """Single-prompt autoregressive generation.

    Fast-path contract:
      * `rule_cache` — output of build_rule_candidate_cache; avoids per-step
        H2D copies and normalisation of candidate embeddings.
      * `token_to_cell_device` — token→rule map already resident on `device`.
      * If `expert_store` has been warmed with all active rules (cache pre-
        loaded), the per-step `build_runtime([current_rule], ...)` is a pure
        in-memory dict lookup and effectively free.
    """
    max_ctx_len = max(1, int(model.embed_size * max(math.log(model.num_rules), 1.0)))
    ctx_tokens = ctx_tokens[-max_ctx_len:]

    if rule_cache is None:
        rule_cache = build_rule_candidate_cache(rule_to_tokens, embedding_weight_cpu, device)
    if token_to_cell_device is None:
        token_to_cell_device = token_to_cell.to(device)

    ctx_tensor_cpu = torch.tensor([ctx_tokens], dtype=torch.long)
    ctx_embeds = embedding_weight_cpu[ctx_tensor_cpu].to(device)
    ctx_rules = token_to_cell_device[ctx_tensor_cpu.to(device)]
    warm_rules = sorted(set(ctx_rules.view(-1).tolist()))
    runtime_pages = expert_store.build_runtime(warm_rules, device=device, training=False) if expert_store is not None else None

    generated_tokens: List[int] = []
    _, _, _, past_key_values = model(ctx_embeds, target_rule_ids=ctx_rules, runtime_expert_pages=runtime_pages)
    rope_offset = ctx_tensor_cpu.size(1)
    current_rule = int(ctx_rules[0, -1].item()) if ctx_rules.numel() > 0 else 0

    # Rule mask restricted to rules with at least one active candidate.
    valid_rule_mask = None
    if len(rule_cache) < model.num_rules:
        valid_rule_mask = torch.zeros(model.num_rules, dtype=torch.bool, device=device)
        valid_rules_tensor = torch.tensor(list(rule_cache.keys()), dtype=torch.long, device=device)
        valid_rule_mask[valid_rules_tensor] = True

    # Pre-allocate scratch tensors reused every step to avoid Python-side
    # tensor construction in the hot loop.  CPU scratch is used for gathering
    # from `embedding_weight_cpu` (which by design stays on host RAM), the
    # device scratch carries the target rule id into the forward pass.
    cpu_scratch = torch.zeros(1, 1, dtype=torch.long)
    scratch_rule = torch.zeros(1, 1, dtype=torch.long, device=device)

    capacity_scaler = 1.0 / max(math.log(model.embed_size), 1.0)
    temp_floor = math.exp(-capacity_scaler)

    last_token = ctx_tokens[-1]
    for _ in range(max_new_tokens):
        cpu_scratch[0, 0] = last_token
        scratch_rule.fill_(current_rule)
        current_input_embeds = embedding_weight_cpu[cpu_scratch].to(device, non_blocking=True)
        runtime_pages = expert_store.build_runtime([current_rule], device=device, training=False) if expert_store is not None else None

        rule_logits, truth_field_state, _, past_key_values = model(
            current_input_embeds,
            target_rule_ids=scratch_rule,
            start_pos=rope_offset,
            past_key_values=past_key_values,
            runtime_expert_pages=runtime_pages,
        )

        rule_probs = torch.softmax(rule_logits[0, -1], dim=-1)
        if torch.isnan(rule_probs).any():
            print("\n[警告] 模型输出了 NaN，这通常意味着模型在训练过程中崩溃（梯度爆炸/消失），其权重已经损坏。你需要重新训练模型。")
            break

        if valid_rule_mask is not None:
            rule_probs = rule_probs * valid_rule_mask
            prob_sum = rule_probs.sum()
            if prob_sum > 0:
                rule_probs = rule_probs / prob_sum
            else:
                break

        rule_id = int(torch.multinomial(rule_probs, 1).item())
        cache_entry = rule_cache.get(rule_id)
        if cache_entry is None:
            break

        candidate_tokens_gpu = cache_entry["tokens"]           # [K] long on device
        candidate_embeddings = cache_entry["norm_embeds"]       # [K, D] on device (already unit-norm)
        field_vector = truth_field_state[0, -1]
        field_norm = F.normalize(field_vector, p=2, dim=-1)
        candidate_similarity = torch.matmul(candidate_embeddings, field_norm)
        K = candidate_similarity.numel()
        eps = torch.finfo(candidate_similarity.dtype).eps

        # std() is only defined for K≥2; for K=1 fall back to deterministic pick.
        if K <= 1:
            token_index = 0
        else:
            similarity_std = candidate_similarity.std(unbiased=False).item() + eps
            temperature = max(math.exp(-similarity_std), temp_floor)
            scaled_similarity = (candidate_similarity / temperature)
            scaled_similarity = scaled_similarity - scaled_similarity.max()
            token_probs = torch.softmax(scaled_similarity, dim=-1)

            tp_std = token_probs.std(unbiased=False).item()
            dynamic_p = 1.0 - math.exp(-capacity_scaler / max(tp_std * math.sqrt(K), eps))
            sorted_probs, sorted_indices = torch.sort(token_probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative_probs > dynamic_p
            sorted_mask[0] = False
            token_probs[sorted_indices[sorted_mask]] = 0.0
            probability_mass = token_probs.sum()
            if not torch.isfinite(probability_mass) or probability_mass <= eps:
                token_index = int(torch.argmax(candidate_similarity).item())
            else:
                token_probs = token_probs / probability_mass
                token_index = int(torch.multinomial(token_probs, 1).item())

        token_id = int(candidate_tokens_gpu[token_index].item())
        generated_tokens.append(token_id)
        # Keep rule state on device — token_to_cell_device is already resident.
        if token_id < token_to_cell_device.numel():
            current_rule = int(token_to_cell_device[token_id].item())
        else:
            current_rule = rule_id
        last_token = token_id
        rope_offset += 1

        if token_id == eos_id:
            break

        if past_key_values and past_key_values[0] is not None:
            first_cache = past_key_values[0]
            kv_len = first_cache["k"].size(2) if isinstance(first_cache, dict) else first_cache[0].size(2)
        else:
            kv_len = 0
        if kv_len >= max_ctx_len:
            keep = max_ctx_len - 1
            past_key_values = [_trim_rule_kv_cache(layer_cache, keep) for layer_cache in past_key_values]

    return generated_tokens


def run_large_scale_inference():
    vocab = load_vocab_info()
    enc, special_tokens = get_custom_tokenizer()
    protocol_tokens = resolve_protocol_tokens(vocab)
    special_tokens_inv = {value: key for key, value in special_tokens.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = discover_model_path()
    model_config = infer_model_config(vocab["vocab_size"], checkpoint_path)
    model = RuleTokenCausalModel(vocab_size=vocab["vocab_size"], **model_config).to(device)
    expert_store, embedding_weight_cpu = load_runtime_assets(checkpoint_path, model)
    model.eval()

    rule_to_tokens, token_to_cell = build_rule_token_index(model, embedding_weight_cpu)
    
    logger.info("正在加载训练集活跃 Token 词表 (用于过滤未训练的随机初始化 Token)...")
    active_tokens = load_active_tokens()
    if not active_tokens:
        logger.warning("未能加载到活跃 Token 集合，推理可能会输出随机乱码！")
    else:
        logger.info(f"成功加载 {len(active_tokens)} 个活跃 Token。")
        for rule_id in list(rule_to_tokens.keys()):
            valid_t = [t for t in rule_to_tokens[rule_id] if t in active_tokens]
            if valid_t:
                rule_to_tokens[rule_id] = valid_t
            else:
                del rule_to_tokens[rule_id]

    max_new_tokens = max(1, int(model.embed_size * max(math.log(model.num_rules), 1.0)))
    eos_id = protocol_tokens["eos"]["id"]

    interactive = sys.stdin.isatty() or "--interactive" in sys.argv

    if interactive:
        logger.info("进入交互模式 (输入 'exit' 退出)...")
        while True:
            try:
                prompt = input("\n[Prompt] > ")
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.strip().lower() in ("exit", "quit", "q"):
                break
            if not prompt.strip():
                continue
            ctx_tokens = enc.encode(prompt, allowed_special="all")
            if not ctx_tokens:
                continue
            with torch.no_grad():
                generated = generate_tokens(
                    model, expert_store, embedding_weight_cpu,
                    token_to_cell, rule_to_tokens,
                    ctx_tokens, eos_id, device, max_new_tokens,
                )
            output_str = decode_sequence(enc, special_tokens_inv, protocol_tokens, generated)
            try:
                print(f"[输出] {output_str}")
            except UnicodeEncodeError:
                print(f"[输出] {output_str}".encode('utf-8', 'replace').decode(sys.stdout.encoding or 'gbk', 'replace'))
    else:
        contexts = load_contexts(vocab["vocab_size"])
        if not contexts:
            raise FileNotFoundError("未发现可用的数据分块，无法构建自适应推理语境。")

        with torch.no_grad():
            for case_id, ctx_tokens in enumerate(contexts, start=1):
                generated = generate_tokens(
                    model, expert_store, embedding_weight_cpu,
                    token_to_cell, rule_to_tokens,
                    ctx_tokens, eos_id, device, max_new_tokens,
                )
                context_str = decode_sequence(enc, special_tokens_inv, protocol_tokens, ctx_tokens)
                output_str = decode_sequence(enc, special_tokens_inv, protocol_tokens, generated)
                try:
                    print(f"用例 {case_id}:")
                    print(f"  [模型看到的前置语境] {context_str}")
                    print(f"  [连续自回归输出] {output_str}")
                    print("-" * 80)
                except UnicodeEncodeError:
                    print(f"用例 {case_id}:")
                    print(f"  [模型看到的前置语境] {context_str}".encode('utf-8', 'replace').decode(sys.stdout.encoding or 'gbk', 'replace'))
                    print(f"  [连续自回归输出] {output_str}".encode('utf-8', 'replace').decode(sys.stdout.encoding or 'gbk', 'replace'))
                    print("-" * 80)


if __name__ == "__main__":
    run_large_scale_inference()
