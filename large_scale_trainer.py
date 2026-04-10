import glob
import argparse
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

from rule_token_engine import RuleAwareProjection, RulePagedExpertStore, RuleTokenCausalModel, compute_rule_sort_context
from tokenizer_utils import (
    atomic_torch_save,
    derive_seed,
    discover_model_path,
    get_logger,
    infer_model_config,
    load_checkpoint_payload,
    load_vocab_info,
    next_power_of_two,
    rms_normalize,
    resolve_project_path,
    resolve_protocol_tokens,
    set_deterministic_seed,
)

logger = get_logger()


class StreamingDataset(IterableDataset):
    def __init__(self, chunk_files: List[str]):
        self.chunk_files = chunk_files

    def __iter__(self):
        shuffled_files = list(self.chunk_files)
        random.shuffle(shuffled_files)
        for chunk_file in shuffled_files:
            logger.info("正在流式加载分块: %s ...", chunk_file)
            data_chunk = torch.load(chunk_file, map_location="cpu")
            random.shuffle(data_chunk)
            for item in data_chunk:
                yield item


@dataclass
class TrainingStage:
    name: str
    chunk_files: List[str]
    assistant_only_loss: bool
    estimated_seq_len: int


@dataclass
class AdaptiveRuleState:
    enabled: bool
    initial_num_rules: int
    warmup_optimizer_steps: int
    min_target_rules: int
    post_adapt_rule_only_steps: int
    token_lr_scale: float
    token_gate_entropy_ratio: float
    token_gate_patience: int
    adapted: bool = False
    warmup_rule_usage: Optional[torch.Tensor] = None
    warmup_entropy: Optional[List[float]] = None
    total_warmup_tokens: int = 0
    target_num_rules: Optional[int] = None
    post_adapt_rule_only_until_update: int = 0
    token_phase_enabled: bool = False
    post_adapt_rule_history: Optional[List[float]] = None
    token_update_interval: int = 4
    token_revert_entropy_ratio: float = 0.92
    token_revert_patience: int = 2
    token_cooldown_steps: int = 4
    next_token_update_step: int = 0

    def ensure_buffers(self, num_rules: int):
        if self.warmup_rule_usage is None or self.warmup_rule_usage.numel() != num_rules:
            self.warmup_rule_usage = torch.zeros(num_rules, dtype=torch.long)
        if self.warmup_entropy is None:
            self.warmup_entropy = []
        if self.post_adapt_rule_history is None:
            self.post_adapt_rule_history = []


@dataclass
class GradientTracker:
    """Log-space EMA for gradient norms with bounded-variance guarantees.

    Gradient norms in high-dimensional spaces follow a log-normal-like
    distribution (multiplicative CLT).  All tracking happens in log-space
    which compresses outliers and maps the multiplicative process to an
    additive one.

    Key invariants maintained by this implementation:
      1. log_ema ∈ ℝ  (never inf/nan — non-finite inputs are rejected)
      2. log_var ∈ [0, 4]  (bounded variance prevents death spirals)
      3. adaptive_max_norm ≤ exp(log_ema + 2)  (σ capped at 1.0 breaks
         the positive feedback loop: instability → high var → high clip → …)
      4. lr_damping ∈ [0.047, 0.98]  (sigmoid with center=1.5, slope=3;
         gentler than the original, prevents the LR from collapsing to zero
         during transient instability)

    SNR = 1/(exp(σ²)-1) from Var[X] = E[X]²·(exp(σ²)-1) for log-normal X.
    """
    log_ema: float = 0.0
    log_var: float = 0.1
    initialized: bool = False
    alpha: float = 0.1

    def update(self, grad_norm: float):
        if not math.isfinite(grad_norm) or grad_norm <= 0:
            return
        ln = math.log(grad_norm)
        if not self.initialized:
            self.log_ema = ln
            self.log_var = 0.1
            self.initialized = True
        else:
            delta = ln - self.log_ema
            self.log_var = (1 - self.alpha) * self.log_var + self.alpha * min(delta * delta, 2.0)
            self.log_var = min(self.log_var, 4.0)
            self.log_ema = (1 - self.alpha) * self.log_ema + self.alpha * ln

    @property
    def adaptive_max_norm(self) -> float:
        if not math.isfinite(self.log_ema):
            return 1.0
        sigma = math.sqrt(max(self.log_var, 0.04))
        return math.exp(self.log_ema + 2.0 * min(sigma, 1.0))

    @property
    def signal_to_noise(self) -> float:
        v = min(self.log_var, 10.0)
        ev = math.exp(v) - 1.0
        if ev < 1e-12:
            return float("inf")
        return 1.0 / ev

    @property
    def lr_damping(self) -> float:
        v = min(self.log_var, 4.0)
        return 1.0 / (1.0 + math.exp(3.0 * (v - 1.5)))


COMPILE_PROBE_CACHE: Dict[str, Tuple[bool, str]] = {}


def discover_training_stages() -> List[TrainingStage]:
    real_chunk_files = sorted(glob.glob(resolve_project_path("real_dataset_chunk_*.pt")))
    sft_chunk_files = sorted(glob.glob(resolve_project_path("sft_dataset_chunk_*.pt")))
    stages: List[TrainingStage] = []

    if real_chunk_files:
        stages.append(TrainingStage(
            name="real",
            chunk_files=real_chunk_files,
            assistant_only_loss=False,
            estimated_seq_len=load_estimated_seq_len(real_chunk_files),
        ))
    if sft_chunk_files:
        stages.append(TrainingStage(
            name="sft",
            chunk_files=sft_chunk_files,
            assistant_only_loss=True,
            estimated_seq_len=load_estimated_seq_len(sft_chunk_files),
        ))

    if not stages:
        raise FileNotFoundError(
            "未找到 real_dataset_chunk_*.pt 或 sft_dataset_chunk_*.pt，请先运行 build_real_dataset.py / build_sft_dataset.py。"
        )
    return stages


def build_assistant_target_mask(sequence: torch.Tensor, assistant_role_id: int, role_token_ids: set, eos_id: int) -> torch.Tensor:
    role_state = "other"
    token_mask = torch.zeros(sequence.size(0), dtype=torch.bool)
    for index, token in enumerate(sequence.tolist()):
        if token == assistant_role_id:
            role_state = "assistant"
            token_mask[index] = True
            continue
        if token in role_token_ids:
            role_state = "other"
            continue
        if token == eos_id:
            token_mask[index] = role_state == "assistant"
            role_state = "other"
            continue
        token_mask[index] = role_state == "assistant"
    return token_mask[1:]


def build_collate_fn(
    pad_token_id: int,
    assistant_only_loss: bool = False,
    assistant_role_id: Optional[int] = None,
    role_token_ids: Optional[set] = None,
    eos_id: Optional[int] = None,
):
    def collate_fn(batch):
        sequences = [item["sequence"].long() for item in batch]
        padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=pad_token_id)
        if assistant_only_loss:
            if assistant_role_id is None or role_token_ids is None or eos_id is None:
                raise ValueError("SFT assistant-only loss 需要 assistant_role_id、role_token_ids 与 eos_id。")
            loss_masks = [
                build_assistant_target_mask(sequence, assistant_role_id, role_token_ids, eos_id)
                for sequence in sequences
            ]
        else:
            loss_masks = [torch.ones(max(sequence.size(0) - 1, 0), dtype=torch.bool) for sequence in sequences]
        padded_loss_masks = torch.nn.utils.rnn.pad_sequence(loss_masks, batch_first=True, padding_value=False)
        return {
            "context_tokens": padded_sequences[:, :-1],
            "tokens": padded_sequences[:, 1:],
            "loss_mask": padded_loss_masks,
        }
    return collate_fn


def get_compile_probe_key(device: torch.device) -> str:
    if device.type != "cuda":
        return device.type
    device_index = device.index
    if device_index is None and torch.cuda.is_available():
        device_index = torch.cuda.current_device()
    return f"{device.type}:{device_index}"


def configure_torch_compile_fallback(enabled: bool = True) -> Optional[bool]:
    dynamo = getattr(torch, "_dynamo", None)
    config = getattr(dynamo, "config", None)
    if config is not None and hasattr(config, "suppress_errors"):
        previous = bool(config.suppress_errors)
        config.suppress_errors = enabled
        return previous
    return None


def probe_torch_compile(device: torch.device) -> Tuple[bool, str]:
    cache_key = get_compile_probe_key(device)
    cached = COMPILE_PROBE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not hasattr(torch, "compile"):
        result = (False, "当前 PyTorch 版本不提供 torch.compile。")
        COMPILE_PROBE_CACHE[cache_key] = result
        return result

    probe = None
    compiled_probe = None
    sample = None
    previous_suppress_errors = configure_torch_compile_fallback(False)
    try:
        probe = nn.Sequential(
            nn.Linear(8, 8),
            nn.GELU(),
            nn.Linear(8, 8),
        ).to(device).eval()
        compiled_probe = torch.compile(probe)
        with torch.no_grad():
            sample = torch.randn(2, 8, device=device)
            compiled_probe(sample)
        result = (True, "")
    except Exception as exc:
        result = (False, f"{type(exc).__name__}: {exc}")
    finally:
        if previous_suppress_errors is not None:
            configure_torch_compile_fallback(previous_suppress_errors)
        del sample, compiled_probe, probe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    COMPILE_PROBE_CACHE[cache_key] = result
    return result


def maybe_enable_torch_compile(model: nn.Module, device: torch.device, scope: str) -> nn.Module:
    can_compile, reason = probe_torch_compile(device)
    if not can_compile:
        logger.info("[TorchCompile] %s 未启用编译，已自动回退 eager。原因: %s", scope, reason)
        return model

    configure_torch_compile_fallback(True)
    try:
        compiled_model = torch.compile(model)
    except Exception as exc:
        logger.warning("[TorchCompile] %s 编译包装失败: %s", scope, exc)
        return model

    logger.info("[TorchCompile] %s 已启用 torch.compile。", scope)
    return compiled_model


def infer_training_schedule(model: RuleTokenCausalModel, estimated_seq_len: int, vocab_size: int, device: torch.device) -> Tuple[int, int]:
    if device.type == "cuda":
        total_memory = torch.cuda.get_device_properties(device).total_memory
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        optimizer_bytes = param_bytes * 2
        usable_memory = max(1, total_memory - param_bytes - optimizer_bytes)
        scalar_bytes = 2
    else:
        usable_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        scalar_bytes = 4
    activation_per_token = (model.embed_size + model.hidden_size + model.num_rules) * model.num_layers * model.num_heads * scalar_bytes
    batch_size = max(1, int(math.sqrt(usable_memory / max(activation_per_token * max(estimated_seq_len, 1), 1))))
    target_effective_batch = max(batch_size, int(math.sqrt(vocab_size) / max(math.log(vocab_size), 1.0)))
    accumulation_steps = max(1, math.ceil(target_effective_batch / batch_size))
    return batch_size, accumulation_steps


def infer_training_windows(estimated_seq_len: int, device: torch.device) -> Tuple[int, int]:
    if device.type == "cuda":
        total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    else:
        total_gb = 8.0
    if total_gb <= 12:
        suffix = max(128, estimated_seq_len // 4)
    elif total_gb <= 24:
        suffix = max(256, estimated_seq_len // 3)
    else:
        suffix = max(384, estimated_seq_len // 2)
    suffix = min(estimated_seq_len, suffix)
    context = min(estimated_seq_len, max(suffix, suffix * 2))
    return context, suffix


def load_estimated_seq_len(chunk_files: List[str]) -> int:
    sample_lengths: List[int] = []
    max_samples = max(1, int(math.sqrt(10000)))
    sampled_files = random.sample(chunk_files, min(len(chunk_files), 5))
    samples_per_chunk = max(1, max_samples // len(sampled_files))
    for chunk_file in sampled_files:
        chunk_data = torch.load(chunk_file, map_location="cpu")
        sample_lengths.extend([len(item["sequence"]) for item in chunk_data[:samples_per_chunk]])
    return max(1, int(sum(sample_lengths) / len(sample_lengths)))


def smooth_rule_step(value: int) -> int:
    value = max(1, int(value))
    if value <= 256:
        return 16
    if value <= 512:
        return 32
    return 64


def smooth_floor_rule_count(value: float, minimum: int = 32, maximum: Optional[int] = None) -> int:
    value = max(float(minimum), float(value))
    step = smooth_rule_step(int(value))
    floored = max(step, int(value // step) * step)
    floored = max(minimum, floored)
    if maximum is not None:
        floored = min(int(maximum), floored)
    return int(floored)


def build_sorted_active_embeddings(
    cpu_token_embedding: nn.Embedding,
    active_token_ids_cpu: torch.Tensor,
    device: torch.device,
    train_token: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    active_token_ids_cpu = torch.unique(active_token_ids_cpu.to(torch.long).cpu(), sorted=True)
    active_embeds_cpu = cpu_token_embedding(active_token_ids_cpu)
    active_embeds_gpu = active_embeds_cpu.to(device, non_blocking=True)
    if not train_token:
        active_embeds_gpu = active_embeds_gpu.detach()
    active_ids_gpu = active_token_ids_cpu.to(device, non_blocking=True)
    return active_token_ids_cpu, active_ids_gpu, active_embeds_cpu, active_embeds_gpu


def lookup_local_embeddings(token_ids: torch.Tensor, active_ids_gpu: torch.Tensor, active_embeds_gpu: torch.Tensor) -> torch.Tensor:
    flat = token_ids.reshape(-1).to(device=active_ids_gpu.device, dtype=torch.long)
    positions = torch.searchsorted(active_ids_gpu, flat)
    gathered = active_embeds_gpu.index_select(0, positions)
    return gathered.view(*token_ids.shape, active_embeds_gpu.size(-1))


def prepare_training_window(
    ctx_ids_cpu: torch.Tensor,
    target_ids_cpu: torch.Tensor,
    loss_mask_cpu: torch.Tensor,
    train_context_len: int,
    train_suffix_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_len = target_ids_cpu.size(1)
    if seq_len > train_context_len:
        start = seq_len - train_context_len
        ctx_ids_cpu = ctx_ids_cpu[:, start:]
        target_ids_cpu = target_ids_cpu[:, start:]
        loss_mask_cpu = loss_mask_cpu[:, start:]
    local_seq = target_ids_cpu.size(1)
    focus_start = max(0, local_seq - train_suffix_len)
    focus_mask = torch.zeros((target_ids_cpu.size(0), local_seq), dtype=torch.bool)
    focus_mask[:, focus_start:] = True
    return ctx_ids_cpu, target_ids_cpu, loss_mask_cpu, focus_mask


def load_checkpoint_into_runtime(
    checkpoint_path: str,
    model: RuleTokenCausalModel,
    cpu_token_embedding: nn.Embedding,
    expert_store: RulePagedExpertStore,
) -> Dict:
    payload = load_checkpoint_payload(checkpoint_path)
    if payload is None:
        return {}
    state_dict = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    if isinstance(state_dict, dict):
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
        model.load_state_dict(filtered_state_dict, strict=False)
    if isinstance(payload, dict) and "expert_store" in payload:
        expert_state = payload["expert_store"]
        current_state = expert_store.state_dict()
        filtered_expert_state = {k: v for k, v in expert_state.items() if k in current_state and current_state[k].shape == v.shape}
        if filtered_expert_state:
            current_state.update(filtered_expert_state)
            expert_store.load_state_dict(current_state, strict=False)
            model.offload_experts_to_runtime()
    if isinstance(payload, dict) and "cpu_token_embedding.weight" in payload:
        loaded_weight = payload["cpu_token_embedding.weight"]
        if loaded_weight.shape == cpu_token_embedding.weight.shape:
            cpu_token_embedding.weight.data.copy_(loaded_weight)
        else:
            logger.warning("跳过加载 Token Embedding，形状不匹配 (%s != %s)", loaded_weight.shape, cpu_token_embedding.weight.shape)
    return payload if isinstance(payload, dict) else {}


def infer_adaptive_state(
    vocab_size: int,
    batch_size: int,
    accumulation_steps: int,
    train_suffix_len: int,
    configured_num_rules: int,
    checkpoint_payload: Dict,
) -> Tuple[AdaptiveRuleState, int]:
    tokens_per_opt_step = batch_size * train_suffix_len * accumulation_steps
    saved_state = checkpoint_payload.get("adaptive_rule_state", {}) if isinstance(checkpoint_payload, dict) else {}
    saved_token_lr_scale = float(saved_state.get("token_lr_scale", 0.15)) if saved_state else 0.15
    saved_post_adapt_steps = int(saved_state.get("post_adapt_rule_only_steps", 0)) if saved_state else 0
    saved_gate_ratio = float(saved_state.get("token_gate_entropy_ratio", 0.86)) if saved_state else 0.86
    saved_gate_patience = int(saved_state.get("token_gate_patience", 2)) if saved_state else 2
    default_token_interval = max(3, min(6, int(math.ceil(math.log2(max(configured_num_rules, 2) + 1) / 2))))
    saved_token_interval = int(saved_state.get("token_update_interval", default_token_interval)) if saved_state else default_token_interval
    saved_revert_ratio = float(saved_state.get("token_revert_entropy_ratio", min(0.96, saved_gate_ratio + 0.06))) if saved_state else min(0.96, saved_gate_ratio + 0.06)
    saved_revert_patience = int(saved_state.get("token_revert_patience", max(2, saved_gate_patience))) if saved_state else max(2, saved_gate_patience)
    saved_token_cooldown = int(saved_state.get("token_cooldown_steps", max(saved_post_adapt_steps, saved_token_interval))) if saved_state else max(saved_post_adapt_steps, saved_token_interval)
    saved_token_phase_enabled = bool(saved_state.get("token_phase_enabled", True)) if saved_state else True
    if isinstance(checkpoint_payload, dict) and saved_state.get("adapted", False):
        resumed_rule_only_until = 0 if saved_token_phase_enabled else max(1, saved_token_cooldown)
        final_rules = int(saved_state.get("target_num_rules", configured_num_rules))
        state = AdaptiveRuleState(
            enabled=False,
            initial_num_rules=final_rules,
            warmup_optimizer_steps=0,
            min_target_rules=max(32, min(final_rules, 128)),
            post_adapt_rule_only_steps=max(0, saved_post_adapt_steps),
            token_lr_scale=saved_token_lr_scale,
            token_gate_entropy_ratio=saved_gate_ratio,
            token_gate_patience=max(1, saved_gate_patience),
            adapted=True,
            target_num_rules=final_rules,
            post_adapt_rule_only_until_update=resumed_rule_only_until,
            token_phase_enabled=saved_token_phase_enabled,
            post_adapt_rule_history=list(saved_state.get("post_adapt_rule_history", [])),
            token_update_interval=max(1, saved_token_interval),
            token_revert_entropy_ratio=saved_revert_ratio,
            token_revert_patience=max(1, saved_revert_patience),
            token_cooldown_steps=max(1, saved_token_cooldown),
            next_token_update_step=(max(1, saved_token_interval) if saved_token_phase_enabled else resumed_rule_only_until + max(1, saved_token_interval)),
        )
        state.ensure_buffers(final_rules)
        return state, final_rules

    # 使用第一性原理重新推导初始规则数
    base_entropy = math.log2(max(vocab_size, 2))
    embed_size_raw = base_entropy * base_entropy
    embed_size = next_power_of_two(int(embed_size_raw))
    num_rules_raw = math.sqrt(embed_size) * math.log(max(vocab_size, 2))
    adaptive_upper = next_power_of_two(int(num_rules_raw))
    adaptive_upper = min(configured_num_rules, adaptive_upper)
    adaptive_upper = max(32, adaptive_upper)
    warmup_optimizer_steps = max(4, min(12, int(math.ceil(math.log2(adaptive_upper + 1)))))
    raw_min_target = max(32.0, math.sqrt(max(tokens_per_opt_step, 1)) / 2.0)
    min_target_rules = smooth_floor_rule_count(
        min(raw_min_target, max(64.0, adaptive_upper / 2.0)),
        minimum=32,
        maximum=adaptive_upper,
    )
    post_adapt_rule_only_steps = max(2, min(6, int(math.ceil(math.log2(adaptive_upper + 1)) / 2)))
    token_update_interval = max(3, min(6, int(math.ceil(math.log2(adaptive_upper + 1) / 2))))
    state = AdaptiveRuleState(
        enabled=True,
        initial_num_rules=adaptive_upper,
        warmup_optimizer_steps=warmup_optimizer_steps,
        min_target_rules=min_target_rules,
        post_adapt_rule_only_steps=post_adapt_rule_only_steps,
        token_lr_scale=0.15,
        token_gate_entropy_ratio=0.86,
        token_gate_patience=2,
        token_update_interval=token_update_interval,
        token_revert_entropy_ratio=0.92,
        token_revert_patience=2,
        token_cooldown_steps=max(post_adapt_rule_only_steps, token_update_interval),
    )
    return state, adaptive_upper


def decide_adaptive_num_rules(
    adaptive_state: AdaptiveRuleState,
    current_num_rules: int,
    batch_size: int,
    train_suffix_len: int,
    accumulation_steps: int,
) -> Tuple[int, Dict[str, float]]:
    if adaptive_state.warmup_rule_usage is None:
        return current_num_rules, {"reason": "no_usage"}

    usage = adaptive_state.warmup_rule_usage.clone().to(torch.float64)
    total_tokens = int(usage.sum().item())
    active_rules = int((usage > 0).sum().item())
    if total_tokens <= 0 or active_rules <= 0:
        return current_num_rules, {"reason": "empty_usage"}

    tokens_per_opt_step = batch_size * train_suffix_len * accumulation_steps
    min_tokens_per_rule = max(24, int(math.sqrt(max(tokens_per_opt_step, 1))))
    data_cap_raw = max(2.0, float(total_tokens) / max(float(min_tokens_per_rule), 1.0))

    sorted_usage, _ = torch.sort(usage, descending=True)
    cumulative = torch.cumsum(sorted_usage, dim=0)
    coverage_threshold = total_tokens * 0.98
    coverage_count = int((cumulative < coverage_threshold).sum().item()) + 1

    avg_entropy = float(sum(adaptive_state.warmup_entropy) / max(len(adaptive_state.warmup_entropy), 1)) if adaptive_state.warmup_entropy else 0.0
    max_entropy = math.log(current_num_rules)
    entropy_ratio = avg_entropy / max(max_entropy, 1e-8) if max_entropy > 0 else 1.0

    if entropy_ratio > 0.90:
        active_multiplier = 1.25
        coverage_multiplier = 2.75
    elif entropy_ratio > 0.80:
        active_multiplier = 1.40
        coverage_multiplier = 3.00
    else:
        active_multiplier = 1.60
        coverage_multiplier = 3.25

    active_cap_raw = float(active_rules) * active_multiplier
    coverage_cap_raw = float(coverage_count) * coverage_multiplier
    entropy_effective = max(2.0, math.exp(avg_entropy))
    entropy_soft_cap = min(float(current_num_rules), max(float(active_rules), entropy_effective * 0.40))

    proposed_raw = min(
        float(current_num_rules),
        data_cap_raw,
        max(float(adaptive_state.min_target_rules), active_cap_raw, coverage_cap_raw),
        max(float(adaptive_state.min_target_rules), entropy_soft_cap),
    )
    proposed = smooth_floor_rule_count(
        proposed_raw,
        minimum=adaptive_state.min_target_rules,
        maximum=current_num_rules,
    )

    if proposed >= current_num_rules:
        proposed = current_num_rules

    stats = {
        "total_tokens": float(total_tokens),
        "active_rules": float(active_rules),
        "coverage_count": float(coverage_count),
        "avg_entropy": float(avg_entropy),
        "entropy_ratio": float(entropy_ratio),
        "data_cap_raw": float(data_cap_raw),
        "active_cap_raw": float(active_cap_raw),
        "coverage_cap_raw": float(coverage_cap_raw),
        "entropy_soft_cap": float(entropy_soft_cap),
        "proposed_raw": float(proposed_raw),
        "proposed": float(proposed),
    }
    return int(proposed), stats


def build_rule_remap(old_model: RuleTokenCausalModel, usage_counts: torch.Tensor, target_num_rules: int) -> torch.Tensor:
    old_num_rules = old_model.num_rules
    usage_counts = usage_counts.to(torch.float32)
    k = min(target_num_rules, old_num_rules)
    topk = torch.topk(usage_counts, k=k, largest=True, sorted=True).indices
    predictor_weight = old_model.rule_boundary_predictor.weight.detach().cpu().to(torch.float32)
    selected = predictor_weight[topk]
    selected = F.normalize(selected, p=2, dim=-1)
    all_norm = F.normalize(predictor_weight, p=2, dim=-1)
    sims = torch.matmul(all_norm, selected.transpose(0, 1))
    mapping = torch.argmax(sims, dim=-1)
    for new_idx, old_idx in enumerate(topk.tolist()):
        mapping[old_idx] = new_idx
    return mapping.to(torch.long)


def merge_rule_tensor(old_tensor: torch.Tensor, mapping: torch.Tensor, new_num_rules: int) -> torch.Tensor:
    out = torch.zeros((new_num_rules,) + tuple(old_tensor.shape[1:]), dtype=old_tensor.dtype)
    counts = torch.zeros(new_num_rules, dtype=torch.float32)
    for old_idx in range(old_tensor.size(0)):
        new_idx = int(mapping[old_idx].item())
        out[new_idx] += old_tensor[old_idx].detach().cpu()
        counts[new_idx] += 1.0
    nonzero = counts > 0
    if nonzero.any():
        out[nonzero] = out[nonzero] / counts[nonzero].view(-1, *([1] * (out.dim() - 1)))
    if (~nonzero).any():
        global_mean = old_tensor.detach().cpu().mean(dim=0)
        out[~nonzero] = global_mean
    return out


def merge_projection_rules(old_proj: RuleAwareProjection, new_proj: RuleAwareProjection, mapping: torch.Tensor):
    new_proj.shared_in.data.copy_(old_proj.shared_in.data)
    new_proj.shared_out.data.copy_(old_proj.shared_out.data)
    new_proj.rule_in.data.copy_(merge_rule_tensor(old_proj.rule_in.data, mapping, new_proj.rule_in.size(0)))
    new_proj.rule_out.data.copy_(merge_rule_tensor(old_proj.rule_out.data, mapping, new_proj.rule_out.size(0)))
    new_proj.rule_block_logits.data.copy_(merge_rule_tensor(old_proj.rule_block_logits.data, mapping, new_proj.rule_block_logits.size(0)))


def build_adapted_runtime(
    old_model: RuleTokenCausalModel,
    old_expert_store: RulePagedExpertStore,
    target_num_rules: int,
    usage_counts: torch.Tensor,
    vocab_size: int,
    device: torch.device,
) -> Tuple[RuleTokenCausalModel, RulePagedExpertStore, torch.Tensor]:
    mapping = build_rule_remap(old_model, usage_counts, target_num_rules)
    new_model = RuleTokenCausalModel(
        vocab_size=vocab_size,
        num_rules=target_num_rules,
        embed_size=old_model.embed_size,
        hidden_size=old_model.hidden_size,
        num_heads=old_model.num_heads,
        num_layers=old_model.num_layers,
        expert_dim=old_model.expert_dim,
    ).to(device)

    old_state = old_model.state_dict()
    new_state = new_model.state_dict()
    skip_fragments = [
        "rule_boundary_predictor",
        ".rule_in",
        ".rule_out",
        ".rule_block_logits",
        ".rule_router_",
        ".experts.",
        "vq_layer.basis",
        "vq_layer.levels_t",
    ]
    for key, value in new_state.items():
        if any(fragment in key for fragment in skip_fragments):
            continue
        if key in old_state and old_state[key].shape == value.shape:
            value.copy_(old_state[key])
    new_model.load_state_dict(new_state, strict=False)

    if new_model.vq_layer.project_in.weight.shape == old_model.vq_layer.project_in.weight.shape:
        new_model.vq_layer.project_in.load_state_dict(old_model.vq_layer.project_in.state_dict())
        new_model.vq_layer.project_out.load_state_dict(old_model.vq_layer.project_out.state_dict())

    new_model.rule_boundary_predictor.weight.data.copy_(
        merge_rule_tensor(old_model.rule_boundary_predictor.weight.data, mapping, target_num_rules)
    )
    new_model.rule_boundary_predictor.bias.data.copy_(
        merge_rule_tensor(old_model.rule_boundary_predictor.bias.data.unsqueeze(-1), mapping, target_num_rules).squeeze(-1)
    )

    for old_block, new_block in zip(old_model.layers, new_model.layers):
        merge_projection_rules(old_block.attn.q_proj, new_block.attn.q_proj, mapping)
        merge_projection_rules(old_block.attn.k_proj, new_block.attn.k_proj, mapping)
        merge_projection_rules(old_block.attn.v_proj, new_block.attn.v_proj, mapping)
        merge_projection_rules(old_block.attn.out_proj, new_block.attn.out_proj, mapping)
        
        new_block.attn.rule_router_q.data.copy_(merge_rule_tensor(old_block.attn.rule_router_q.data, mapping, target_num_rules))
        new_block.attn.rule_router_k.data.copy_(merge_rule_tensor(old_block.attn.rule_router_k.data, mapping, target_num_rules))

    new_expert_store = RulePagedExpertStore.from_model(new_model)
    for layer_idx in range(old_model.num_layers):
        old_layer = old_expert_store.layers[layer_idx]
        new_layer = new_expert_store.layers[layer_idx]
        new_layer.w1.data.copy_(merge_rule_tensor(old_layer.w1.data, mapping, target_num_rules))
        new_layer.b1.data.copy_(merge_rule_tensor(old_layer.b1.data, mapping, target_num_rules))
        new_layer.w2.data.copy_(merge_rule_tensor(old_layer.w2.data, mapping, target_num_rules))
        new_layer.b2.data.copy_(merge_rule_tensor(old_layer.b2.data, mapping, target_num_rules))

    new_model.offload_experts_to_runtime()
    return new_model, new_expert_store, mapping


def rebuild_optimizers(
    model: RuleTokenCausalModel,
    cpu_token_embedding: nn.Embedding,
    expert_store: RulePagedExpertStore,
    token_lr_scale: float,
) -> Tuple[optim.Optimizer, optim.Optimizer, optim.Optimizer]:
    base_lr = 1.0 / math.sqrt(model.embed_size * model.hidden_size)
    optimizer_rule = optim.Adam(model.parameters(), lr=base_lr)
    token_lr = base_lr * float(token_lr_scale)
    optimizer_token = optim.SparseAdam(cpu_token_embedding.parameters(), lr=token_lr) if cpu_token_embedding.sparse else optim.Adam(cpu_token_embedding.parameters(), lr=token_lr)
    optimizer_expert = optim.Adam(expert_store.parameters(), lr=base_lr)
    return optimizer_rule, optimizer_token, optimizer_expert


def clip_grad_norm_sparse_aware(parameters, max_norm: float) -> float:
    params = list(parameters)
    total_norm_sq = 0.0
    for p in params:
        if p.grad is not None:
            if p.grad.is_sparse:
                total_norm_sq += p.grad.coalesce()._values().norm(2).item() ** 2
            else:
                total_norm_sq += p.grad.data.norm(2).item() ** 2
    total_norm = math.sqrt(total_norm_sq)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            if p.grad is not None:
                if p.grad.is_sparse:
                    p.grad.data._values().mul_(clip_coef)
                else:
                    p.grad.data.mul_(clip_coef)
    return total_norm


def compute_parameter_trust_region(parameters) -> float:
    params = [p for p in parameters if p.requires_grad]
    total_norm_sq = 0.0
    total_dof = 0
    for p in params:
        total_norm_sq += p.data.float().norm(2).item() ** 2
        total_dof += p.numel()
    if total_dof <= 0:
        return 1.0
    parameter_norm = math.sqrt(total_norm_sq)
    parameter_rms = parameter_norm / math.sqrt(total_dof)
    manifold_scale = math.sqrt(max(math.log(total_dof + math.e), 1.0))
    return max(parameter_rms * manifold_scale, 1e-6)


def cosine_lr_multiplier(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max(0.01, step / max(warmup_steps, 1))
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max(0.01, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))


def apply_lr_schedule(optimizers: list, base_lr: float, multiplier: float):
    lr = base_lr * multiplier
    for opt in optimizers:
        for pg in opt.param_groups:
            pg["lr"] = lr


def maybe_adapt_num_rules(
    adaptive_state: AdaptiveRuleState,
    update_step: int,
    model: RuleTokenCausalModel,
    expert_store: RulePagedExpertStore,
    cpu_token_embedding: nn.Embedding,
    vocab_size: int,
    device: torch.device,
    batch_size: int,
    train_suffix_len: int,
    accumulation_steps: int,
) -> Tuple[RuleTokenCausalModel, RulePagedExpertStore, Dict[str, optim.Optimizer], bool, Optional[Dict[str, float]]]:
    if not adaptive_state.enabled or adaptive_state.adapted or update_step < adaptive_state.warmup_optimizer_steps:
        return model, expert_store, {}, False, None

    target_num_rules, stats = decide_adaptive_num_rules(
        adaptive_state,
        model.num_rules,
        batch_size=batch_size,
        train_suffix_len=train_suffix_len,
        accumulation_steps=accumulation_steps,
    )
    adaptive_state.target_num_rules = target_num_rules
    adaptive_state.adapted = True

    if target_num_rules >= model.num_rules:
        return model, expert_store, {}, False, stats

    logger.info(
        "[AdaptiveRules] Warmup 完成：num_rules %d -> %d | "
        "总 token=%d | 活跃规则=%d | 覆盖规则=%d | 平均微观熵=%.4f | 熵占比=%.4f",
        model.num_rules, target_num_rules,
        int(stats["total_tokens"]), int(stats["active_rules"]),
        int(stats["coverage_count"]), stats["avg_entropy"], stats["entropy_ratio"],
    )

    new_model, new_expert_store, _ = build_adapted_runtime(
        old_model=model,
        old_expert_store=expert_store,
        target_num_rules=target_num_rules,
        usage_counts=adaptive_state.warmup_rule_usage,
        vocab_size=vocab_size,
        device=device,
    )
    adaptive_state.post_adapt_rule_only_until_update = update_step + adaptive_state.post_adapt_rule_only_steps
    optimizer_rule, optimizer_token, optimizer_expert = rebuild_optimizers(
        new_model,
        cpu_token_embedding,
        new_expert_store,
        token_lr_scale=adaptive_state.token_lr_scale,
    )
    return new_model, new_expert_store, {
        "rule": optimizer_rule,
        "token": optimizer_token,
        "expert": optimizer_expert,
    }, True, stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Large-scale rule/token trainer")
    parser.add_argument("--real-epochs", type=int, default=None, help="Limit REAL stage epochs (e.g. 3). Omit to use auto schedule.")
    parser.add_argument("--sft-epochs", type=int, default=None, help="Limit SFT stage epochs (e.g. 3). Omit to use auto schedule.")
    return parser.parse_args()


def train_large_model(*, real_epochs: Optional[int] = None, sft_epochs: Optional[int] = None):
    logger.info("加载词表...")
    vocab = load_vocab_info()
    protocol_tokens = resolve_protocol_tokens(vocab)
    pad_id = protocol_tokens["pad"]["id"]
    eos_id = protocol_tokens["eos"]["id"]
    assistant_role = protocol_tokens["roles"].get("assistant")
    assistant_role_id = assistant_role["id"] if assistant_role is not None else None
    role_token_ids = {payload["id"] for payload in protocol_tokens["role_slots"]}

    training_stages = discover_training_stages()
    if any(stage.assistant_only_loss for stage in training_stages):
        if assistant_role_id is None:
            raise KeyError("当前词表中未发现 assistant 角色协议符号，无法启用 assistant-only SFT loss。")
        if not role_token_ids:
            raise KeyError("当前词表中未发现角色协议符号，无法构建 assistant-only SFT 掩码。")

    seed = derive_seed(vocab["vocab_size"], len(training_stages), *[s.name for s in training_stages])
    set_deterministic_seed(seed)
    logger.info("确定性种子: %d", seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = discover_model_path()
    checkpoint_payload = load_checkpoint_payload(checkpoint_path)

    estimated_seq_len = max(stage.estimated_seq_len for stage in training_stages)
    bootstrap_config = infer_model_config(vocab["vocab_size"], checkpoint_path)
    
    # 填充由第一性原理推导的配置，避免 KeyError
    if not bootstrap_config:
        base_entropy = math.log2(max(vocab["vocab_size"], 2))
        embed_size = next_power_of_two(int(base_entropy * base_entropy))
        num_rules = next_power_of_two(int(math.sqrt(embed_size) * math.log(max(vocab["vocab_size"], 2))))
        hidden_size = next_power_of_two(int(embed_size * math.e))
        num_layers = max(4, int(math.ceil(base_entropy / math.e)))
        from tokenizer_utils import infer_num_heads
        num_heads = infer_num_heads(embed_size)
        bootstrap_config = {
            "num_rules": num_rules,
            "embed_size": embed_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "expert_dim": None,
        }

    bootstrap_model = RuleTokenCausalModel(vocab_size=vocab["vocab_size"], **bootstrap_config).to(device)
    bootstrap_batch_size, bootstrap_accumulation_steps = infer_training_schedule(bootstrap_model, estimated_seq_len, vocab["vocab_size"], device)
    train_context_len, train_suffix_len = infer_training_windows(estimated_seq_len, device)

    adaptive_state, initial_num_rules = infer_adaptive_state(
        vocab_size=vocab["vocab_size"],
        batch_size=bootstrap_batch_size,
        accumulation_steps=bootstrap_accumulation_steps,
        train_suffix_len=train_suffix_len,
        configured_num_rules=bootstrap_config["num_rules"],
        checkpoint_payload=checkpoint_payload if isinstance(checkpoint_payload, dict) else {},
    )
    
    bootstrap_config["num_rules"] = initial_num_rules
    del bootstrap_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model_config = bootstrap_config
    model = RuleTokenCausalModel(vocab_size=vocab["vocab_size"], **model_config).to(device)
    train_model = model

    cpu_token_embedding = nn.Embedding(vocab["vocab_size"], model.embed_size, sparse=True).to("cpu")

    expert_store = RulePagedExpertStore.from_model(model)
    model.offload_experts_to_runtime()

    payload = load_checkpoint_into_runtime(checkpoint_path, model, cpu_token_embedding, expert_store)
    train_model = maybe_enable_torch_compile(model, device, "初始模型")

    batch_size, accumulation_steps = infer_training_schedule(model, estimated_seq_len, vocab["vocab_size"], device)
    if adaptive_state.enabled and not adaptive_state.adapted:
        adaptive_state, _ = infer_adaptive_state(
            vocab_size=vocab["vocab_size"],
            batch_size=batch_size,
            accumulation_steps=accumulation_steps,
            train_suffix_len=train_suffix_len,
            configured_num_rules=model.num_rules,
            checkpoint_payload=payload if isinstance(payload, dict) else {},
        )
        adaptive_state.ensure_buffers(model.num_rules)
    base_lr = 1.0 / math.sqrt(model.embed_size * model.hidden_size)

    optimizer_rule, optimizer_token, optimizer_expert = rebuild_optimizers(model, cpu_token_embedding, expert_store, token_lr_scale=adaptive_state.token_lr_scale)

    saved_opt_state = payload.get("optimizer_state") if isinstance(payload, dict) else None
    if saved_opt_state is not None:
        try:
            if "rule" in saved_opt_state:
                optimizer_rule.load_state_dict(saved_opt_state["rule"])
            if "token" in saved_opt_state:
                optimizer_token.load_state_dict(saved_opt_state["token"])
            if "expert" in saved_opt_state:
                optimizer_expert.load_state_dict(saved_opt_state["expert"])
            logger.info("已恢复优化器状态。")
        except Exception as e:
            logger.warning("优化器状态恢复失败，使用新初始化: %s", e)

    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    autocast_enabled = device.type == "cuda"

    grad_tracker = GradientTracker()

    max_epochs_total = sum(max(1, int(math.ceil(math.log2(len(s.chunk_files) + 1)))) for s in training_stages)
    warmup_steps = max(1, int(math.log(vocab["vocab_size"])))

    train_model.train()
    optimizer_rule.zero_grad(set_to_none=True)
    optimizer_token.zero_grad(set_to_none=True)
    expert_store.zero_grad(set_to_none=True)

    eps = torch.finfo(torch.float32).eps
    prev_loss_boundary = math.log(model.num_rules)

    logger.info("词表大小: %d", vocab["vocab_size"])
    logger.info("模型配置: %s", model_config)
    logger.info("自适应批大小: %d | 梯度累积: %d | 训练窗口: context=%d, suffix=%d", batch_size, accumulation_steps, train_context_len, train_suffix_len)

    with torch.no_grad():
        all_token_embeds = cpu_token_embedding.weight.data
    cell_state = model.compute_rule_vocab_mask(token_chunk_embeds=all_token_embeds)

    global_update_step = 0
    max_possible_macro_entropy = math.log(model.num_rules)

    best_loss = float('inf')
    best_state_dict = None

    for stage_index, training_stage in enumerate(training_stages, start=1):
        logger.info("=" * 80)
        logger.info(
            "进入阶段 %d/%d: %s | 分块数=%d | 目标=%s",
            stage_index, len(training_stages), training_stage.name.upper(),
            len(training_stage.chunk_files),
            "assistant-only SFT" if training_stage.assistant_only_loss else "causal LM",
        )

        dataloader = DataLoader(
            StreamingDataset(training_stage.chunk_files),
            batch_size=batch_size,
            collate_fn=build_collate_fn(
                pad_token_id=pad_id,
                assistant_only_loss=training_stage.assistant_only_loss,
                assistant_role_id=assistant_role_id,
                role_token_ids=role_token_ids,
                eos_id=eos_id,
            ),
        )
        train_model.train()
        optimizer_rule.zero_grad(set_to_none=True)
        optimizer_token.zero_grad(set_to_none=True)
        expert_store.zero_grad(set_to_none=True)

        max_epochs = max(1, int(math.ceil(math.log2(len(training_stage.chunk_files) + 1))))
        stage_name = (training_stage.name or "").strip().lower()
        override_epochs: Optional[int] = None
        if stage_name == "real" and real_epochs is not None:
            override_epochs = real_epochs
        elif stage_name == "sft" and sft_epochs is not None:
            override_epochs = sft_epochs
        if override_epochs is not None:
            if override_epochs <= 0:
                raise ValueError(f"--{stage_name}-epochs must be a positive integer, got {override_epochs}")
            if override_epochs != max_epochs:
                logger.info("阶段 %s 使用配置轮数上限: %d (原计划: %d)", training_stage.name.upper(), override_epochs, max_epochs)
            max_epochs = override_epochs
        steady_state_epochs = 0
        previous_epoch_loss = None

        for epoch in range(max_epochs):
            total_loss = 0.0
            epoch_macro_entropies: List[float] = []
            epoch_micro_entropies: List[float] = []
            batch_counter = 0
            runtime_expert_pages = None

            for batch_idx, batch in enumerate(dataloader):
                step_loss = None
                micro_step = (batch_idx % accumulation_steps) + 1
                if device.type == "cuda" and micro_step == 1:
                    torch.cuda.reset_peak_memory_stats(device)
                    accumulation_window_start = time.perf_counter()

                ctx_ids_cpu = batch["context_tokens"].long()
                target_ids_cpu = batch["tokens"].long()
                loss_mask_cpu = batch["loss_mask"].bool()
                ctx_ids_cpu, target_ids_cpu, loss_mask_cpu, focus_mask_cpu = prepare_training_window(
                    ctx_ids_cpu, target_ids_cpu, loss_mask_cpu,
                    train_context_len, train_suffix_len,
                )
                B, Seq = target_ids_cpu.size()

                cell_matrix, cell_counts, token_to_cell, token_pos = cell_state
                target_rule_ids = token_to_cell[target_ids_cpu.clamp_min(0).to(device)]

                max_rule_entropy = math.log(model.num_rules)
                geometric_capacity = math.sqrt(model.embed_size)
                if batch_counter == 0:
                    micro_entropy_ratio = 1.0
                else:
                    _mer = mean_s_inf.item() / max(max_rule_entropy, eps)
                    micro_entropy_ratio = _mer if math.isfinite(_mer) else 1.0
                num_local_negatives = max(1, int(round(math.exp(micro_entropy_ratio * math.log(geometric_capacity)))))
                num_global_negatives = max(2, int(round(micro_entropy_ratio * 8)))

                flat_target_rule_ids = target_rule_ids.reshape(-1)
                loss_sort_ctx = compute_rule_sort_context(flat_target_rule_ids)
                unique_target_rules = loss_sort_ctx.unique_rules
                inverse_indices = loss_sort_ctx.inverse
                num_unique_rules = unique_target_rules.size(0)
                counts = cell_counts[unique_target_rules].clamp_min(1)
                rand_vals = torch.rand(num_unique_rules, num_local_negatives, device=device)
                rand_indices = (rand_vals * counts.unsqueeze(1)).long()
                sampled_local_negatives = cell_matrix[unique_target_rules.unsqueeze(1), rand_indices]
                shared_global_negatives = torch.randint(0, vocab["vocab_size"], (num_global_negatives,), device=device)

                active_token_ids_cpu = torch.cat([
                    ctx_ids_cpu.reshape(-1),
                    target_ids_cpu.reshape(-1),
                    sampled_local_negatives.reshape(-1).cpu(),
                    shared_global_negatives.cpu(),
                ], dim=0)

                boundary_nats = prev_loss_boundary if batch_counter > 0 else max_rule_entropy
                token_lr_signal = min(1.0, boundary_nats / max(max_rule_entropy, 1.0))
                dynamic_token_lr = base_lr * adaptive_state.token_lr_scale * token_lr_signal * grad_tracker.lr_damping

                for param_group in optimizer_token.param_groups:
                    param_group['lr'] = dynamic_token_lr

                active_rules = torch.unique(target_rule_ids.detach().cpu(), sorted=True).tolist()
                runtime_expert_pages = expert_store.build_runtime(active_rules, device=device, training=True)

                active_ids_cpu, active_ids_gpu, active_embeds_cpu, active_embeds_gpu = build_sorted_active_embeddings(
                    cpu_token_embedding, active_token_ids_cpu, device, train_token=True,
                )

                ctx_embeds = lookup_local_embeddings(ctx_ids_cpu.to(device), active_ids_gpu, active_embeds_gpu)
                target_embeds = lookup_local_embeddings(target_ids_cpu.to(device), active_ids_gpu, active_embeds_gpu)
                local_neg_embeds = lookup_local_embeddings(sampled_local_negatives, active_ids_gpu, active_embeds_gpu)
                global_neg_embeds = lookup_local_embeddings(shared_global_negatives.unsqueeze(0), active_ids_gpu, active_embeds_gpu).squeeze(0)

                train_model.requires_grad_(True)

                with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                    rule_logits, truth_field_state, memory, _ = train_model(
                        ctx_embeds,
                        target_rule_ids=target_rule_ids,
                        runtime_expert_pages=runtime_expert_pages,
                    )

                    valid_mask = (
                        (target_ids_cpu.to(device) != pad_id)
                        & focus_mask_cpu.to(device)
                        & loss_mask_cpu.to(device)
                    ).float()
                    valid_sum = valid_mask.sum().clamp_min(eps)

                    with torch.amp.autocast(device_type=device.type, enabled=False):
                        rule_logits_fp32 = rule_logits.float()
                        truth_field_state_fp32 = truth_field_state.float()
                        target_features = rms_normalize(target_embeds.float()) * math.sqrt(model.embed_size)
                        with torch.no_grad():
                            z_target = model.vq_layer.project_in(target_features.detach())
                            bound_target = torch.tanh(z_target)
                            scaled_target = (bound_target + 1.0) / 2.0 * (model.vq_layer.levels_t.float() - 1.0)
                            quantized_target = torch.round(scaled_target)
                            target_rule_ids_vq = torch.sum(quantized_target.long() * model.vq_layer.basis, dim=-1)

                            rule_idx_tensor = torch.arange(model.num_rules, device=device)
                            coords = []
                            rem = rule_idx_tensor
                            for b, l in zip(model.vq_layer.basis.tolist(), model.vq_layer.levels_t.tolist()):
                                coords.append((rem % int(l)).float())
                                rem = rem // int(l)
                            all_rule_coords = torch.stack(coords, dim=-1)

                        expected_coords = torch.matmul(F.softmax(rule_logits_fp32, dim=-1), all_rule_coords)
                        homo_grav_loss = F.mse_loss(
                            expected_coords[:, 1:],
                            expected_coords[:, :-1].detach()
                        )

                        flat_rule_logits = rule_logits_fp32.reshape(-1, model.num_rules)
                        flat_targets = target_rule_ids_vq.reshape(-1)
                        target_logits = flat_rule_logits.gather(1, flat_targets.unsqueeze(1)).squeeze(1)
                        loss_rule_boundary_flat = torch.logsumexp(flat_rule_logits, dim=-1) - target_logits

                        soft_rule_probs = F.softmax(rule_logits_fp32, dim=-1)
                        p_r_per_sample = soft_rule_probs.mean(dim=1)
                        s_inf_per_sample = -torch.sum(p_r_per_sample * torch.log(p_r_per_sample.clamp_min(eps)), dim=-1)
                        mean_s_inf = s_inf_per_sample.mean()

                        local_neg_features = rms_normalize(local_neg_embeds.float()) * math.sqrt(model.embed_size)
                        global_neg_features = rms_normalize(global_neg_embeds.float()) * math.sqrt(model.embed_size)

                        with torch.no_grad():
                            z = model.vq_layer.project_in(local_neg_features.reshape(-1, model.embed_size))
                            bound = torch.tanh(z)
                            scaled = (bound + 1.0) / 2.0 * (model.vq_layer.levels_t.float() - 1.0)
                            quantized = torch.round(scaled)
                            neg_primary_rules = torch.sum(quantized.long() * model.vq_layer.basis, dim=-1).reshape(num_unique_rules, num_local_negatives)
                            residual = torch.abs(scaled - quantized).max(dim=-1)[0].reshape(num_unique_rules, num_local_negatives)
                            split_threshold = residual.mean() + residual.std(unbiased=False)
                            is_ambiguous = residual > split_threshold
                            rule_valid_mask = (neg_primary_rules == unique_target_rules.unsqueeze(1)) | is_ambiguous

                        token_local_valid_mask = rule_valid_mask[inverse_indices]

                        field_norm = F.normalize(truth_field_state_fp32, p=2, dim=-1).reshape(-1, model.embed_size)
                        target_norm = F.normalize(target_features, p=2, dim=-1).reshape(-1, model.embed_size)
                        local_neg_norm = F.normalize(local_neg_features, p=2, dim=-1)
                        global_neg_norm = F.normalize(global_neg_features, p=2, dim=-1)
                    del target_features, local_neg_features, global_neg_features

                    pos_sim = torch.sum(field_norm * target_norm, dim=-1)
                    del target_norm

                    N_flat = B * Seq
                    sorted_field = field_norm[loss_sort_ctx.sort_idx]
                    padded_field = field_norm.new_zeros(num_unique_rules, loss_sort_ctx.max_g, model.embed_size)
                    padded_field[loss_sort_ctx.sorted_groups, loss_sort_ctx.pos_in_group] = sorted_field
                    local_neg_sim_padded = torch.bmm(
                        padded_field,
                        local_neg_norm.transpose(1, 2).to(padded_field.dtype),
                    )
                    sorted_local_sim = local_neg_sim_padded[loss_sort_ctx.sorted_groups, loss_sort_ctx.pos_in_group]
                    local_neg_sim = torch.empty_like(sorted_local_sim)
                    local_neg_sim[loss_sort_ctx.sort_idx] = sorted_local_sim
                    del padded_field, local_neg_sim_padded, sorted_local_sim, sorted_field

                    global_neg_sim = torch.matmul(field_norm, global_neg_norm.t())

                    tau = 1.0 / math.sqrt(model.embed_size)
                    masked_local = torch.where(
                        token_local_valid_mask,
                        local_neg_sim / tau,
                        torch.full_like(local_neg_sim, -1e4),
                    )
                    contrastive_logits = torch.cat([
                        (pos_sim / tau).unsqueeze(-1),
                        masked_local,
                        global_neg_sim / tau,
                    ], dim=-1)
                    loss_local_readout_flat = F.cross_entropy(
                        contrastive_logits,
                        torch.zeros(N_flat, dtype=torch.long, device=device),
                        reduction="none",
                    )
                    del local_neg_sim, global_neg_sim, token_local_valid_mask
                    del contrastive_logits, masked_local

                    flat_valid = valid_mask.reshape(-1)

                    loss_boundary = (loss_rule_boundary_flat * flat_valid).sum() / valid_sum
                    loss_boundary += homo_grav_loss * 0.1  # 引入 0.1 权重的同态引力场正则化
                    loss_readout = (loss_local_readout_flat * flat_valid).sum() / valid_sum
                    _lb = loss_boundary.detach().item()
                    if math.isfinite(_lb):
                        prev_loss_boundary = _lb

                    batch_rule_distribution = soft_rule_probs.reshape(-1, model.num_rules).mean(dim=0)
                    macro_entropy = -torch.sum(batch_rule_distribution * torch.log(batch_rule_distribution.clamp_min(eps)))
                    kl_divergence = math.log(model.num_rules) - macro_entropy
                    beta = (loss_boundary.detach().item() + loss_readout.detach().item()) / max(math.log(model.num_rules), 1.0)

                    # 纯高维逻辑空间的“降维打击”训练（解耦双阶段训练）
                    # 交替进行高维 Rule 逻辑推演和低维 Token 发射，大幅降低显存和复杂度
                    # 使用基于黄金分割率 (Golden Ratio) 的自适应相位波函数
                    phi = (1.0 + math.sqrt(5.0)) / 2.0
                    phase_interval = max(4, int(math.log(model.num_rules) * phi))
                    current_phase = "rule_logic" if (global_update_step // phase_interval) % 2 == 0 else "token_emission"

                    if current_phase == "rule_logic":
                        # 锁定高维到低维投影，只训练高维打腹稿
                        prediction_loss = loss_boundary
                        loss = (prediction_loss + beta * kl_divergence) / accumulation_steps
                    else:
                        # 锁定高维逻辑链，只训练具体 Token 翻译
                        prediction_loss = loss_readout
                        loss = prediction_loss / accumulation_steps

                if (batch_idx + 1) % accumulation_steps == 0:
                    chunk_size = max(1, int(math.sqrt(vocab["vocab_size"])))
                    global_step = batch_counter
                    start_idx = (global_step * chunk_size) % vocab["vocab_size"]
                    end_idx = start_idx + chunk_size
                    if end_idx <= vocab["vocab_size"]:
                        token_chunk_ids = torch.arange(start_idx, end_idx, device="cpu")
                    else:
                        token_chunk_ids = torch.cat([
                            torch.arange(start_idx, vocab["vocab_size"], device="cpu"),
                            torch.arange(0, end_idx % vocab["vocab_size"], device="cpu"),
                        ])
                    with torch.no_grad():
                        token_chunk_embeds = cpu_token_embedding(token_chunk_ids).to(device)
                    cell_state = model.compute_rule_vocab_mask(
                        token_chunk_embeds=token_chunk_embeds,
                        token_chunk_ids=token_chunk_ids.to(device),
                        cell_state=cell_state,
                    )

                scaler.scale(loss).backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    global_update_step += 1

                    expert_store.scatter_runtime_grads(runtime_expert_pages)

                    scaler.unscale_(optimizer_rule)
                    scaler.unscale_(optimizer_token)
                    scaler.unscale_(optimizer_expert)

                    _dense_norm_sq = 0.0
                    _sparse_norm_sq = 0.0
                    for opt in [optimizer_rule, optimizer_token, optimizer_expert]:
                        for pg in opt.param_groups:
                            for p in pg["params"]:
                                if p.grad is not None:
                                    if p.grad.is_sparse:
                                        _sparse_norm_sq += p.grad.coalesce()._values().norm(2).item() ** 2
                                    else:
                                        _dense_norm_sq += p.grad.data.norm(2).item() ** 2
                    total_norm_sq = _dense_norm_sq + _sparse_norm_sq
                    current_grad_norm = math.sqrt(total_norm_sq)
                    grad_finite = math.isfinite(current_grad_norm)

                    if grad_finite:
                        grad_tracker.update(current_grad_norm)
                        clip_norm_rule = min(grad_tracker.adaptive_max_norm, compute_parameter_trust_region(model.parameters()))
                        clip_norm_token = min(grad_tracker.adaptive_max_norm, compute_parameter_trust_region(cpu_token_embedding.parameters()))
                        clip_norm_expert = min(grad_tracker.adaptive_max_norm, compute_parameter_trust_region(expert_store.parameters()))
                        clip_grad_norm_sparse_aware(model.parameters(), max_norm=clip_norm_rule)
                        clip_grad_norm_sparse_aware(cpu_token_embedding.parameters(), max_norm=clip_norm_token)
                        clip_grad_norm_sparse_aware(expert_store.parameters(), max_norm=clip_norm_expert)
                        clip_norm = min(clip_norm_rule, clip_norm_token, clip_norm_expert)
                    else:
                        clip_norm = grad_tracker.adaptive_max_norm
                        logger.warning("  [gradient guard] non-finite grad at step %d, skipping parameter update", global_update_step)

                    lr_mult = cosine_lr_multiplier(global_update_step, warmup_steps, warmup_steps + max_epochs_total * max(batch_counter, 1))
                    damping = grad_tracker.lr_damping
                    effective_lr = lr_mult * damping
                    apply_lr_schedule([optimizer_rule, optimizer_expert], base_lr, effective_lr)

                    if grad_finite:
                        scaler.step(optimizer_rule)
                        scaler.step(optimizer_token)
                        scaler.step(optimizer_expert)
                    scaler.update()

                    optimizer_rule.zero_grad(set_to_none=True)
                    optimizer_token.zero_grad(set_to_none=True)
                    expert_store.zero_grad(set_to_none=True)

                    if adaptive_state.enabled and not adaptive_state.adapted:
                        adaptive_state.ensure_buffers(model.num_rules)
                        usage = torch.bincount(target_rule_ids.reshape(-1).detach().cpu(), minlength=model.num_rules)
                        adaptive_state.warmup_rule_usage[: usage.numel()] += usage
                        adaptive_state.warmup_entropy.append(float(mean_s_inf.item()))
                        adaptive_state.total_warmup_tokens += int(valid_sum.item())

                    update_step = global_update_step
                    step_loss = loss.item() * accumulation_steps
                    window_elapsed = time.perf_counter() - accumulation_window_start if device.type == "cuda" else 0.0
                    logger.info(
                        "  [%s E%d] step=%d | loss=%.4f | micro_H=%.4f | grad_norm=%.2f | clip=%.2f | SNR=%.1f | lr=%.4f(D=%.2f) | neg=%d+%d | %.1fs",
                        training_stage.name.upper(), epoch + 1, update_step,
                        step_loss, mean_s_inf.item(),
                        current_grad_norm, clip_norm, grad_tracker.signal_to_noise,
                        effective_lr, damping, num_local_negatives, num_global_negatives, window_elapsed,
                    )

                    snr = grad_tracker.signal_to_noise
                    if snr < 1.0 and global_update_step > warmup_steps * 2:
                        logger.warning("  [泛化监控] SNR=%.2f log_var=%.3f damping=%.3f", snr, grad_tracker.log_var, damping)

                    maybe_result = maybe_adapt_num_rules(
                        adaptive_state=adaptive_state,
                        update_step=update_step,
                        model=model,
                        expert_store=expert_store,
                        cpu_token_embedding=cpu_token_embedding,
                        vocab_size=vocab["vocab_size"],
                        device=device,
                        batch_size=batch_size,
                        train_suffix_len=train_suffix_len,
                        accumulation_steps=accumulation_steps,
                    )
                    model, expert_store, new_optimizers, adapted_now, _ = maybe_result
                    if adapted_now:
                        train_model = maybe_enable_torch_compile(model, device, f"num_rules={model.num_rules} 重建模型")
                        model.train()
                        optimizer_rule = new_optimizers["rule"]
                        optimizer_token = new_optimizers["token"]
                        optimizer_expert = new_optimizers["expert"]
                        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
                        max_possible_macro_entropy = math.log(model.num_rules)
                        adaptive_state.ensure_buffers(model.num_rules)
                        grad_tracker = GradientTracker()
                        prev_loss_boundary = math.log(model.num_rules)
                        with torch.no_grad():
                            all_token_embeds = cpu_token_embedding.weight.data
                        cell_state = model.compute_rule_vocab_mask(token_chunk_embeds=all_token_embeds)

                del rule_logits, truth_field_state, memory
                del pos_sim, field_norm, local_neg_norm, global_neg_norm
                del ctx_embeds, target_embeds, local_neg_embeds, global_neg_embeds, active_embeds_gpu, active_embeds_cpu, active_ids_cpu, active_ids_gpu, loss
                if (batch_idx + 1) % accumulation_steps == 0 and device.type == "cuda":
                    torch.cuda.empty_cache()

                if step_loss is not None:
                    total_loss += step_loss
                epoch_macro_entropies.append(macro_entropy.item())
                epoch_micro_entropies.append(mean_s_inf.item())
                batch_counter += 1

            if batch_counter % accumulation_steps != 0 and runtime_expert_pages is not None:
                expert_store.scatter_runtime_grads(runtime_expert_pages)
                scaler.unscale_(optimizer_rule)
                scaler.unscale_(optimizer_token)
                scaler.unscale_(optimizer_expert)
                
                tail_clip_rule = min(grad_tracker.adaptive_max_norm if grad_tracker.initialized else 1.0, compute_parameter_trust_region(model.parameters()))
                tail_clip_token = min(grad_tracker.adaptive_max_norm if grad_tracker.initialized else 1.0, compute_parameter_trust_region(cpu_token_embedding.parameters()))
                tail_clip_expert = min(grad_tracker.adaptive_max_norm if grad_tracker.initialized else 1.0, compute_parameter_trust_region(expert_store.parameters()))
                clip_grad_norm_sparse_aware(model.parameters(), max_norm=tail_clip_rule)
                clip_grad_norm_sparse_aware(cpu_token_embedding.parameters(), max_norm=tail_clip_token)
                clip_grad_norm_sparse_aware(expert_store.parameters(), max_norm=tail_clip_expert)

                scaler.step(optimizer_rule)
                scaler.step(optimizer_token)
                scaler.step(optimizer_expert)
                scaler.update()

                optimizer_rule.zero_grad(set_to_none=True)
                optimizer_token.zero_grad(set_to_none=True)
                expert_store.zero_grad(set_to_none=True)

            avg_loss = total_loss / max(batch_counter, 1)
            avg_macro_entropy = sum(epoch_macro_entropies) / max(len(epoch_macro_entropies), 1)
            avg_micro_entropy = sum(epoch_micro_entropies) / max(len(epoch_micro_entropies), 1)
            logger.info(
                "%s E%02d | avg_loss=%.4f | macro_H=%.4f | micro_H=%.4f | grad_SNR=%.1f",
                training_stage.name.upper(), epoch + 1, avg_loss, avg_macro_entropy, avg_micro_entropy,
                grad_tracker.signal_to_noise,
            )

            if previous_epoch_loss is not None:
                loss_improvement = (previous_epoch_loss - avg_loss) / max(abs(previous_epoch_loss), eps)
                stability_threshold = 1.0 / math.sqrt(model.num_rules * max_epochs)
                entropy_floor = max_possible_macro_entropy / max(math.log(model.num_rules), 1.0)
                snr_stable = grad_tracker.signal_to_noise < 2.0 and grad_tracker.initialized
                if (loss_improvement < stability_threshold and avg_macro_entropy < entropy_floor) or snr_stable:
                    steady_state_epochs += 1
                else:
                    steady_state_epochs = 0
                if steady_state_epochs >= max(1, int(math.ceil(math.log(model.num_layers + 1)))):
                    logger.info("阶段 %s 已进入稳态演化区，训练在第 %d 轮停止。", training_stage.name.upper(), epoch + 1)
                    break
            previous_epoch_loss = avg_loss
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state_dict = {
                    "model_state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    "expert_store": {k: v.cpu().clone() for k, v in expert_store.state_dict().items()},
                    "cpu_token_embedding.weight": cpu_token_embedding.weight.data.cpu().clone(),
                }
                logger.info("发现更好模型，已缓存最佳状态 (loss: %.4f)", best_loss)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict["model_state"])
        expert_store.load_state_dict(best_state_dict["expert_store"])
        cpu_token_embedding.weight.data.copy_(best_state_dict["cpu_token_embedding.weight"])
        logger.info("已加载整个训练过程中的最佳模型状态 (best_loss: %.4f)", best_loss)

    state_dict_to_save = {
        "format": "hrsp_adaptive_v33",
        "config": {
            "num_rules": model.num_rules,
            "embed_size": model.embed_size,
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
            "num_heads": model.num_heads,
            "expert_dim": model.expert_dim,
        },
        "adaptive_rule_state": {
            "enabled": adaptive_state.enabled,
            "initial_num_rules": adaptive_state.initial_num_rules,
            "warmup_optimizer_steps": adaptive_state.warmup_optimizer_steps,
            "min_target_rules": adaptive_state.min_target_rules,
            "post_adapt_rule_only_steps": adaptive_state.post_adapt_rule_only_steps,
            "post_adapt_rule_only_until_update": adaptive_state.post_adapt_rule_only_until_update,
            "token_lr_scale": adaptive_state.token_lr_scale,
            "token_gate_entropy_ratio": adaptive_state.token_gate_entropy_ratio,
            "token_gate_patience": adaptive_state.token_gate_patience,
            "token_update_interval": adaptive_state.token_update_interval,
            "token_revert_entropy_ratio": adaptive_state.token_revert_entropy_ratio,
            "token_revert_patience": adaptive_state.token_revert_patience,
            "token_cooldown_steps": adaptive_state.token_cooldown_steps,
            "next_token_update_step": adaptive_state.next_token_update_step,
            "token_phase_enabled": adaptive_state.token_phase_enabled,
            "post_adapt_rule_history": adaptive_state.post_adapt_rule_history[-8:] if adaptive_state.post_adapt_rule_history else [],
            "adapted": adaptive_state.adapted,
            "target_num_rules": adaptive_state.target_num_rules if adaptive_state.target_num_rules is not None else model.num_rules,
            "total_warmup_tokens": adaptive_state.total_warmup_tokens,
            "warmup_avg_entropy": float(sum(adaptive_state.warmup_entropy) / max(len(adaptive_state.warmup_entropy), 1)) if adaptive_state.warmup_entropy else None,
        },
        "training_recipe": {
            "stage_order": [stage.name for stage in training_stages],
            "assistant_only_stages": [stage.name for stage in training_stages if stage.assistant_only_loss],
        },
        "model_state": model.state_dict(),
        "expert_store": expert_store.state_dict(),
        "cpu_token_embedding.weight": cpu_token_embedding.weight.data.cpu(),
        "optimizer_state": {
            "rule": optimizer_rule.state_dict(),
            "token": optimizer_token.state_dict(),
            "expert": optimizer_expert.state_dict(),
        },
    }
    atomic_torch_save(state_dict_to_save, checkpoint_path)
    logger.info("模型权重已原子保存至 %s", checkpoint_path)


if __name__ == "__main__":
    args = _parse_args()
    train_large_model(real_epochs=args.real_epochs, sft_epochs=args.sft_epochs)
