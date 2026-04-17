import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from large_scale_inference import (
    build_rule_candidate_cache,
    build_rule_token_index,
    decode_sequence,
    generate_tokens,
    load_active_tokens,
    load_runtime_assets,
)
from rule_token_engine import RuleTokenCausalModel
from tokenizer_utils import (
    discover_model_path,
    get_custom_tokenizer,
    get_logger,
    infer_model_config,
    load_vocab_info,
    resolve_project_path,
    resolve_protocol_tokens,
)

logger = get_logger()
_WHITESPACE_RE = re.compile(r"\s+")
_CLAUSE_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")


@dataclass
class EvalRecord:
    instruction: str
    input_text: str
    output: str
    answer_from: str
    human_verified: bool
    task_major: Tuple[str, ...]
    task_minor: Tuple[str, ...]
    domains: Tuple[str, ...]


@dataclass
class PromptBenchmarkItem:
    prompt_id: str
    instruction: str
    input_text: str
    prompt_text: str
    references: List[str]
    split: str
    expected_behavior: str
    task_major: List[str]
    task_minor: List[str]
    domains: List[str]
    source_counts: Dict[str, int]
    total_records: int
    selected_reference_count: int
    verified_reference_ratio: float


def normalize_text(text: str) -> str:
    text = str(text or "")
    text = _WHITESPACE_RE.sub(" ", text.strip())
    return text.lower()


def prompt_to_text(instruction: str, input_text: str) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if instruction and input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction or input_text


def prompt_hash(instruction: str, input_text: str) -> str:
    key = f"{instruction}\n<SEP>\n{input_text}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def stable_split(prompt_id: str) -> str:
    bucket = int(prompt_id[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "dev"
    return "test"


def load_eval_records(dataset_path: str) -> List[EvalRecord]:
    records: List[EvalRecord] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            data = json.loads(line)
            output = str(data.get("output", "")).strip()
            if not output:
                continue
            task_type = data.get("task_type", {}) or {}
            records.append(EvalRecord(
                instruction=str(data.get("instruction", "")).strip(),
                input_text=str(data.get("input", "")).strip(),
                output=output,
                answer_from=str(data.get("answer_from", "")).strip(),
                human_verified=bool(data.get("human_verified", False)),
                task_major=tuple(str(x).strip() for x in task_type.get("major", []) if str(x).strip()),
                task_minor=tuple(str(x).strip() for x in task_type.get("minor", []) if str(x).strip()),
                domains=tuple(str(x).strip() for x in data.get("domain", []) if str(x).strip()),
            ))
    return records


def choose_reference_records(records: Sequence[EvalRecord]) -> List[EvalRecord]:
    verified = [record for record in records if record.human_verified]
    chosen = verified if verified else list(records)
    dedup = {}
    for record in chosen:
        key = normalize_text(record.output)
        if key and key not in dedup:
            dedup[key] = record
    return list(dedup.values())


def extract_behavior_template(text: str, max_chars: int = 24) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
    first_clause = _CLAUSE_SPLIT_RE.split(normalized, maxsplit=1)[0].strip()
    if not first_clause:
        first_clause = normalized
    return first_clause[:max_chars]


def quantile(sorted_values: Sequence[int], q: float) -> int:
    if not sorted_values:
        return 0
    q = min(max(float(q), 0.0), 1.0)
    index = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * q))))
    return int(sorted_values[index])


def mine_behavior_templates(grouped_records: Dict[Tuple[str, str], List[EvalRecord]]) -> set:
    group_refs = [choose_reference_records(group) for group in grouped_records.values()]
    ref_lengths = [
        min(len(normalize_text(record.output)) for record in refs if normalize_text(record.output))
        for refs in group_refs
        if refs
    ]
    if not ref_lengths:
        return set()
    ref_lengths.sort()
    short_threshold = max(8, quantile(ref_lengths, 0.25))
    prompt_counter = Counter()
    short_group_count = 0
    for refs in group_refs:
        if not refs:
            continue
        shortest = min(len(normalize_text(record.output)) for record in refs if normalize_text(record.output))
        if shortest > short_threshold:
            continue
        short_group_count += 1
        templates = {
            extract_behavior_template(record.output)
            for record in refs
            if extract_behavior_template(record.output)
        }
        prompt_counter.update(templates)
    min_support = max(3, int(math.sqrt(max(short_group_count, 1)) / 2))
    return {
        template
        for template, count in prompt_counter.items()
        if count >= min_support
    }


def infer_expected_behavior(references: Sequence[str], behavior_templates: set) -> str:
    if not references or not behavior_templates:
        return "answer"
    matched = 0
    for ref in references:
        if extract_behavior_template(ref) in behavior_templates:
            matched += 1
    return "refusal" if matched * 2 >= len(references) else "answer"


def build_benchmark(records: Sequence[EvalRecord]) -> List[PromptBenchmarkItem]:
    grouped_records: Dict[Tuple[str, str], List[EvalRecord]] = {}
    for record in records:
        grouped_records.setdefault((record.instruction, record.input_text), []).append(record)

    behavior_templates = mine_behavior_templates(grouped_records)
    benchmark: List[PromptBenchmarkItem] = []
    for (instruction, input_text), group in grouped_records.items():
        references = choose_reference_records(group)
        if not references:
            continue
        prompt_id = prompt_hash(instruction, input_text)
        prompt_text = prompt_to_text(instruction, input_text)
        if not prompt_text:
            continue
        source_counts = Counter(record.answer_from or "unknown" for record in group)
        major_counts = Counter(tag for record in group for tag in record.task_major)
        minor_counts = Counter(tag for record in group for tag in record.task_minor)
        domain_counts = Counter(tag for record in group for tag in record.domains)
        benchmark.append(PromptBenchmarkItem(
            prompt_id=prompt_id,
            instruction=instruction,
            input_text=input_text,
            prompt_text=prompt_text,
            references=[record.output for record in references],
            split=stable_split(prompt_id),
            expected_behavior=infer_expected_behavior([record.output for record in references], behavior_templates),
            task_major=[name for name, _ in major_counts.most_common()],
            task_minor=[name for name, _ in minor_counts.most_common()],
            domains=[name for name, _ in domain_counts.most_common()],
            source_counts=dict(source_counts),
            total_records=len(group),
            selected_reference_count=len(references),
            verified_reference_ratio=sum(1 for record in references if record.human_verified) / max(len(references), 1),
        ))
    benchmark.sort(key=lambda item: (item.split, item.prompt_id))
    return benchmark


def counter_f1(pred_counter: Counter, ref_counter: Counter) -> float:
    overlap = sum((pred_counter & ref_counter).values())
    pred_total = sum(pred_counter.values())
    ref_total = sum(ref_counter.values())
    if pred_total == 0 or ref_total == 0 or overlap == 0:
        return 0.0
    precision = overlap / pred_total
    recall = overlap / ref_total
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def char_counter(text: str, ngram: int) -> Counter:
    normalized = normalize_text(text)
    if not normalized:
        return Counter()
    if len(normalized) < ngram:
        return Counter([normalized]) if ngram == 1 else Counter()
    return Counter(normalized[i:i + ngram] for i in range(len(normalized) - ngram + 1))


def lcs_length(a: str, b: str) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        curr = [0]
        for j, cb in enumerate(b, start=1):
            if ca == cb:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred = normalize_text(prediction)
    ref = normalize_text(reference)
    if not pred or not ref:
        return 0.0
    lcs = lcs_length(pred, ref)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def length_similarity(prediction: str, reference: str) -> float:
    pred_len = max(1, len(normalize_text(prediction)))
    ref_len = max(1, len(normalize_text(reference)))
    return math.exp(-abs(math.log(pred_len / ref_len)))


def score_prediction(prediction: str, references: Sequence[str]) -> Dict[str, float]:
    if not references:
        return {
            "final_score": 0.0,
            "char_f1": 0.0,
            "bigram_f1": 0.0,
            "rouge_l_f1": 0.0,
            "length_similarity": 0.0,
        }
    pred_uni = char_counter(prediction, 1)
    pred_bi = char_counter(prediction, 2)
    best = None
    for reference in references:
        metrics = {
            "char_f1": counter_f1(pred_uni, char_counter(reference, 1)),
            "bigram_f1": counter_f1(pred_bi, char_counter(reference, 2)),
            "rouge_l_f1": rouge_l_f1(prediction, reference),
            "length_similarity": length_similarity(prediction, reference),
        }
        metrics["final_score"] = (
            0.35 * metrics["char_f1"]
            + 0.25 * metrics["bigram_f1"]
            + 0.40 * metrics["rouge_l_f1"]
        ) * metrics["length_similarity"]
        if best is None or metrics["final_score"] > best["final_score"]:
            best = metrics
    return best


def aggregate_metric_rows(rows: Sequence[Dict]) -> Dict[str, float]:
    if not rows:
        return {}
    metrics = ("final_score", "char_f1", "bigram_f1", "rouge_l_f1", "length_similarity")
    aggregate = {name: 0.0 for name in metrics}
    empty_predictions = 0
    for row in rows:
        for name in metrics:
            aggregate[name] += float(row["metrics"][name])
        empty_predictions += int(not normalize_text(row["prediction"]))
    count = len(rows)
    aggregate = {name: value / count for name, value in aggregate.items()}
    aggregate["sample_count"] = count
    aggregate["empty_rate"] = empty_predictions / count
    return aggregate


def build_slice_report(rows: Sequence[Dict], key: str, top_k: int = 10) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[Dict]] = {}
    for row in rows:
        values = row["item"].get(key) or []
        for value in values[:1]:
            buckets.setdefault(value, []).append(row)
    ranked = sorted(buckets.items(), key=lambda item: len(item[1]), reverse=True)[:top_k]
    return {name: aggregate_metric_rows(bucket_rows) for name, bucket_rows in ranked}


def benchmark_summary(items: Sequence[PromptBenchmarkItem]) -> Dict:
    split_counts = Counter(item.split for item in items)
    behavior_counts = Counter(item.expected_behavior for item in items)
    major_counts = Counter(tag for item in items for tag in item.task_major[:1])
    domain_counts = Counter(tag for item in items for tag in item.domains[:1])
    reference_counts = sorted(item.selected_reference_count for item in items)
    return {
        "prompt_count": len(items),
        "split_counts": dict(split_counts),
        "behavior_counts": dict(behavior_counts),
        "top_task_major": major_counts.most_common(10),
        "top_domains": domain_counts.most_common(10),
        "reference_count_p50": quantile(reference_counts, 0.50) if reference_counts else 0,
        "reference_count_p90": quantile(reference_counts, 0.90) if reference_counts else 0,
    }


class LocalInferenceEngine:
    def __init__(self, checkpoint_path: Optional[str] = None, max_new_tokens: Optional[int] = None):
        self.vocab = load_vocab_info()
        self.enc, self.special_tokens = get_custom_tokenizer()
        self.protocol_tokens = resolve_protocol_tokens(self.vocab)
        self.special_tokens_inv = {value: key for key, value in self.special_tokens.items()}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path or discover_model_path()
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"未找到模型检查点: {self.checkpoint_path}。"
                "请先训练模型，或通过 --checkpoint 显式指定包含当前格式权重的检查点。"
            )
        model_config = infer_model_config(self.vocab["vocab_size"], self.checkpoint_path)
        self.model = RuleTokenCausalModel(vocab_size=self.vocab["vocab_size"], **model_config).to(self.device)
        try:
            self.expert_store, self.embedding_weight_cpu = load_runtime_assets(self.checkpoint_path, self.model)
        except KeyError as exc:
            raise RuntimeError(
                "当前检查点缺少评测所需的 `cpu_token_embedding.weight`，"
                "通常说明它不是当前训练链路导出的完整检查点。"
                "请重新训练，或指定新的完整检查点。"
            ) from exc
        self.model.eval()
        self.rule_to_tokens, self.token_to_cell = build_rule_token_index(self.model, self.embedding_weight_cpu)

        active_tokens = load_active_tokens()
        if active_tokens:
            filtered = {}
            for rule_id, token_ids in self.rule_to_tokens.items():
                valid = [token_id for token_id in token_ids if token_id in active_tokens]
                if valid:
                    filtered[rule_id] = valid
            self.rule_to_tokens = filtered
        self.eos_id = self.protocol_tokens["eos"]["id"]

        # Formula ceiling D · log R is for open-ended generative playback; far
        # too long for benchmark decoding.  Callers can override via CLI.
        formula_cap = max(1, int(self.model.embed_size * max(math.log(self.model.num_rules), 1.0)))
        self.max_new_tokens = int(max_new_tokens) if max_new_tokens is not None else formula_cap

        # Fast-path state: constant for the lifetime of the engine.
        self.rule_cache = build_rule_candidate_cache(
            self.rule_to_tokens, self.embedding_weight_cpu, self.device
        )
        self.token_to_cell_device = self.token_to_cell.to(self.device)

        # Warm the paged expert cache for every active rule so per-step calls
        # become dict lookups (no CPU→GPU traffic inside the hot loop).
        if self.expert_store is not None and self.rule_cache:
            self.expert_store.build_runtime(
                sorted(self.rule_cache.keys()), device=self.device, training=False
            )

    def generate(self, prompt_text: str) -> str:
        ctx_tokens = self.enc.encode(prompt_text, allowed_special="all")
        if not ctx_tokens:
            return ""
        generated = generate_tokens(
            self.model,
            self.expert_store,
            self.embedding_weight_cpu,
            self.token_to_cell,
            self.rule_to_tokens,
            ctx_tokens,
            self.eos_id,
            self.device,
            self.max_new_tokens,
            rule_cache=self.rule_cache,
            token_to_cell_device=self.token_to_cell_device,
        )
        return decode_sequence(self.enc, self.special_tokens_inv, self.protocol_tokens, generated).strip()


def save_json(payload: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def stratified_subsample(
    items: Sequence[PromptBenchmarkItem],
    sample_size: int,
) -> List[PromptBenchmarkItem]:
    """Deterministic stratified subsampling over (behavior, primary task).

    Rationale (first-principles): Monte-Carlo std error on aggregate metrics
    shrinks only as 1/√N, so evaluating N=4380 vs N=500 changes the estimator
    precision by ~3×.  That trade is worth taking when the population is
    heavily multi-modal (multiple task domains, refusal vs answer behavior),
    *provided* each stratum stays proportionally represented.

    Strategy:
      * Partition by (expected_behavior, task_major[0]).
      * Allocate per-stratum quotas ∝ stratum size, guaranteeing ≥1 item per
        non-empty stratum so small but important slices aren't silently
        eliminated.
      * Within a stratum, prompt_id order is already a uniform hash of the
        prompt text (see `stable_split`), so taking the prefix is equivalent
        to uniform random sampling while remaining reproducible.
    """
    if sample_size <= 0 or sample_size >= len(items):
        return list(items)

    strata: Dict[Tuple[str, str], List[PromptBenchmarkItem]] = {}
    for item in items:
        key = (item.expected_behavior, item.task_major[0] if item.task_major else "unknown")
        strata.setdefault(key, []).append(item)

    total = sum(len(v) for v in strata.values())
    # First pass: floor allocation with a guaranteed minimum of 1.
    quotas: Dict[Tuple[str, str], int] = {}
    allocated = 0
    for key, group in strata.items():
        quota = max(1, int(sample_size * len(group) / total))
        quota = min(quota, len(group))
        quotas[key] = quota
        allocated += quota

    # Reconcile over/under-allocation by adjusting the largest strata.
    sorted_keys = sorted(strata.keys(), key=lambda k: -len(strata[k]))
    while allocated > sample_size:
        for key in sorted_keys:
            if allocated <= sample_size:
                break
            if quotas[key] > 1:
                quotas[key] -= 1
                allocated -= 1
    while allocated < sample_size:
        for key in sorted_keys:
            if allocated >= sample_size:
                break
            if quotas[key] < len(strata[key]):
                quotas[key] += 1
                allocated += 1

    selected: List[PromptBenchmarkItem] = []
    for key, group in strata.items():
        group_sorted = sorted(group, key=lambda it: it.prompt_id)
        selected.extend(group_sorted[:quotas[key]])
    selected.sort(key=lambda it: (it.split, it.prompt_id))
    return selected


def evaluate_items(items: Sequence[PromptBenchmarkItem], engine: LocalInferenceEngine, limit: Optional[int] = None) -> List[Dict]:
    rows = []
    selected_items = list(items[:limit]) if limit is not None else list(items)
    for index, item in enumerate(selected_items, start=1):
        prediction = engine.generate(item.prompt_text)
        metrics = score_prediction(prediction, item.references)
        row = {
            "item": asdict(item),
            "prediction": prediction,
            "metrics": metrics,
        }
        rows.append(row)
        if index % 10 == 0 or index == len(selected_items):
            logger.info("评测进度: %d/%d", index, len(selected_items))
    return rows


def build_report(rows: Sequence[Dict], summary: Dict) -> Dict:
    return {
        "benchmark_summary": summary,
        "overall": aggregate_metric_rows(rows),
        "by_behavior": {
            behavior: aggregate_metric_rows([row for row in rows if row["item"]["expected_behavior"] == behavior])
            for behavior in sorted({row["item"]["expected_behavior"] for row in rows})
        },
        "by_task_major": build_slice_report(rows, "task_major"),
        "by_domain": build_slice_report(rows, "domains"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于本地 JSONL 数据构建并执行模型评测。")
    parser.add_argument("--dataset", type=str, default=resolve_project_path("COIG-CQIA-full.jsonl"), help="本地评测数据路径")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型检查点路径，默认自动发现")
    parser.add_argument("--split", type=str, default="test", choices=("train", "dev", "test"), help="执行评测的数据切分")
    parser.add_argument("--limit", type=int, default=None, help="仅评测前 N 条样本（按 prompt_id 排序裁前缀，不保证分层均衡）")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="在 split 内做确定性分层抽样，按 (expected_behavior, task_major[0]) 分层按比例抽取，保证小切片不被清零。推荐 300-800。",
    )
    parser.add_argument("--prepare-only", action="store_true", help="仅构建 benchmark 与统计信息，不加载模型")
    parser.add_argument("--output-prefix", type=str, default=resolve_project_path("local_eval"), help="输出文件前缀")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="每条样本最多生成的 token 数上限（默认 128）。原公式 D·log(R) 会给出 2000+ 的无效上限。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"未找到评测数据: {args.dataset}")

    logger.info("加载本地评测数据: %s", args.dataset)
    records = load_eval_records(args.dataset)
    benchmark = build_benchmark(records)
    summary = benchmark_summary(benchmark)
    logger.info("Prompt 数: %d | 切分: %s", summary["prompt_count"], summary["split_counts"])

    benchmark_path = f"{args.output_prefix}_benchmark.json"
    summary_path = f"{args.output_prefix}_summary.json"
    save_json({
        "dataset_path": args.dataset,
        "summary": summary,
        "items": [asdict(item) for item in benchmark],
    }, benchmark_path)
    save_json(summary, summary_path)

    if args.prepare_only:
        logger.info("已完成 benchmark 构建，输出: %s, %s", benchmark_path, summary_path)
        return

    eval_items = [item for item in benchmark if item.split == args.split]
    if args.sample_size is not None and args.sample_size > 0:
        before = len(eval_items)
        eval_items = stratified_subsample(eval_items, args.sample_size)
        logger.info(
            "已对 split=%s 做分层抽样：%d → %d（按 expected_behavior × task_major 保比例）",
            args.split, before, len(eval_items),
        )
    effective_count = len(eval_items) if args.limit is None else min(len(eval_items), args.limit)
    logger.info("开始评测 split=%s | 样本数=%d", args.split, effective_count)
    engine = LocalInferenceEngine(checkpoint_path=args.checkpoint, max_new_tokens=args.max_new_tokens)
    rows = evaluate_items(eval_items, engine, limit=args.limit)
    report = build_report(rows, summary)

    prediction_path = f"{args.output_prefix}_{args.split}_predictions.json"
    report_path = f"{args.output_prefix}_{args.split}_report.json"
    save_json({"rows": rows}, prediction_path)
    save_json(report, report_path)
    logger.info("评测完成，输出: %s, %s", prediction_path, report_path)


if __name__ == "__main__":
    main()
