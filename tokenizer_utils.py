import glob
import hashlib
import json
import logging
import math
import os
import random
import re
import tempfile
from typing import Callable, List, Optional

import tiktoken
import torch


_logger = logging.getLogger("hrsp")
TAG_PATTERN = re.compile(r"<(/?)([A-Za-z0-9_:\-]+)>")


def setup_logging(level=None):
    if _logger.handlers:
        return _logger
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    ))
    _logger.addHandler(handler)
    _logger.setLevel(level or logging.INFO)
    return _logger


def get_logger():
    if not _logger.handlers:
        setup_logging()
    return _logger


def get_project_dir():
    return os.path.dirname(os.path.abspath(__file__))


def resolve_project_path(*parts):
    return os.path.join(get_project_dir(), *parts)


def next_power_of_two(value):
    if value <= 1:
        return 1
    return 1 << math.ceil(math.log2(value))


def derive_seed(*artifacts) -> int:
    h = hashlib.sha256()
    for a in artifacts:
        h.update(str(a).encode())
    return int.from_bytes(h.digest()[:4], "big")


def set_deterministic_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(obj, path):
    dir_name = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def rms_normalize(x: torch.Tensor, dim: int = -1, eps: Optional[float] = None) -> torch.Tensor:
    x_fp32 = x.float()
    axis = dim if dim >= 0 else x_fp32.dim() + dim
    feature_dim = x_fp32.size(axis) if x_fp32.dim() > 0 else 1
    epsilon = eps if eps is not None else torch.finfo(torch.float32).eps * max(feature_dim, 1)
    rms = torch.sqrt(torch.mean(x_fp32 * x_fp32, dim=dim, keepdim=True) + epsilon)
    return (x_fp32 / rms).to(x.dtype)


def validate_sequence(sequence, vocab_size: int, min_length: int = 2) -> bool:
    if isinstance(sequence, torch.Tensor):
        if sequence.numel() < min_length:
            return False
        if sequence.max().item() >= vocab_size or sequence.min().item() < 0:
            return False
    elif isinstance(sequence, list):
        if len(sequence) < min_length:
            return False
        if not sequence or max(sequence) >= vocab_size or min(sequence) < 0:
            return False
    else:
        return False
    return True


def sliding_window_chunks(token_ids: list, max_seq_len: int) -> List[list]:
    if len(token_ids) <= max_seq_len:
        return [token_ids]
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    stride = max(1, int(max_seq_len / phi))
    chunks = []
    for start in range(0, len(token_ids) - max_seq_len + 1, stride):
        chunks.append(token_ids[start:start + max_seq_len])
    last_start = len(token_ids) - max_seq_len
    if chunks and chunks[-1] != token_ids[last_start:]:
        chunks.append(token_ids[last_start:last_start + max_seq_len])
    return chunks


def infer_dataset_geometry_shared(
    jsonl_path: str,
    enc,
    vocab_size: int,
    extract_fn: Callable[[str], Optional[str]],
):
    file_size = max(1, os.path.getsize(jsonl_path))
    sample_budget = max(16, int(math.sqrt(file_size / max(math.log(max(vocab_size, 2)), 1.0))))
    token_lengths: List[int] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            text = extract_fn(raw)
            if not text:
                continue
            token_lengths.append(len(enc.encode(text, allowed_special="all")))
            if len(token_lengths) >= sample_budget:
                break
    if not token_lengths:
        token_lengths = [max(1, int(math.sqrt(vocab_size)))]
    t = torch.tensor(token_lengths, dtype=torch.float32)
    max_len = torch.quantile(t, 0.95).item()
    max_seq_len = next_power_of_two(max(2, int(max_len)))
    chunk_token_budget = max(vocab_size, int(vocab_size * max(math.log(max(max_seq_len, 2)), 1.0)))
    return max_seq_len, chunk_token_budget


def discover_dataset_path():
    matches = glob.glob(resolve_project_path("*.jsonl"))
    if not matches:
        return None
    return max(matches, key=os.path.getsize)


def discover_vocab_path():
    matches = glob.glob(resolve_project_path("*vocab*.json"))
    if matches:
        return max(matches, key=os.path.getmtime)
    return resolve_project_path("distilled_vocab.json")


def discover_model_path():
    matches = glob.glob(resolve_project_path("*.pth"))
    if matches:
        return max(matches, key=os.path.getmtime)
    return resolve_project_path("large_scale_latent_model.pth")


def iter_dataset_records(jsonl_path, sample_budget=None):
    if jsonl_path is None or not os.path.exists(jsonl_path):
        return
    yielded = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                data = line
            yield data
            yielded += 1
            if sample_budget is not None and yielded >= sample_budget:
                break


def build_empty_schema(mode="raw_text"):
    return {
        "mode": mode,
        "record_fields": [],
        "message_container": None,
        "message_role_key": None,
        "message_content_key": None,
        "message_auxiliary_keys": [],
        "semantic_layout": [],
    }


def normalize_field_name(field_name):
    collapsed = re.sub(r"[^a-z0-9]+", "_", str(field_name).lower()).strip("_")
    tokens = tuple(part for part in collapsed.split("_") if part)
    return collapsed, tokens


def semantic_field_score(field_name, aliases):
    collapsed, tokens = normalize_field_name(field_name)
    score = 0
    for alias in aliases:
        alias_collapsed, alias_tokens = normalize_field_name(alias)
        if collapsed == alias_collapsed:
            score = max(score, 100)
            continue
        if alias_collapsed and (collapsed.startswith(alias_collapsed + "_") or collapsed.endswith("_" + alias_collapsed)):
            score = max(score, 40)
        if alias_tokens and all(token in tokens for token in alias_tokens):
            score = max(score, 20 if len(alias_tokens) == 1 else 30)
    return score


def infer_semantic_layout(field_names, field_stats=None):
    if not field_names:
        return []

    semantic_aliases = {
        "prompt": ("instruction", "prompt", "question", "query", "request", "user"),
        "context": ("input", "context", "background", "passage", "document", "reference"),
        "response": ("output", "response", "answer", "completion", "reply", "assistant", "target", "result"),
        "system": ("system", "system_prompt", "developer", "persona"),
    }
    field_order = {field_name: index for index, field_name in enumerate(field_names)}

    def rank_field(semantic_name, minimum_score, exclude=None):
        exclude = exclude or set()
        ranked = []
        for field_name in field_names:
            if field_name in exclude:
                continue
            score = semantic_field_score(field_name, semantic_aliases[semantic_name])
            if score < minimum_score:
                continue
            stats = field_stats.get(field_name, {}) if field_stats else {}
            ranked.append((
                score,
                stats.get("count", 0),
                stats.get("char_sum", 0.0),
                -field_order[field_name],
                field_name,
            ))
        if not ranked:
            return None
        ranked.sort(reverse=True)
        return ranked[0][-1]

    response_field = rank_field("response", minimum_score=40)
    prompt_field = rank_field("prompt", minimum_score=20, exclude={response_field} if response_field else set())
    if prompt_field is None or response_field is None:
        return []

    used_fields = {prompt_field, response_field}
    context_field = rank_field("context", minimum_score=40, exclude=used_fields)
    if context_field is not None:
        used_fields.add(context_field)
    system_field = rank_field("system", minimum_score=40, exclude=used_fields)

    user_fields = [prompt_field]
    if context_field is not None:
        user_fields.append(context_field)
    user_fields.sort(key=lambda field_name: field_order[field_name])

    layout = []
    if system_field is not None:
        layout.append({"role": "system", "fields": [system_field]})
    layout.append({"role": "user", "fields": user_fields})
    layout.append({"role": "assistant", "fields": [response_field]})
    return layout


def infer_dataset_schema(jsonl_path=None):
    jsonl_path = jsonl_path or discover_dataset_path()
    if jsonl_path is None:
        return build_empty_schema()

    file_size = max(1, os.path.getsize(jsonl_path))
    estimated_bytes_per_line = max(1.0, math.log(file_size + math.e)) ** 2
    sample_budget = max(1, int(math.sqrt(file_size / estimated_bytes_per_line)))
    raw_text_count = 0
    record_field_stats = {}
    container_stats = {}

    for record in iter_dataset_records(jsonl_path, sample_budget=sample_budget):
        if isinstance(record, str):
            raw_text_count += 1
            continue
        if not isinstance(record, dict):
            continue

        ordered_keys = list(record.keys())
        for position, key in enumerate(ordered_keys):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                stats = record_field_stats.setdefault(key, {"count": 0, "position_sum": 0.0, "char_sum": 0.0})
                stats["count"] += 1
                stats["position_sum"] += position
                stats["char_sum"] += len(value)
            elif isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                stats = container_stats.setdefault(key, {"count": 0, "entries": 0, "field_stats": {}})
                stats["count"] += 1
                stats["entries"] += len(value)
                for item in value:
                    for item_key, item_value in item.items():
                        if isinstance(item_value, str) and item_value.strip():
                            field_stats = stats["field_stats"].setdefault(item_key, {"count": 0, "char_sum": 0.0, "values": {}})
                            field_stats["count"] += 1
                            field_stats["char_sum"] += len(item_value)
                            field_stats["values"][item_value] = field_stats["values"].get(item_value, 0) + 1

    record_fields = sorted(
        record_field_stats.items(),
        key=lambda item: (
            item[1]["position_sum"] / max(item[1]["count"], 1),
            -item[1]["count"],
            item[0]
        )
    )
    best_container = None
    if container_stats:
        best_container = max(
            container_stats.items(),
            key=lambda item: (
                item[1]["count"],
                item[1]["entries"],
                sum(stat["char_sum"] for stat in item[1]["field_stats"].values())
            )
        )

    if best_container is None:
        ordered_record_fields = [key for key, stats in record_fields if stats["count"] > 0]
        semantic_layout = infer_semantic_layout(
            ordered_record_fields,
            {key: stats for key, stats in record_fields},
        )
        if semantic_layout:
            schema = build_empty_schema(mode="semantic_messages")
            schema["record_fields"] = ordered_record_fields
            schema["semantic_layout"] = semantic_layout
            return schema
        mode = "record_fields" if record_fields else "raw_text"
        schema = build_empty_schema(mode=mode)
        schema["record_fields"] = ordered_record_fields
        return schema

    container_name, container_payload = best_container
    field_stats = container_payload["field_stats"]
    content_key = max(
        field_stats.items(),
        key=lambda item: (item[1]["char_sum"], item[1]["count"], item[0])
    )[0]
    role_candidates = []
    for field_name, stats in field_stats.items():
        if field_name == content_key:
            continue
        unique_count = len(stats["values"])
        low_cardinality_score = stats["count"] / max(unique_count, 1)
        role_candidates.append((low_cardinality_score, stats["count"], field_name))
    role_key = max(role_candidates)[2] if role_candidates else None
    auxiliary_keys = sorted(
        [
            (field_name, stats)
            for field_name, stats in field_stats.items()
            if field_name not in {content_key, role_key}
        ],
        key=lambda item: (-item[1]["char_sum"], -item[1]["count"], item[0])
    )
    schema = build_empty_schema(mode="messages")
    schema["record_fields"] = [key for key, stats in record_fields if stats["count"] > 0]
    schema["message_container"] = container_name
    schema["message_role_key"] = role_key
    schema["message_content_key"] = content_key
    schema["message_auxiliary_keys"] = [key for key, _ in auxiliary_keys]
    return schema


def extract_record_text(data):
    schema = infer_record_schema(data)
    if schema["mode"] == "raw_text":
        return data if isinstance(data, str) else ""
    if schema["mode"] in {"messages", "semantic_messages"}:
        segments = []
        for message in normalize_schema_messages(data, schema):
            if message.get("content", ""):
                segments.append(message["content"])
            if message.get("auxiliary_content", ""):
                segments.append(message["auxiliary_content"])
        return "\n".join(segments)
    return "\n".join(part for part in extract_schema_record_parts(data, schema) if part)


def infer_record_schema(data):
    if isinstance(data, str):
        return build_empty_schema()
    if not isinstance(data, dict):
        return build_empty_schema()
    for key, value in data.items():
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            field_names = {}
            for item in value:
                for item_key, item_value in item.items():
                    if isinstance(item_value, str) and item_value.strip():
                        stats = field_names.setdefault(item_key, {"count": 0, "char_sum": 0.0, "values": {}})
                        stats["count"] += 1
                        stats["char_sum"] += len(item_value)
                        stats["values"][item_value] = stats["values"].get(item_value, 0) + 1
            if field_names:
                content_key = max(field_names.items(), key=lambda item: (item[1]["char_sum"], item[1]["count"], item[0]))[0]
                role_candidates = []
                for field_name, stats in field_names.items():
                    if field_name == content_key:
                        continue
                    role_candidates.append((stats["count"] / max(len(stats["values"]), 1), stats["count"], field_name))
                role_key = max(role_candidates)[2] if role_candidates else None
                auxiliary_keys = [
                    field_name
                    for field_name, stats in sorted(
                        field_names.items(),
                        key=lambda item: (-item[1]["char_sum"], -item[1]["count"], item[0])
                    )
                    if field_name not in {content_key, role_key}
                ]
                record_fields = [field for field, value in data.items() if isinstance(value, str) and value.strip()]
                schema = build_empty_schema(mode="messages")
                schema["record_fields"] = record_fields
                schema["message_container"] = key
                schema["message_role_key"] = role_key
                schema["message_content_key"] = content_key
                schema["message_auxiliary_keys"] = auxiliary_keys
                return schema
    record_fields = [field for field, value in data.items() if isinstance(value, str) and value.strip()]
    semantic_layout = infer_semantic_layout(record_fields)
    if semantic_layout:
        schema = build_empty_schema(mode="semantic_messages")
        schema["record_fields"] = record_fields
        schema["semantic_layout"] = semantic_layout
        return schema
    schema = build_empty_schema(mode="record_fields" if record_fields else "raw_text")
    schema["record_fields"] = record_fields
    return schema


def extract_schema_record_parts(data, schema):
    if not isinstance(data, dict):
        return [data] if isinstance(data, str) and data else []
    return [data.get(field, "") for field in schema.get("record_fields", []) if isinstance(data.get(field, ""), str) and data.get(field, "").strip()]


def normalize_schema_messages(data, schema):
    ROLE_MAPPING = {
        "human": "user", "user": "user", "prompter": "user", "client": "user",
        "gpt": "assistant", "assistant": "assistant", "bot": "assistant", "model": "assistant",
        "system": "system", "instruction": "system"
    }

    if schema["mode"] == "semantic_messages" and isinstance(data, dict):
        messages = []
        for entry in schema.get("semantic_layout", []):
            fields = entry.get("fields", [])
            segments = [
                data.get(field_name, "").strip()
                for field_name in fields
                if isinstance(data.get(field_name, ""), str) and data.get(field_name, "").strip()
            ]
            if not segments:
                continue
            messages.append({
                "role": entry.get("role", "__slot_0__"),
                "content": "\n\n".join(segments),
                "auxiliary_content": "",
            })
        return messages

    if schema["mode"] == "messages" and isinstance(data, dict):
        messages = []
        for index, item in enumerate(data.get(schema["message_container"], [])):
            if not isinstance(item, dict):
                continue
            auxiliary_parts = []
            for auxiliary_key in schema.get("message_auxiliary_keys", []):
                value = item.get(auxiliary_key, "")
                if isinstance(value, str) and value.strip():
                    auxiliary_parts.append(value)
            
            raw_role = item.get(schema.get("message_role_key"), f"__slot_{index}__") if schema.get("message_role_key") else f"__slot_{index}__"
            raw_role_lower = str(raw_role).lower()
            mapped_role = ROLE_MAPPING.get(raw_role_lower, raw_role)
            
            messages.append({
                "role": mapped_role,
                "content": item.get(schema.get("message_content_key"), "") if schema.get("message_content_key") else "",
                "auxiliary_content": "\n".join(auxiliary_parts),
            })
        return messages
    if schema["mode"] == "record_fields":
        parts = extract_schema_record_parts(data, schema)
        return [
            {
                "role": f"__slot_{index}__",
                "content": part,
                "auxiliary_content": "",
            }
            for index, part in enumerate(parts)
        ]
    if isinstance(data, str):
        return [{"role": "__slot_0__", "content": data, "auxiliary_content": ""}]
    return []


def compute_file_sha256(file_path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def discover_base_tokenizer(jsonl_path=None):
    jsonl_path = jsonl_path or discover_dataset_path()
    candidate_names = ("o200k_base", "cl100k_base", "p50k_base", "r50k_base", "gpt2")
    available_encodings = []
    for name in candidate_names:
        try:
            available_encodings.append((name, tiktoken.get_encoding(name)))
        except Exception:
            continue
    if not available_encodings:
        raise ValueError("未发现可用的基础 tokenizer。")
    if jsonl_path is None:
        return available_encodings[0][0]

    sample_budget = 1000
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                text = extract_record_text(data)
            except json.JSONDecodeError:
                text = line
            if not text:
                continue
            samples.append(text)
            if len(samples) >= sample_budget:
                break
    if not samples:
        return available_encodings[0][0]

    scored_candidates = []
    for name, encoding in available_encodings:
        token_lengths = []
        char_lengths = []
        for sample in samples:
            token_lengths.append(len(encoding.encode(sample, allowed_special="all")))
            char_lengths.append(max(1, len(sample)))
        token_tensor = torch.tensor(token_lengths, dtype=torch.float32)
        char_tensor = torch.tensor(char_lengths, dtype=torch.float32)
        
        mean_token_length = token_tensor.mean().item()
        mean_char_length = char_tensor.mean().item()
        compression_ratio = mean_token_length / max(mean_char_length, 1e-5)
        score = compression_ratio
        scored_candidates.append((score, mean_token_length, name))
    scored_candidates.sort()
    return scored_candidates[0][2]


def infer_protocol_tokens(jsonl_path):
    role_stats = {}
    segment_stats = {}
    content_lengths = []
    sequence_index = 0

    def observe_role(source_role, amount=1):
        stats = role_stats.setdefault(source_role, {"count": 0, "position_sum": 0.0})
        stats["count"] += amount
        stats["position_sum"] += sequence_index

    def observe_segment(source_tag, boundary, amount=1):
        stats = segment_stats.setdefault((source_tag, boundary), {"count": 0, "position_sum": 0.0})
        stats["count"] += amount
        stats["position_sum"] += sequence_index

    if jsonl_path is None:
        return [
            {"topology_class": 0, "slot": 0, "source": "__pad__", "count": 1, "symbol": "<|pad_0|>"},
            {"topology_class": 3, "slot": 0, "source": "__eos__", "count": 1, "symbol": "<|eos_0|>"},
        ]

    schema = infer_dataset_schema(jsonl_path)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            sample_char_len = 0
            messages = normalize_schema_messages(data, schema)
            for message in messages:
                role = message.get("role", "") or "__unknown__"
                content = message.get("content", "")
                auxiliary_content = message.get("auxiliary_content", "")
                observe_role(role)
                sequence_index += 1
                sample_char_len += len(content) + len(auxiliary_content)

                for slash, tag_name in TAG_PATTERN.findall(content):
                    observe_segment(tag_name.lower(), "close" if slash else "open")
                    sequence_index += 1

                if auxiliary_content:
                    observe_segment("__auxiliary__", "open")
                    sequence_index += 1
                    observe_segment("__auxiliary__", "close")
                    sequence_index += 1

            if sample_char_len > 0:
                content_lengths.append(sample_char_len)

    role_order = sorted(
        role_stats.items(),
        key=lambda item: (
            -item[1]["count"],
            item[1]["position_sum"] / max(item[1]["count"], 1),
            item[0],
        ),
    )
    role_catalog = [
        {
            "topology_class": 1,
            "slot": index,
            "source": source_role,
            "count": stats["count"],
            "symbol": f"<|role_{index}|>",
        }
        for index, (source_role, stats) in enumerate(role_order)
        if stats["count"] > 0
    ]

    segment_groups = {}
    for (source_tag, boundary), stats in segment_stats.items():
        group = segment_groups.setdefault(source_tag, {"count": 0, "position_sum": 0.0, "boundaries": {}})
        group["count"] += stats["count"]
        group["position_sum"] += stats["position_sum"]
        group["boundaries"][boundary] = stats

    segment_order = sorted(
        segment_groups.items(),
        key=lambda item: (
            -item[1]["count"],
            item[1]["position_sum"] / max(item[1]["count"], 1),
            item[0],
        ),
    )
    segment_catalog = []
    for slot_index, (source_tag, group) in enumerate(segment_order):
        for boundary in ("open", "close"):
            stats = group["boundaries"].get(boundary)
            if stats is None:
                continue
            segment_catalog.append({
                "topology_class": 2,
                "slot": slot_index,
                "boundary": boundary,
                "source": source_tag,
                "count": stats["count"],
                "symbol": f"<|segment_{slot_index}_{boundary}|>",
            })

    if content_lengths:
        content_tensor = torch.tensor(content_lengths, dtype=torch.float32)
        variance = content_tensor.var(unbiased=False).item()
        pad_count = max(1, int(math.sqrt(math.sqrt(variance))))
    else:
        pad_count = 1

    eos_count = max(1, sum(stats["count"] for stats in role_stats.values()))
    return [
        {"topology_class": 0, "slot": 0, "source": "__pad__", "count": pad_count, "symbol": "<|pad_0|>"},
        *role_catalog,
        *segment_catalog,
        {"topology_class": 3, "slot": 0, "source": "__eos__", "count": eos_count, "symbol": "<|eos_0|>"},
    ]


def iter_training_texts(jsonl_path):
    if jsonl_path is None or not os.path.exists(jsonl_path):
        return
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                text = line
            else:
                text = extract_record_text(data) if isinstance(data, dict) else data if isinstance(data, str) else ""
            if isinstance(text, str) and text.strip():
                yield text


def build_dataset_vocab_info(jsonl_path=None):
    jsonl_path = jsonl_path or discover_dataset_path()
    if jsonl_path is None:
        raise FileNotFoundError("未发现可用的 JSONL 数据集，无法构建训练数据词表。")

    base_tokenizer = discover_base_tokenizer(jsonl_path)
    base_enc = tiktoken.get_encoding(base_tokenizer)
    observed_base_token_ids = set()
    for text in iter_training_texts(jsonl_path):
        observed_base_token_ids.update(base_enc.encode(text, allowed_special=set()))

    observed_base_token_ids = sorted(int(token_id) for token_id in observed_base_token_ids)
    protocol_catalog = infer_protocol_tokens(jsonl_path)
    normal_vocab_size = len(observed_base_token_ids)
    special_tokens = {
        entry["symbol"]: normal_vocab_size + offset
        for offset, entry in enumerate(protocol_catalog)
    }
    return {
        "vocab_size": normal_vocab_size + len(special_tokens),
        "normal_vocab_size": normal_vocab_size,
        "observed_base_token_ids": observed_base_token_ids,
        "special_tokens": special_tokens,
        "base_tokenizer": base_tokenizer,
        "protocol_catalog": protocol_catalog,
        "_source_hash": compute_file_sha256(jsonl_path),
    }


def write_vocab_info(vocab_path=None, jsonl_path=None):
    vocab_path = vocab_path or discover_vocab_path()
    vocab_info = build_dataset_vocab_info(jsonl_path=jsonl_path)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_info, f, ensure_ascii=False)
    return vocab_info


class DatasetBoundTokenizer:
    def __init__(self, vocab_info):
        self.vocab_info = vocab_info
        self.base_enc = tiktoken.get_encoding(vocab_info["base_tokenizer"])
        observed_ids = vocab_info["observed_base_token_ids"]
        self.local_to_base = [int(token_id) for token_id in observed_ids]
        self.base_to_local = {base_id: local_id for local_id, base_id in enumerate(self.local_to_base)}
        self.normal_vocab_size = len(self.local_to_base)
        self.special_tokens = {str(k): int(v) for k, v in vocab_info["special_tokens"].items()}
        self.special_tokens_inv = {token_id: symbol for symbol, token_id in self.special_tokens.items()}
        special_symbols = sorted(self.special_tokens.keys(), key=len, reverse=True)
        self._special_pattern = re.compile("|".join(map(re.escape, special_symbols))) if special_symbols else None

    def _encode_plain_text(self, text: str) -> List[int]:
        if not text:
            return []
        base_ids = self.base_enc.encode(text, allowed_special=set())
        return [self.base_to_local[token_id] for token_id in base_ids if token_id in self.base_to_local]

    def encode(self, text: str, allowed_special="all") -> List[int]:
        if not text:
            return []
        if allowed_special == "all":
            allowed_symbols = set(self.special_tokens.keys())
        elif isinstance(allowed_special, (set, list, tuple)):
            allowed_symbols = {str(symbol) for symbol in allowed_special}
        else:
            allowed_symbols = set()

        if not allowed_symbols or self._special_pattern is None:
            return self._encode_plain_text(text)

        token_ids = []
        cursor = 0
        for match in self._special_pattern.finditer(text):
            symbol = match.group(0)
            if symbol not in allowed_symbols:
                continue
            if match.start() > cursor:
                token_ids.extend(self._encode_plain_text(text[cursor:match.start()]))
            token_ids.append(self.special_tokens[symbol])
            cursor = match.end()
        if cursor < len(text):
            token_ids.extend(self._encode_plain_text(text[cursor:]))
        return token_ids

    def decode(self, token_ids, errors="replace") -> str:
        parts = []
        normal_base_ids = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id in self.special_tokens_inv:
                if normal_base_ids:
                    parts.append(self.base_enc.decode(normal_base_ids, errors=errors))
                    normal_base_ids = []
                parts.append(self.special_tokens_inv[token_id])
            elif 0 <= token_id < self.normal_vocab_size:
                normal_base_ids.append(self.local_to_base[token_id])
        if normal_base_ids:
            parts.append(self.base_enc.decode(normal_base_ids, errors=errors))
        return "".join(parts)

    def is_normal_token_id(self, token_id: int) -> bool:
        token_id = int(token_id)
        return 0 <= token_id < self.normal_vocab_size


def load_vocab_info():
    vocab_path = discover_vocab_path()
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return write_vocab_info(vocab_path=vocab_path)


def get_custom_tokenizer():
    vocab_info = load_vocab_info()
    special_tokens = {k: int(v) for k, v in vocab_info.get("special_tokens", {}).items()}
    enc = DatasetBoundTokenizer(vocab_info)
    return enc, special_tokens


def resolve_protocol_tokens(vocab_info):
    special_tokens = {k: int(v) for k, v in vocab_info.get("special_tokens", {}).items()}
    protocol = {
        "roles": {},
        "role_slots": [],
        "segments": {},
        "segment_slots": {},
    }
    for entry in vocab_info.get("protocol_catalog", []):
        symbol = entry["symbol"]
        if symbol not in special_tokens:
            continue
        topology_class = entry.get("topology_class")
        if topology_class not in (0, 1, 2, 3):
            continue
        payload = {
            "symbol": symbol,
            "id": special_tokens[symbol],
            "count": entry.get("count", 0),
            "slot": entry.get("slot", 0),
            "source": entry.get("source", ""),
            "boundary": entry.get("boundary", ""),
        }
        if topology_class == 0:
            protocol["pad"] = payload
        elif topology_class == 3:
            protocol["eos"] = payload
        elif topology_class == 1:
            protocol["roles"][payload["source"]] = payload
            protocol["role_slots"].append(payload)
        else:
            protocol["segments"][(payload["source"], payload["boundary"])] = payload
            slot_bucket = protocol["segment_slots"].setdefault(payload["slot"], {})
            slot_bucket[payload["boundary"]] = payload

    protocol["role_slots"].sort(key=lambda item: item["slot"])
    return protocol


def infer_num_heads(model_dim):
    divisors = [head for head in range(1, model_dim + 1) if model_dim % head == 0]
    target_head_dim = max(1, int(math.sqrt(model_dim)))
    return min(divisors, key=lambda head: (abs((model_dim // head) - target_head_dim), -head))


def load_checkpoint_payload(checkpoint_path=None):
    checkpoint_path = checkpoint_path or discover_model_path()
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def infer_model_config(vocab_size, checkpoint_path=None):
    checkpoint_payload = load_checkpoint_payload(checkpoint_path)
    if isinstance(checkpoint_payload, dict) and "config" in checkpoint_payload:
        cfg = checkpoint_payload["config"]
        return {
            "num_rules": int(cfg["num_rules"]),
            "embed_size": int(cfg["embed_size"]),
            "hidden_size": int(cfg["hidden_size"]),
            "num_layers": int(cfg["num_layers"]),
            "num_heads": int(cfg["num_heads"]),
            "expert_dim": int(cfg["expert_dim"]) if "expert_dim" in cfg else None,
        }

    return {}
