import json
import math
import os
import re

import tiktoken
import torch

from tokenizer_utils import discover_base_tokenizer, discover_dataset_path, infer_dataset_schema, normalize_schema_messages, resolve_project_path


TAG_PATTERN = re.compile(r"<(/?)([A-Za-z0-9_:\-]+)>")


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
            item[0]
        )
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
            item[0]
        )
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
        # 智能驱动：PAD 数量应当由系统的对齐颗粒度（通常与序列方差相关）自然推演，而非人工指定为 1
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


def main():
    dataset_path = discover_dataset_path()
    vocab_path = resolve_project_path("distilled_vocab.json")

    if os.path.exists(vocab_path):
        import hashlib
        file_hash = hashlib.sha256(dataset_path.encode() if dataset_path else b"none").hexdigest()[:16]
        with open(vocab_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing_hash = existing.get("_source_hash", "")
        if existing_hash == file_hash:
            print(f"词表文件已存在且与当前数据集匹配，跳过重建。(hash={file_hash})")
            return
        print(f"检测到数据集变更 ({existing_hash} -> {file_hash})，重建词表。")

    base_tokenizer = discover_base_tokenizer(dataset_path)
    base_enc = tiktoken.get_encoding(base_tokenizer)
    base_vocab_size = base_enc.max_token_value + 1
    protocol_catalog = infer_protocol_tokens(dataset_path)
    special_tokens = {
        entry["symbol"]: base_vocab_size + offset
        for offset, entry in enumerate(protocol_catalog)
    }

    import hashlib
    source_hash = hashlib.sha256(dataset_path.encode() if dataset_path else b"none").hexdigest()[:16]

    vocab_info = {
        "vocab_size": base_vocab_size + len(special_tokens),
        "special_tokens": special_tokens,
        "base_tokenizer": base_tokenizer,
        "protocol_catalog": protocol_catalog,
        "_source_hash": source_hash,
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_info, f, ensure_ascii=False)

    print(f"底层引擎: {base_tokenizer}")
    print(f"总词表大小: {vocab_info['vocab_size']}")
    print(f"自适应协议符号: {list(special_tokens.keys())}")

if __name__ == "__main__":
    main()
