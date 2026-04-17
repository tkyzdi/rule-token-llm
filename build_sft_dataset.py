import json
import os
import re

import torch

from tokenizer_utils import (
    discover_dataset_path,
    get_custom_tokenizer,
    get_logger,
    infer_dataset_geometry_shared,
    infer_dataset_schema,
    load_vocab_info,
    normalize_schema_messages,
    resolve_project_path,
    resolve_protocol_tokens,
    sanitize_training_text,
    sliding_window_chunks,
    validate_sequence,
)

logger = get_logger()

def inject_protocol_segments(content, protocol_tokens):
    if not hasattr(inject_protocol_segments, "_pattern"):
        replacements = {}
        ranked_segments = sorted(protocol_tokens["segments"].items(), key=lambda item: (-len(item[0][0]), item[0][1], item[0][0]))
        for (source, boundary), payload in ranked_segments:
            if source == "__auxiliary__":
                continue
            literal = f"</{source}>" if boundary == "close" else f"<{source}>"
            replacements[literal] = payload["symbol"]
            replacements[literal.upper()] = payload["symbol"]
        
        # Escape keys and build a single regex
        if replacements:
            pattern = re.compile("|".join(map(re.escape, replacements.keys())))
        else:
            pattern = None
        inject_protocol_segments._pattern = pattern
        inject_protocol_segments._replacements = replacements

    if inject_protocol_segments._pattern is None:
        return content
    return inject_protocol_segments._pattern.sub(lambda m: inject_protocol_segments._replacements[m.group(0)], content)


def role_slot_for_message(role, index_in_record, total_messages, protocol_tokens):
    if role in protocol_tokens["roles"]:
        return protocol_tokens["roles"][role]
    if not protocol_tokens["role_slots"]:
        return None
    relative_position = index_in_record / max(total_messages - 1, 1)
    slot_index = round(relative_position * (len(protocol_tokens["role_slots"]) - 1))
    return protocol_tokens["role_slots"][slot_index]


def auxiliary_segment_pair(protocol_tokens):
    if ("__auxiliary__", "open") in protocol_tokens["segments"] and ("__auxiliary__", "close") in protocol_tokens["segments"]:
        return (
            protocol_tokens["segments"][("__auxiliary__", "open")],
            protocol_tokens["segments"][("__auxiliary__", "close")]
        )
    return None


def _make_sft_extractor(schema):
    def extract_fn(raw_line):
        try:
            data = json.loads(raw_line)
        except json.JSONDecodeError:
            return None
        messages = normalize_schema_messages(data, schema)
        parts = []
        for msg in messages:
            cleaned_content = sanitize_training_text(msg.get("content", ""))
            if cleaned_content:
                parts.append(cleaned_content)
            if msg.get("auxiliary_content", ""):
                cleaned_aux = sanitize_training_text(msg["auxiliary_content"])
                if cleaned_aux:
                    parts.append(cleaned_aux)
        joined = "\n".join(parts)
        return joined if joined.strip() else None
    return extract_fn


def main():
    vocab_info = load_vocab_info()
    enc, _ = get_custom_tokenizer()
    protocol_tokens = resolve_protocol_tokens(vocab_info)
    eos_id = protocol_tokens["eos"]["id"]

    jsonl_path = discover_dataset_path()
    if jsonl_path is None:
        raise FileNotFoundError("未发现可用的 JSONL 数据集。")

    schema = infer_dataset_schema(jsonl_path)
    max_seq_len, chunk_token_budget = infer_dataset_geometry_shared(
        jsonl_path, enc, vocab_info["vocab_size"], _make_sft_extractor(schema)
    )
    checkpoint_file = resolve_project_path("sft_checkpoint.json")
    dataset = []
    buffered_tokens = 0
    start_line = 0
    chunk_index = 0
    total_samples = 0

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        if ckpt.get("dataset") == jsonl_path:
            start_line = ckpt.get("line_count", 0)
            chunk_index = ckpt.get("chunk_index", 0)
            total_samples = ckpt.get("total_samples", 0)
            logger.info("检测到断点记录，将从第 %d 行继续处理。当前已有 chunk: %d，累计样本: %d", start_line, chunk_index, total_samples)

    logger.info("开始构建大一统 SFT 数据集...")
    logger.info("自适应序列长度: %d | 自适应分块 Token 预算: %d", max_seq_len, chunk_token_budget)

    line_count = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            if line_count < start_line:
                line_count += 1
                continue

            line = raw_line.strip()
            if not line:
                line_count += 1
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                line_count += 1
                continue

            messages = normalize_schema_messages(data, schema)
            sequence = []
            aux_pair = auxiliary_segment_pair(protocol_tokens)
            for message_index, message in enumerate(messages):
                role = message.get("role", "")
                content = sanitize_training_text(message.get("content", ""))
                content = inject_protocol_segments(content, protocol_tokens)
                role_token = role_slot_for_message(role, message_index, len(messages), protocol_tokens)
                if role_token is not None:
                    sequence.append(role_token["id"])
                token_ids = enc.encode(content, allowed_special="all")
                sequence.extend(token_ids)
                auxiliary_content = sanitize_training_text(message.get("auxiliary_content", ""))
                if auxiliary_content and aux_pair is not None:
                    aux_open, aux_close = aux_pair
                    sequence.append(aux_open["id"])
                    sequence.extend(enc.encode(auxiliary_content, allowed_special="all"))
                    sequence.append(aux_close["id"])
                if message_index == len(messages) - 1:
                    sequence.append(eos_id)

            if sequence:
                for chunk in sliding_window_chunks(sequence, max_seq_len):
                    seq_tensor = torch.tensor(chunk, dtype=torch.int32)
                    if not validate_sequence(seq_tensor, vocab_info["vocab_size"]):
                        continue
                    dataset.append({"sequence": seq_tensor})
                    buffered_tokens += len(chunk)
                    total_samples += 1

            if buffered_tokens >= chunk_token_budget and dataset:
                chunk_path = resolve_project_path(f"sft_dataset_chunk_{chunk_index}.pt")
                torch.save(dataset, chunk_path)
                logger.info("已保存分块: %s (样本数: %d)", chunk_path, len(dataset))
                dataset = []
                buffered_tokens = 0
                chunk_index += 1
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "dataset": jsonl_path,
                        "line_count": line_count + 1,
                        "chunk_index": chunk_index,
                        "total_samples": total_samples
                    }, f, ensure_ascii=False)

            line_count += 1

    if dataset:
        chunk_path = resolve_project_path(f"sft_dataset_chunk_{chunk_index}.pt")
        torch.save(dataset, chunk_path)
        logger.info("已保存分块: %s (样本数: %d)", chunk_path, len(dataset))

    logger.info("SFT 数据集构建完成，共 %d 条样本！", total_samples)


if __name__ == "__main__":
    main()
