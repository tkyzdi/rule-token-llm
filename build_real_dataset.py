import json
import os

import torch

from tokenizer_utils import (
    discover_dataset_path,
    extract_record_text,
    get_custom_tokenizer,
    get_logger,
    infer_dataset_geometry_shared,
    load_vocab_info,
    resolve_project_path,
    resolve_protocol_tokens,
    sanitize_training_text,
    sliding_window_chunks,
    validate_sequence,
)

logger = get_logger()


def _extract_real_text(raw_line):
    try:
        data = json.loads(raw_line)
    except json.JSONDecodeError:
        return sanitize_training_text(raw_line)
    return extract_record_text(data)


def build_real_dataset(jsonl_path=None, max_seq_len=None):
    vocab_info = load_vocab_info()
    enc, _ = get_custom_tokenizer()
    jsonl_path = jsonl_path or discover_dataset_path()
    if jsonl_path is None:
        raise FileNotFoundError("未发现可用的 JSONL 数据集。")
    inferred_seq_len, chunk_token_budget = infer_dataset_geometry_shared(
        jsonl_path, enc, vocab_info["vocab_size"], _extract_real_text,
    )
    max_seq_len = max_seq_len or inferred_seq_len
    checkpoint_file = resolve_project_path("real_checkpoint.json")
    
    protocol_tokens = resolve_protocol_tokens(vocab_info)
    eos_id = protocol_tokens["eos"]["id"]

    dataset = []
    buffered_tokens = 0
    line_count = 0
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

    logger.info("开始解析 %s 提取无监督预训练序列...", jsonl_path)
    logger.info("自适应序列长度: %d | 自适应分块 Token 预算: %d", max_seq_len, chunk_token_budget)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            if line_count < start_line:
                line_count += 1
                continue

            line_count += 1
            line = raw_line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                data = line
            text = extract_record_text(data)
            if not text:
                continue

            token_ids = enc.encode(text, allowed_special="all")
            token_ids.append(eos_id)

            for chunk in sliding_window_chunks(token_ids, max_seq_len):
                seq_tensor = torch.tensor(chunk, dtype=torch.int32)
                if not validate_sequence(seq_tensor, vocab_info["vocab_size"]):
                    continue
                dataset.append({"sequence": seq_tensor})
                buffered_tokens += len(chunk)
                total_samples += 1

            if buffered_tokens >= chunk_token_budget and dataset:
                chunk_path = resolve_project_path(f"real_dataset_chunk_{chunk_index}.pt")
                torch.save(dataset, chunk_path)
                logger.info("已保存无监督分块: %s (样本数: %d)", chunk_path, len(dataset))
                dataset = []
                buffered_tokens = 0
                chunk_index += 1
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "dataset": jsonl_path,
                        "line_count": line_count,
                        "chunk_index": chunk_index,
                        "total_samples": total_samples
                    }, f, ensure_ascii=False)

    if dataset:
        chunk_path = resolve_project_path(f"real_dataset_chunk_{chunk_index}.pt")
        torch.save(dataset, chunk_path)
        logger.info("已保存无监督分块: %s (样本数: %d)", chunk_path, len(dataset))

    logger.info("JSONL 序列提取完毕，共 %d 条。", total_samples)


if __name__ == "__main__":
    build_real_dataset()
