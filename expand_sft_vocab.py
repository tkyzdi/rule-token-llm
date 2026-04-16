import os
import json

from tokenizer_utils import compute_file_sha256, discover_dataset_path, resolve_project_path, write_vocab_info


def main():
    dataset_path = discover_dataset_path()
    if dataset_path is None:
        raise FileNotFoundError("未发现可用的 JSONL 数据集，无法构建训练数据词表。")
    vocab_path = resolve_project_path("distilled_vocab.json")
    dataset_hash = compute_file_sha256(dataset_path)

    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing_hash = existing.get("_source_hash", "")
        if existing_hash == dataset_hash:
            print(f"词表文件已存在且与当前训练数据匹配，跳过重建。(hash={dataset_hash[:16]})")
            return
        print(f"检测到训练数据或词表格式变更 ({existing_hash[:16]} -> {dataset_hash[:16]})，重建词表。")

    vocab_info = write_vocab_info(vocab_path=vocab_path, jsonl_path=dataset_path)
    print(f"底层分词器: {vocab_info['base_tokenizer']}")
    print(f"训练数据普通词元数: {vocab_info['normal_vocab_size']}")
    print(f"协议词元数: {len(vocab_info['special_tokens'])}")
    print(f"总词表大小: {vocab_info['vocab_size']}")
    print(f"协议符号: {list(vocab_info['special_tokens'].keys())}")

if __name__ == "__main__":
    main()
