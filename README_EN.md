# rule-token-llm

[English](README_EN.md) | [简体中文](README.md)

This project is a from-scratch language-model training and inference framework centered on rule-conditioned generation. Instead of a standard "flat vocabulary softmax + vanilla Transformer" stack, the current codebase combines **Rule Token hierarchical decoding**, **rule-aware attention/projection**, **paged expert storage**, and a **dataset-bound tokenizer/vocabulary pipeline**.


---

## Core Capabilities

1. **Rule Token Hierarchical Generation**
   - `RuleTokenCausalModel` first predicts a rule distribution, then performs token readout inside the candidate token set attached to that rule.
   - Rule partitions are built from embedding-space quantization via `FiniteScalarQuantizer` and `compute_rule_vocab_mask()`, not from a manually fixed mapping.
   - Inference can terminate by predicting `<EOS>`, but generation still keeps a `max_new_tokens` cap to avoid unbounded decoding.

2. **Rule-Aware Attention and Projection**
   - `RuleAwareProjection` provides rule-conditioned Kronecker adapters for Q/K/V/O projections.
   - The attention core still uses causal SDPA, but augments it with a differentiable rule-similarity bias rather than a hard discrete mask.
   - The model also includes a hierarchical rule-level attention path that aggregates prefix state per active rule.

3. **Paged Rule MoE**
   - `RulePagedExpertStore` keeps expert parameters on CPU and pages only active-rule experts onto the runtime device.
   - Both training and inference construct runtime pages from currently active rules and maintain a cache/eviction policy driven by rule activity.
   - The real claim here is reduced VRAM pressure, not "free" scaling.

4. **Adaptive Training Pipeline**
   - The trainer infers initial model shape, batch size, gradient accumulation, and training windows from vocabulary size, sequence geometry, and available device memory.
   - Training implements dynamic negative sampling, log-space EMA gradient tracking, LR damping, gradient clipping, and adaptive rule-count shrinkage.
   - When rule count is reduced, parameters are merged with a Sinkhorn optimal-transport plan instead of naive hard reassignment.

5. **Dataset-Bound Vocabulary and Protocol Tokens**
   - `tokenizer_utils.py` auto-discovers local JSONL data, infers record/message schema, and derives role / segment / EOS protocol tokens.
   - `expand_sft_vocab.py` actually builds or refreshes `distilled_vocab.json`; it is not only an SFT-specific vocab expander.
   - Both dataset-building scripts depend on this inferred vocabulary/protocol layer.

6. **Local Evaluation Pipeline**
   - `evaluate_local_benchmark.py` builds a benchmark from local JSONL data, runs inference, and emits sliced evaluation reports.
   - This is a real repository feature that was missing from the original README.

---

## Project Structure

```text
📦 rule-token-llm
 ┣ 📜 build_real_dataset.py       # Build unsupervised pretraining chunks from local JSONL
 ┣ 📜 build_sft_dataset.py        # Build SFT chunks with role/protocol symbols
 ┣ 📜 expand_sft_vocab.py         # Build or refresh dataset-bound vocabulary: distilled_vocab.json
 ┣ 📜 large_scale_trainer.py      # Training entry: adaptive scheduling, rule shrinkage, checkpoint save/load
 ┣ 📜 large_scale_inference.py    # Inference entry: interactive mode or dataset-chunk playback
 ┣ 📜 evaluate_local_benchmark.py # Local benchmark construction and evaluation
 ┣ 📜 rule_token_engine.py        # Core model: rule attention, FSQ, HRR, paged experts
 ┣ 📜 tokenizer_utils.py          # Schema inference, vocabulary building, protocol tokens, tokenizer utilities
```

---

## Quick Start

### 1. Environment

Recommended: Python 3.8+ and PyTorch 2.x

```bash
pip install torch tiktoken
```

### 2. Put Your Data in the Project Root

Place your training `.jsonl` file in the project root. The scripts will automatically inspect local data and infer:

- vocabulary and protocol tokens
- record / message schema
- sequence length and chunk budget for dataset building

If you want to explicitly build the vocabulary first, run:

```bash
python expand_sft_vocab.py
```

### 3. Build Training Chunks

```bash
python build_real_dataset.py
python build_sft_dataset.py
```

This generates:

- `real_dataset_chunk_*.pt`
- `sft_dataset_chunk_*.pt`

### 4. Train

```bash
python large_scale_trainer.py
```

Notes:

- If a `.pth` checkpoint already exists, the trainer will try to restore model, expert-store, and optimizer state.
- If `torch.compile` is supported by the environment, it is enabled automatically; otherwise the trainer falls back to eager mode.
- Training stages are inferred from the presence of `real_dataset_chunk_*.pt` and `sft_dataset_chunk_*.pt`.

### 5. Inference

Interactive inference:

```bash
python large_scale_inference.py --interactive
```

Batch-style playback from existing dataset chunks:

```bash
python large_scale_inference.py
```

### 6. Local Evaluation

```bash
python evaluate_local_benchmark.py --dataset your_eval.jsonl --split test
```

The script writes benchmark, prediction, and summary/report JSON outputs.

---

## Architecture Notes

### 1. How are rules obtained?

- Rules are not hardcoded into a fixed token mapping.
- `FiniteScalarQuantizer` projects embeddings into a finite discrete coordinate system and derives `rule_id`.
- `compute_rule_vocab_mask()` then builds both `token -> rule` and `rule -> candidate tokens`.


### 2. What does attention actually do?

- The code does not implement a purely hard "top-down mask decides everything" mechanism.
- The implemented path is: standard causal attention + rule-similarity bias + rule-level prefix aggregation.
- So rules influence the attention landscape, but do not replace it with a manually fixed topology.

### 3. Why paged experts?

- A sequence only activates a small subset of rules at any moment.
- Keeping all experts resident on GPU is therefore wasteful.
- `RulePagedExpertStore` pages only the currently active-rule experts to the device and scatters gradients back to the CPU-side store after training steps.

### 4. What is the Truth Field?

- The model outputs both rule logits and a `truth_field_state`.
- `truth_field_state` is the sum of a linear semantic field and a `HolographicRuleBinding` path.
- During inference, the final token is sampled by comparing this field against candidate tokens inside the selected rule.

---


## License

This project is open-sourced under the **Apache License 2.0**.
