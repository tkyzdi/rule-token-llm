# rule-token-llm

[English](README_EN.md) | [简体中文](README.md)

This project is an innovative Large Language Model (LLM) training and inference framework built from scratch. It abandons some of the inefficient designs of traditional Transformers and introduces **Rule Token**, **Non-Euclidean Spherical Attention**, and **Adaptive Mixture of Experts (Adaptive MoE)**, aiming to explore a more efficient and interpretable way of modeling high-dimensional semantic spaces.

---

## 🌟 Core Innovations

1. **Rule Token Causal Model (Endogenous Rule-Region Paradigm)**
   - Completely reconstructed the Token mapping mechanism, adopting the **Endogenous Rule-Region Paradigm**, which partitions the high-dimensional vocabulary space into "Gaussian manifolds" of rule regions.
   - Inference becomes a two-step hierarchical calculation: 1. Macro-routing to locate the rule region; 2. Micro-collapse to locate the specific Token. It eliminates external hardcoded mappings to fully embrace endogenous gravitational routing.
   - **Continuous Auto-Regressive Generation**: The model possesses autonomous termination capability by predicting a true `<EOS>`, completely eliminating hardcoded length limits and arbitrary punctuation-based stops.

2. **Non-Euclidean Spherical Attention & Top-Down Masking**
   - Discards traditional unbounded Dot-product Attention, adopting Cosine Gravity on the L2 hypersphere. Truth is directional, not numerical magnitude.
   - **Top-Down Attention Masking**: The Rules themselves decide the attention topology, forcing sparse and structurally relevant attention (e.g., Adjective to Noun) while masking out high-dimensional noise, significantly improving gradient fidelity.

3. **RulePagedExpertStore (Rule-based Paged MoE)**
   - Dynamically calculates Rule usage frequency, automatically loading and unloading expert networks (Expert FFNs), achieving extremely fine-grained computational resource scheduling.
   - Significantly improves model parameter capacity without increasing VRAM pressure.

4. **Fully Intelligent Driven Architecture**
   - It **dynamically adjusts negative sampling** and temperature coefficients based on the geometric capacity ($\sqrt{embed\_size}$) and current information entropy.
   - Dynamically adjusts learning rate and rule quantity based on gradient norm and information entropy. Built-in Log-space EMA Gradient Tracker prevents the "death spiral" during training.

---

## 📁 Project Structure

```text
📦 Open Source LLM
 ┣ 📜 build_real_dataset.py      # Pre-training corpus dataset building script
 ┣ 📜 build_sft_dataset.py       # Instruction fine-tuning (SFT) dataset building script
 ┣ 📜 expand_sft_vocab.py        # Vocabulary expansion tool for SFT data
 ┣ 📜 large_scale_trainer.py     # Large-scale distributed training core script (with adaptive state)
 ┣ 📜 large_scale_inference.py   # LLM inference engine (supports interactive & batch inference)
 ┣ 📜 rule_token_engine.py       # Core model architecture (Attention, FSQ, MoE Expert Networks)
 ┣ 📜 tokenizer_utils.py         # Vocabulary management, tokenization, and tensor utilities
```

---

## 🚀 Quick Start

### 1. Environment Preparation
Recommended to use Python 3.8+ and PyTorch 2.0+ environment. Install necessary dependencies:
```bash
pip install torch tiktoken
```

### 2. Data Preparation
First, you need to build tensor dataset chunks for training from the original text corpus:
```bash
# Build pre-training dataset
python build_real_dataset.py

# Build SFT dataset
python build_sft_dataset.py
```
*(The scripts will automatically generate `real_dataset_chunk_*.pt` and `sft_dataset_chunk_*.pt`)*

### 3. Model Training
Run the large-scale training script. The engine will automatically detect the GPU environment and start `torch.compile` acceleration:
```bash
python large_scale_trainer.py
```
During training, `GradientTracker` will be automatically applied to prevent gradient explosion, and the routing distribution of Rules will be dynamically adjusted.

### 4. Model Inference
After training is completed, or by loading existing weights, you can perform inference. Interactive terminal chat is supported:
```bash
python large_scale_inference.py --interactive
```
*Enter text at the prompt, and the model will perform autoregressive Token generation based on the non-Euclidean spherical gravity field.*

---

## 🧠 Architecture (First Principles)

### Why Spherical Attention?
In traditional Transformers, the dot product of high-dimensional vectors often leads to extreme values dominating the softmax distribution, masking long-tail semantic associations. This model projects Query and Key onto the L2 hypersphere and adjusts them via a dynamic temperature coefficient ($\tau = 1 / \log(\text{rank})$). This design guarantees that "direction is semantics", greatly improving the model's robustness to rare words and complex contexts.

### Rule-Aware Projection
In linear layer calculations, `RuleAwareProjection` is used. All Projections and Expert FFNs share the same `rule_ids`. By precomputing `RuleSortContext`, the $O(N \log N)$ redundant calculation is eliminated, maximizing the computational efficiency routed by Rule.

---

## 📄 License

This project is open-sourced under the **Apache License 2.0**.
