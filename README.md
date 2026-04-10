# rule-token—llm

本项目是一个基于从零构建的创新型大语言模型（LLM）训练与推理框架。摒弃了传统 Transformer 的部分低效设计，引入了 **Rule Token（规则令牌）**、**非欧几何球面注意力（Non-Euclidean Spherical Attention）** 以及 **自适应混合专家系统（Adaptive MoE）**，旨在探索更高效、更具解释性的高维语义空间建模方式。

---

## 🌟 核心创新特性

1. **Rule Token Causal Model (规则令牌因果模型)**
   - 彻底重构了 Token 映射机制，使用连续的“规则”引导生成，取代传统离散的 Token 预测。
   - 结合有限标量量化（FSQ, Finite Scalar Quantizer）技术，实现语义空间的自适应切分。

2. **Non-Euclidean Spherical Attention (非欧球面注意力机制)**
   - 废弃了传统无界的点乘注意力（Dot-product Attention），采用 L2 超球面上的余弦引力（Cosine Gravity）。
   - 真理是方向性的，而非数值大小。这有效防止了高维空间内的内积坍缩（Curse of Dimensionality），并在稀疏化掩码下保证严格的因果律。

3. **RulePagedExpertStore (基于规则的分页专家存储 MoE)**
   - 动态计算 Rule 的使用频率，自动加载和卸载专家网络（Expert FFNs），实现极细粒度的计算资源调度。
   - 在不增加显存压力的前提下，显著提升模型参数容量。

4. **自适应训练状态管理 (Adaptive Rule State)**
   - 根据梯度范数和信息熵动态调整学习率和规则数量，内置 Log-space EMA 梯度追踪器，防止训练过程中的“死亡螺旋”。

---

## 📁 项目结构

```text
📦 开源大模型
 ┣ 📜 build_real_dataset.py      # 构建预训练语料数据集脚本
 ┣ 📜 build_sft_dataset.py       # 构建指令微调 (SFT) 数据集脚本
 ┣ 📜 expand_sft_vocab.py        # 针对 SFT 数据扩展词表工具
 ┣ 📜 large_scale_trainer.py     # 大规模分布式训练核心脚本 (含自适应状态管理)
 ┣ 📜 large_scale_inference.py   # 大模型推理引擎 (支持交互式与批量推理)
 ┣ 📜 rule_token_engine.py       # 核心模型架构 (Attention, FSQ, MoE 专家网络)
 ┣ 📜 tokenizer_utils.py         # 词表管理、切词与张量运算工具库
```

---

## 🚀 快速开始

### 1. 环境准备
建议使用 Python 3.8+ 及 PyTorch 2.0+ 环境。安装必要的依赖：
```bash
pip install torch tiktoken
```

### 2. 数据准备
首先，你需要从原始文本库中构建用于训练的张量数据集分块：
```bash
# 构建预训练数据集
python build_real_dataset.py

# 构建 SFT 指令微调数据集
python build_sft_dataset.py
```
*(脚本会自动生成 `real_dataset_chunk_*.pt` 和 `sft_dataset_chunk_*.pt`)*

### 3. 模型训练
运行大规模训练脚本，引擎会自动检测 GPU 环境并启动 `torch.compile` 编译加速：
```bash
python large_scale_trainer.py
```
训练过程中会自动应用 `GradientTracker` 防止梯度爆炸，并动态调整 Rule 的路由分布。

### 4. 模型推理
训练完成后，或加载已有权重进行推理。支持终端交互式对话：
```bash
python large_scale_inference.py --interactive
```
*在提示符下输入文本，模型将基于非欧球面引力场进行自回归 Token 生成。*

---

## 🧠 架构解析 (First Principles)

### 为什么使用 Spherical Attention？
在传统 Transformer 中，高维向量的点乘容易导致极值主导 softmax 分布，掩盖了长尾的语义关联。本模型将 Query 和 Key 投影至 L2 超球面，并通过动态温度系数（$\tau = 1 / \log(\text{rank})$）调节。这种设计保证了“方向即语义”，极大提升了模型对罕见词和复杂语境的鲁棒性。

### Rule-Aware Projection
在线性层计算中，采用 `RuleAwareProjection`。所有的 Projection 和 Expert FFN 共享相同的 `rule_ids`，通过预计算 `RuleSortContext` 消除 $O(N \log N)$ 的冗余计算，将按 Rule 路由的计算效率提升至极致。

---

## 📄 开源协议

本项目采用 **Apache License 2.0** 协议开源。
