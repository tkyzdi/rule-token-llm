# rule-token-llm

[English](README_EN.md) | [简体中文](README.md)

本项目是一个从零实现的规则驱动语言模型训练与推理框架。它不是标准的“词表 softmax + 常规 Transformer”实现，而是围绕 **Rule Token 分层生成**、**规则感知注意力/投影**、**分页式专家存储** 与 **数据集绑定词表** 展开，目标是在同一套代码中打通数据构建、训练、推理与本地评测。


---

## 核心能力

1. **Rule Token 分层生成**
   - `RuleTokenCausalModel` 先预测规则分布，再在该规则对应的候选 Token 集合内完成局部读出。
   - 规则划分由 `FiniteScalarQuantizer` 与 `compute_rule_vocab_mask()` 在词嵌入空间上建立，不是固定人工表。
   - 推理时会优先在已训练过的活跃 Token 上采样，并支持预测 `<EOS>` 结束；同时仍保留 `max_new_tokens` 上限，避免无界生成。

2. **规则感知注意力与投影**
   - `RuleAwareProjection` 为 Q/K/V/O 投影提供按规则条件化的 Kronecker 适配器。
   - 注意力主体仍使用 PyTorch 的 causal SDPA，但会叠加基于规则相似度的可微 bias，而不是硬性的离散 mask。
   - 额外实现了分层 rule-level attention，对前缀中各规则的聚合状态做因果汇总。

3. **分页式 Rule MoE**
   - `RulePagedExpertStore` 将专家参数常驻 CPU，需要时按活跃规则分页搬运到运行设备。
   - 训练与推理都会基于当前活跃规则构建 runtime pages，并按使用能量进行缓存/淘汰。
   - 这套机制的目标是降低显存压力，而不是宣称“无代价扩容”。

4. **自适应训练流程**
   - 训练脚本会根据词表规模、序列长度与设备容量自动推断模型初始配置、批大小、梯度累积和训练窗口。
   - 训练中实现了动态负样本采样、梯度 Log-space EMA 跟踪、学习率阻尼、梯度裁剪，以及规则数自适应收缩。
   - 当规则数收缩时，使用基于 Sinkhorn 最优传输的参数合并，而不是简单的硬匹配平均。

5. **数据集绑定词表与协议符号**
   - `tokenizer_utils.py` 会从项目目录下自动发现 JSONL 数据，推断记录结构、对话角色、辅助段落与协议符号。
   - `expand_sft_vocab.py` 实际作用是构建或刷新 `distilled_vocab.json`，并不是只针对 SFT 扩词。
   - `build_real_dataset.py` 与 `build_sft_dataset.py` 都依赖这套词表与协议信息来生成训练分块。

6. **本地评测链路**
   - `evaluate_local_benchmark.py` 可以从本地 JSONL 构建 benchmark，执行推理，并输出按任务/领域切分的评测报告。
   - 这是仓库当前真实存在的脚本能力，原 README 未覆盖。

---

## 项目结构

```text
📦 rule-token-llm
 ┣ 📜 build_real_dataset.py       # 从本地 JSONL 构建无监督预训练分块
 ┣ 📜 build_sft_dataset.py        # 从本地 JSONL 构建 SFT 分块（含角色/协议符号）
 ┣ 📜 expand_sft_vocab.py         # 构建或刷新数据集绑定词表 distilled_vocab.json
 ┣ 📜 large_scale_trainer.py      # 训练入口：自适应调度、规则收缩、检查点保存
 ┣ 📜 large_scale_inference.py    # 推理入口：交互式推理或基于数据分块的批量回放
 ┣ 📜 evaluate_local_benchmark.py # 本地 benchmark 构建与评测
 ┣ 📜 rule_token_engine.py        # 核心模型：Rule Attention、FSQ、HRR、Paged Expert
 ┣ 📜 tokenizer_utils.py          # 数据结构推断、词表构建、协议符号与 tokenizer 工具
```

---

## 快速开始

### 1. 环境准备

建议使用 Python 3.8+ 与 PyTorch 2.x：

```bash
pip install torch tiktoken
```

### 2. 放置数据

将训练用 `.jsonl` 数据放在项目根目录。脚本会自动选择根目录下匹配到的本地数据文件，并据此推断：

- 词表与协议符号
- 记录 schema / message schema
- 预训练与 SFT 的序列长度和分块预算

如果你想显式先构建词表，可执行：

```bash
python expand_sft_vocab.py
```

### 3. 构建训练分块

```bash
python build_real_dataset.py
python build_sft_dataset.py
```

脚本会在项目目录生成：

- `real_dataset_chunk_*.pt`
- `sft_dataset_chunk_*.pt`

### 4. 模型训练

```bash
python large_scale_trainer.py
```

说明：

- 若存在已有 `.pth` 检查点，会自动尝试恢复模型、专家存储与优化器状态。
- 若当前环境支持 `torch.compile`，训练脚本会自动启用；否则回退到 eager 模式。
- 训练会按是否存在 `real_dataset_chunk_*.pt` / `sft_dataset_chunk_*.pt` 自动决定阶段顺序。

### 5. 模型推理

交互式推理：

```bash
python large_scale_inference.py --interactive
```

非交互模式下，脚本会从已有数据分块中抽取若干前缀语境进行批量回放：

```bash
python large_scale_inference.py
```

### 6. 本地评测

```bash
python evaluate_local_benchmark.py --dataset your_eval.jsonl --split test
```

该脚本会生成 benchmark、预测结果与汇总报告 JSON 文件。

---

## 架构说明

### 1. 规则是如何得到的？

- 模型不直接把“规则”写死在词表映射里。
- `FiniteScalarQuantizer` 会把嵌入投影到有限离散坐标系，再得到 `rule_id`。
- `compute_rule_vocab_mask()` 再把整个词表映射为 `token -> rule` 与 `rule -> candidate tokens`。


### 2. 注意力到底做了什么？

- 代码并没有实现“硬 top-down mask 决定一切”的版本。
- 实际实现是：标准因果注意力 + 规则相似度 bias + rule-level 前缀汇总路径。
- 这意味着规则信息会改变注意力分布，但不会把注意力退化为完全手工指定的拓扑。

### 3. 为什么需要分页专家？

- 每条序列不会同时激活全部规则。
- 因此把所有专家常驻 GPU 并不经济。
- `RulePagedExpertStore` 只把当前活跃规则的专家页搬到设备端，训练后再把梯度散回 CPU 主存中的专家参数。

### 4. Truth Field 是什么？

- 模型最终输出两部分：规则 logits 与 `truth_field_state`。
- `truth_field_state` 由线性语义场和 `HolographicRuleBinding` 叠加组成。
- 推理时，模型在选中的规则候选 Token 集上计算与该 field 的相似度，再完成最终采样。

---


## 开源协议

本项目采用 **Apache License 2.0** 协议开源。
