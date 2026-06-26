# verl框架8卡3090完整复现计划

> 基于实际硬件资源的可行方案，从零到一掌握verl框架

---

## 目录

1. [硬件资源评估](#1-硬件资源评估)
2. [复现阶段规划](#2-复现阶段规划)
3. [阶段一：环境搭建与SFT验证](#3-阶段一环境搭建与sft验证)
4. [阶段二：小模型RL训练](#4-阶段二小模型rl训练)
5. [阶段三：中等模型RL训练](#5-阶段三中等模型rl训练)
6. [阶段四：高级特性探索](#6-阶段四高级特性探索)
7. [阶段五：Agent训练实践](#7-阶段五agent训练实践)
8. [故障排除指南](#8-故障排除指南)

---

## 1. 硬件资源评估

### 1.1 8卡3090规格

| 参数 | 数值 |
|------|------|
| GPU数量 | 8张 |
| 单卡显存 | 24GB GDDR6X |
| 总显存 | 192GB |
| FP32算力 | 35.6 TFLOPS |
| FP16/BF16算力 | 71 TFLOPS |
| 互联方式 | PCIe（无NVLink） |
| 内存建议 | 256GB+ |
| 存储建议 | 1TB+ NVMe SSD |

### 1.2 显存预算分析

**模型显存需求估算（FP16训练）：**

| 模型大小 | 模型参数 | 优化器状态 | 梯度 | 激活值 | 总计 |
|----------|----------|------------|------|--------|------|
| 0.5B | 1GB | 4GB | 1GB | 2-4GB | 8-10GB |
| 1.5B | 3GB | 12GB | 3GB | 4-8GB | 22-26GB |
| 4B | 8GB | 32GB | 8GB | 8-16GB | 56-64GB |
| 7B | 14GB | 56GB | 14GB | 16-32GB | 100-116GB |

**结论：**
- **单卡可训练**: 0.5B模型
- **需要2-4卡**: 1.5B模型（FSDP）
- **需要8卡+优化**: 4B模型（FSDP + offload）
- **8卡极限**: 7B模型（FSDP + offload + 梯度检查点 + 小batch）

### 1.3 可行模型选择

**推荐模型列表：**

| 阶段 | 模型 | 用途 | 显存需求 |
|------|------|------|----------|
| 入门 | Qwen2.5-0.5B | SFT验证流程 | 8-10GB/卡 |
| 基础 | Qwen2.5-1.5B | RL入门 | 22-26GB/卡（4卡） |
| 进阶 | Qwen2.5-3B | RL深入 | 48-56GB/卡（8卡+offload） |
| 挑战 | Qwen2.5-7B | 极限测试 | 100GB+（8卡+全部优化） |

---

## 2. 复现阶段规划

### 2.1 总体时间线（建议4-6周）

```
Week 1: 环境搭建 + SFT验证
Week 2: 小模型RL（1.5B GRPO）
Week 3: 中等模型RL（3B GRPO + 性能优化）
Week 4: 高级特性（多轮、Agent）
Week 5-6: 实验对比 + 深入理解
```

### 2.2 每阶段目标

| 阶段 | 目标 | 交付物 |
|------|------|--------|
| 阶段一 | 环境可用，跑通SFT | 训练日志、loss曲线 |
| 阶段二 | 理解RL流程 | GRPO训练成功、reward上升 |
| 阶段三 | 掌握性能优化 | 3B模型训练成功 |
| 阶段四 | 理解高级特性 | 多轮对话、Agent训练 |
| 阶段五 | 深入理解 | 实验对比报告 |

---

## 3. 阶段一：环境搭建与SFT验证

### 3.1 环境安装

**Step 1: 基础环境**
```bash
# 创建conda环境
conda create -n verl python=3.10 -y
conda activate verl

# 安装PyTorch（CUDA 11.8版本适配3090）
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 验证CUDA
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

**Step 2: 安装verl**
```bash
# 克隆仓库
cd /data/home/yizhou
git clone https://github.com/verl-project/verl.git
cd verl

# 安装依赖
pip install -e .

# 或者使用requirements.txt
pip install -r requirements.txt
```

**Step 3: 安装vLLM（用于推理）**
```bash
# vLLM需要特定版本适配3090
pip install vllm==0.4.2

# 验证vLLM
python -c "import vllm; print(vllm.__version__)"
```

**Step 4: 准备数据**
```bash
# 创建数据目录
mkdir -p ~/data/gsm8k
mkdir -p ~/data/math

# 下载GSM8K数据（verl内置脚本）
python -m verl.utils.dataset.prepare_data --dataset gsm8k --output_dir ~/data/gsm8k

# 或者手动下载
# GSM8K: https://huggingface.co/datasets/openai/gsm8k
# MATH: https://huggingface.co/datasets/competition_math
```

**Step 5: 配置wandb（可选但推荐）**
```bash
pip install wandb
wandb login  # 输入你的API key
```

### 3.2 SFT验证实验

**目标**: 验证环境正确，跑通完整流程

**选择模型**: Qwen2.5-0.5B（最小，确保能跑通）

**运行SFT训练**:
```bash
# 进入SFT示例目录
cd /data/home/yizhou/verl/examples/sft/gsm8k

# 查看示例脚本
cat run_qwen2_5_0_5b_fsdp.sh

# 运行训练（单卡即可）
bash run_qwen2_5_0_5b_fsdp.sh
```

**如果脚本不存在，手动创建**:
```bash
#!/bin/bash
# sft_qwen_0.5b.sh

set -x

MODEL_PATH=Qwen/Qwen2.5-0.5B
TRAIN_FILE=~/data/gsm8k/train.parquet
TEST_FILE=~/data/gsm8k/test.parquet

python3 -m verl.trainer.main_sft \
    model.path=${MODEL_PATH} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.max_length=512 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=3 \
    trainer.save_freq=100 \
    trainer.logger='["console"]' \
    optim.lr=2e-5 \
    train_batch_size=32 \
    micro_batch_size_per_gpu=4
```

**验证指标**:
- [ ] 训练正常启动，无报错
- [ ] Loss正常下降
- [ ] GPU利用率>50%
- [ ] 显存使用合理（<20GB）

### 3.3 阶段一检查点

完成标志：
```bash
# 检查输出目录
ls -la outputs/

# 查看训练日志
cat outputs/*/train.log | tail -20

# 确认loss下降
grep "loss" outputs/*/train.log
```

---

## 4. 阶段二：小模型RL训练（1.5B GRPO）

### 4.1 资源规划

**模型**: Qwen2.5-1.5B
**GPU**: 4卡3090（留4卡给后续实验）
**算法**: GRPO（无需Critic，节省显存）

**显存估算**:
```
模型参数: 1.5B * 2 bytes = 3GB
优化器状态: 3GB * 4 = 12GB（Adam需要m和v）
梯度: 3GB
激活值: 4-8GB
Rollout (vLLM): 6-8GB
总计: 28-34GB/卡

4卡FSDP分片后: 7-8.5GB/卡 -> 可行
```

### 4.2 GRPO训练配置

**创建训练脚本**:
```bash
#!/bin/bash
# grpo_qwen_1.5b.sh

set -x

# 模型配置
MODEL_PATH=Qwen/Qwen2.5-1.5B

# 数据配置
TRAIN_FILE=~/data/gsm8k/train.parquet
TEST_FILE=~/data/gsm8k/test.parquet

# 训练超参
TRAIN_BATCH_SIZE=128          # 全局batch size
PPO_MINI_BATCH_SIZE=64        # mini batch size
MAX_PROMPT_LENGTH=256         # prompt最大长度
MAX_RESPONSE_LENGTH=512       # response最大长度
ROLLOUT_N=4                   # 每个prompt采样4个response

# 优化器
ACTOR_LR=1e-6
KL_LOSS_COEF=0.001

# Rollout配置
ROLLOUT_TP=1                  # 单卡TP（1.5B模型小）
ROLLOUT_GPU_MEM_UTIL=0.5      # vLLM显存使用比例

# 训练配置
TOTAL_EPOCHS=10
SAVE_FREQ=50
TEST_FREQ=10

# 启动训练
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=verl_grpo_1.5b \
    trainer.experiment_name=qwen2.5_1.5b_grpo_gsm8k
```

### 4.3 内存优化技巧

**如果显存不足，逐步应用以下优化**:

```bash
# 优化1: 启用CPU Offload
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
actor_rollout_ref.ref.fsdp_config.param_offload=True \

# 优化2: 减小batch size
data.train_batch_size=64 \
actor_rollout_ref.actor.ppo_mini_batch_size=32 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \

# 优化3: 减小max_token_len
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2046 \
actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=2046 \

# 优化4: 减小rollout显存
actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
```

### 4.4 训练监控

**关键指标**:
```bash
# 实时监控GPU使用
watch -n 1 nvidia-smi

# 监控训练指标（如果用wandb）
# 访问 wandb.ai 查看实时曲线
```

**正常训练的指标范围**:
- GPU利用率: 70-90%
- 显存使用: 18-22GB/卡
- Reward均值: 应该稳步上升
- KL散度: 保持在0.01-0.1之间
- Loss: 下降后趋于稳定

### 4.5 阶段二检查点

完成标志：
- [ ] GRPO训练成功启动
- [ ] Reward曲线呈上升趋势
- [ ] 生成的response质量可读
- [ ] 无OOM或训练崩溃

---

## 5. 阶段三：中等模型RL训练（3B GRPO + 性能优化）

### 5.1 资源规划

**模型**: Qwen2.5-3B
**GPU**: 8卡3090（全部使用）
**算法**: GRPO + 全部优化

**显存估算**:
```
模型参数: 3B * 2 bytes = 6GB
优化器状态: 6GB * 4 = 24GB
梯度: 6GB
激活值: 8-16GB
Rollout: 10-12GB
总计: 54-64GB/卡

8卡FSDP + offload后: 可行（需要精细调参）
```

### 5.2 完整优化配置

```bash
#!/bin/bash
# grpo_qwen_3b_optimized.sh

set -x

MODEL_PATH=Qwen/Qwen2.5-3B

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=~/data/gsm8k/train.parquet \
    data.val_files=~/data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_epochs=10 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=verl_grpo_3b \
    trainer.experiment_name=qwen2.5_3b_grpo_optimized
```

### 5.3 性能调优要点

**1. 序列打包（Remove Padding）**
```python
# 启用后可以显著提升吞吐量
actor_rollout_ref.model.use_remove_padding=True
```

**2. 动态Batch Size**
```python
# 根据token数动态调整batch大小
actor_rollout_ref.actor.use_dynamic_bsz=True
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
```

**3. 梯度检查点**
```python
# 用计算换内存
actor_rollout_ref.model.enable_gradient_checkpointing=True
```

**4. FSDP2（推荐）**
```python
# FSDP2比FSDP更高效
actor_rollout_ref.actor.strategy=fsdp2
actor_rollout_ref.ref.strategy=fsdp2
```

### 5.4 性能Benchmark

**预期性能（8卡3090）**:

| 指标 | 3B模型 | 7B模型（极限） |
|------|--------|----------------|
| 吞吐量 | 50-100 samples/min | 20-40 samples/min |
| 训练1000步 | 10-20小时 | 25-50小时 |
| 显存使用 | 20-22GB/卡 | 22-24GB/卡 |
| GPU利用率 | 75-85% | 80-90% |

### 5.5 阶段三检查点

完成标志：
- [ ] 3B模型训练成功
- [ ] 掌握内存优化技巧
- [ ] 能够调参平衡速度和显存
- [ ] 训练曲线正常

---

## 6. 阶段四：高级特性探索

### 6.1 PPO训练（对比GRPO）

**目标**: 理解PPO和GRPO的区别

**配置变化**:
```bash
# 主要变化：需要Critic模型
algorithm.adv_estimator=gae \
algorithm.gamma=1.0 \
algorithm.lam=0.95 \

# Critic配置
critic.model.path=Qwen/Qwen2.5-0.5B \
critic.optim.lr=1e-5 \
critic.ppo_mini_batch_size=128 \
critic.ppo_micro_batch_size_per_gpu=2 \
critic.fsdp_config.param_offload=True \
critic.fsdp_config.optimizer_offload=True \
```

**注意**: Critic模型需要额外显存，建议用0.5B的Critic配合3B的Actor

### 6.2 不同Advantage Estimator对比

**实验设计**:
```bash
# 实验1: GRPO
algorithm.adv_estimator=grpo

# 实验2: RLOO
algorithm.adv_estimator=rloo

# 实验3: REINFORCE++
algorithm.adv_estimator=reinforce_plus_plus

# 对比指标：
# 1. 收敛速度
# 2. 最终性能
# 3. 训练稳定性
# 4. 显存占用
```

### 6.3 KL散度控制实验

**实验设计**:
```bash
# 实验1: KL Loss（GRPO默认）
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.001

# 实验2: KL Penalty in Reward
algorithm.use_kl_in_reward=True
algorithm.kl_ctrl.kl_coef=0.01

# 实验3: 不同KL系数
actor_rollout_ref.actor.kl_loss_coef=0.0001  # 小KL
actor_rollout_ref.actor.kl_loss_coef=0.01    # 大KL

# 观察：
# 1. KL散度变化
# 2. Reward变化
# 3. 生成质量变化
```

### 6.4 LoRA训练（节省显存）

**配置**:
```bash
# 启用LoRA
actor_rollout_ref.model.lora_rank=16 \
actor_rollout_ref.model.lora_alpha=32 \
actor_rollout_ref.model.target_modules=all \

# LoRA可以：
# 1. 减少可训练参数
# 2. 降低显存占用
# 3. 加快训练速度
```

### 6.5 阶段四检查点

完成标志：
- [ ] 理解PPO和GRPO的区别
- [ ] 完成至少2组对比实验
- [ ] 理解KL控制的作用
- [ ] 尝试过LoRA训练

---

## 7. 阶段五：Agent训练实践

### 7.1 多轮对话基础

**概念理解**:
```python
# 单轮：prompt -> response
# 多轮：prompt -> response1 -> tool_call -> tool_result -> response2

# 关键技术：
# 1. Delta-based tokenization
# 2. 工具调用
# 3. 多轮reward分配
```

### 7.2 简单Agent训练

**目标**: 训练一个能调用计算器的数学Agent

**Step 1: 准备工具**
```python
# tools/calculator.py
from verl.tools.function_tool import function_tool

@function_tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.
    
    Args:
        expression: A mathematical expression, e.g. "(3+4)*5".
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"
```

**Step 2: 配置多轮训练**
```bash
# agent_training.sh

python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.function_tool_path=tools/calculator.py \
    actor_rollout_ref.rollout.name=sglang \
    ... # 其他配置同GRPO
```

### 7.3 Agent评估

**评估指标**:
```python
# 1. 任务完成率
# 2. 工具调用准确率
# 3. 平均对话轮数
# 4. 最终答案正确率
```

### 7.4 阶段五检查点

完成标志：
- [ ] 理解多轮对话的tokenization
- [ ] 成功配置工具调用
- [ ] 训练一个简单的Agent
- [ ] 评估Agent性能

---

## 8. 故障排除指南

### 8.1 常见问题及解决

**问题1: CUDA OOM**
```bash
# 症状：RuntimeError: CUDA out of memory

# 解决方案（按优先级）：
# 1. 减小batch size
data.train_batch_size=64
actor_rollout_ref.actor.ppo_mini_batch_size=32
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1

# 2. 启用offload
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

# 3. 减小max_token_len
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2046

# 4. 减小rollout显存
actor_rollout_ref.rollout.gpu_memory_utilization=0.4

# 5. 启用梯度检查点
actor_rollout_ref.model.enable_gradient_checkpointing=True
```

**问题2: 训练Loss为NaN**
```bash
# 症状：loss变成NaN或Inf

# 解决方案：
# 1. 降低学习率
actor_rollout_ref.actor.optim.lr=1e-7

# 2. 增加clip ratio
actor_rollout_ref.actor.clip_ratio=0.3

# 3. 检查数据
# 确保数据没有异常值

# 4. 启用混合精度
actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
```

**问题3: vLLM启动失败**
```bash
# 症状：vLLM初始化失败

# 解决方案：
# 1. 检查CUDA版本
nvcc --version

# 2. 降低gpu_memory_utilization
actor_rollout_ref.rollout.gpu_memory_utilization=0.4

# 3. 使用更小的tensor_parallel_size
actor_rollout_ref.rollout.tensor_model_parallel_size=1

# 4. 检查显存是否被占用
nvidia-smi
```

**问题4: 训练速度慢**
```bash
# 症状：每步训练时间过长

# 解决方案：
# 1. 启用序列打包
actor_rollout_ref.model.use_remove_padding=True

# 2. 启用动态batch
actor_rollout_ref.actor.use_dynamic_bsz=True

# 3. 增加micro_batch_size（如果显存允许）
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4

# 4. 检查数据加载
# 增加num_workers
```

**问题5: Reward不收敛**
```bash
# 症状：Reward不上升或波动大

# 解决方案：
# 1. 检查reward函数
# 打印几个样本的reward值

# 2. 增加KL惩罚
actor_rollout_ref.actor.kl_loss_coef=0.01

# 3. 减小学习率
actor_rollout_ref.actor.optim.lr=5e-7

# 4. 增加采样数
actor_rollout_ref.rollout.n=8
```

### 8.2 调试技巧

**1. 使用console logger**
```bash
trainer.logger='["console"]'  # 先不用wandb，方便调试
```

**2. 减小数据量快速验证**
```bash
data.train_files=~/data/gsm8k/train.parquet[:100]  # 只用100条数据
```

**3. 单步调试**
```python
# 在代码中加入断点
import pdb; pdb.set_trace()
```

**4. 监控显存**
```bash
# 实时监控
watch -n 0.5 nvidia-smi

# 或者用Python
import torch
print(torch.cuda.memory_summary())
```

### 8.3 性能优化检查清单

```bash
# 训练前检查
□ use_remove_padding=True（序列打包）
□ enable_gradient_checkpointing=True（梯度检查点）
□ use_dynamic_bsz=True（动态batch）
□ param_offload=True（如果显存紧张）
□ optimizer_offload=True（如果显存紧张）

# 训练中检查
□ GPU利用率 > 70%
□ 显存使用 < 22GB/卡
□ Loss正常下降
□ Reward稳步上升

# 训练后检查
□ Checkpoint正常保存
□ 评估指标合理
□ 生成质量可读
```

---

## 附录A：推荐学习路径

### A.1 代码阅读顺序

```
1. verl/protocol.py           - 理解DataProto
2. verl/trainer/ppo/core_algos.py - 理解RL算法
3. verl/trainer/ppo/ray_trainer.py - 理解训练流程
4. verl/workers/engine_workers.py - 理解Worker设计
5. verl/workers/rollout/      - 理解Rollout
6. verl/workers/reward_manager/ - 理解Reward
```

### A.2 文档阅读顺序

```
1. README.md                  - 项目概览
2. docs/hybrid_flow.rst       - 架构设计
3. docs/algo/grpo.md          - GRPO算法
4. docs/algo/ppo.md           - PPO算法
5. docs/perf/perf_tuning.rst  - 性能调优
6. docs/sglang_multiturn/multiturn.rst - 多轮对话
```

### A.3 实验顺序

```
Week 1: SFT (0.5B)
Week 2: GRPO (1.5B)
Week 3: GRPO (3B) + 性能优化
Week 4: PPO vs GRPO对比
Week 5: 不同超参实验
Week 6: Agent训练
```

---

## 附录B：关键配置参数速查

| 参数 | 默认值 | 说明 | 建议值 |
|------|--------|------|--------|
| `data.train_batch_size` | - | 全局batch size | 64-256 |
| `actor_rollout_ref.rollout.n` | 1 | 每prompt采样数 | 4-8 |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | - | mini batch size | 32-128 |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | - | 每卡micro batch | 1-4 |
| `actor_rollout_ref.actor.optim.lr` | 1e-6 | 学习率 | 5e-7 ~ 2e-6 |
| `actor_rollout_ref.actor.clip_ratio` | 0.2 | PPO clip范围 | 0.2-0.3 |
| `actor_rollout_ref.actor.kl_loss_coef` | 0.001 | KL损失系数 | 0.0001-0.01 |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | 0.5 | vLLM显存比例 | 0.4-0.6 |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | 1 | TP大小 | 1-2 |
| `data.max_prompt_length` | 1024 | prompt最大长度 | 256-1024 |
| `data.max_response_length` | 2048 | response最大长度 | 512-2048 |

---

## 附录C：监控指标说明

| 指标 | 含义 | 正常范围 |
|------|------|----------|
| `actor/reward` | 平均reward | 应该上升 |
| `actor/kl_divergence` | KL散度 | 0.01-0.1 |
| `actor/policy_loss` | 策略损失 | 应该下降 |
| `actor/entropy` | 策略熵 | 不应该太低 |
| `actor/clip_fraction` | 被clip比例 | 0.1-0.3 |
| `actor/grad_norm` | 梯度范数 | < 1.0 |
| `rollout/throughput` | 生成吞吐量 | 越高越好 |
| `gpu_utilization` | GPU利用率 | > 70% |

---

*最后更新: 2026年6月*
*本文档为8卡3090环境下的verl框架复现计划*
