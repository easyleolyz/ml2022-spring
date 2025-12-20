# HW01 Regression (COVID-19 Cases Prediction)

## 1. 任务说明
- 任务类型：回归（Regression）
- 目标：根据给定特征预测未来的确诊相关数值（Kaggle 比赛：ML2022Spring-hw1）
- 评估指标：MSE（Mean Squared Error）

数据文件：
- `covid.train.csv`
- `covid.test.csv`

## 2. 数据列与字段约定（非常重要）
- `id`：样本编号（train/test 均存在）
- 训练标签列（train-only）：`tested_positive.4`
  - 通过 `train.columns - test.columns` 的差集推断得到
- Kaggle 提交要求的预测列名：`tested_positive`
  - 注意：**提交列名必须是 `tested_positive`，否则 Kaggle 会直接报错（status=error）**

特征列选择策略：
- 以 test 的列为准：`feature_cols = test.columns - {id}`
- train 端也只取同一组 `feature_cols`，避免出现 train/test 特征维度不一致问题

## 3. 方法与实现
### 3.1 Baseline：Simple MLP
- 特征标准化：使用训练集统计量（mean/std）做 Z-score
- 模型结构：MLP (116 → 256 → 128 → 1)
  - 激活函数：ReLU
  - Dropout：0.1
- 优化器：Adam
  - lr = 1e-3
  - weight_decay = 1e-5
- 训练策略：
  - 训练/验证划分：80/20
  - Early stopping：patience=10
  - 最佳验证 MSE（本地）：**1.184812**（epoch 70 early stop）

工程化代码位置：
- `src/models.py`：MLP 定义
- `src/train.py`：训练 + early stop + 保存 ckpt（含 mean/std/feature_cols）
- `src/infer.py`：加载 ckpt 生成 submission（列名为 tested_positive）
- `configs/default.yaml`：所有超参与路径配置

## 4. 结果
### 4.1 Kaggle 提交成绩
- Public score：**9.09168**
- Private score：**7.84961**
- 提交说明：`HW01 MLP baseline (fixed submission header)`（status=complete）

> 说明：Public/Private 的差异属正常现象，以 private 更能代表泛化表现。

## 5. 可复现步骤（本地 / WSL2）
### 5.1 环境
建议使用本仓库 conda 环境 `ml-tu`（你已在 `requirements/` 中导出环境文件或记录依赖快照）。

### 5.2 数据放置（不入 Git）
把以下文件放到 `configs/default.yaml` 中配置的 `data_dir` 目录下（默认）：
/mnt/d/ML22/datasets/ml2022spring_hw1/
covid.train.csv
covid.test.csv


### 5.3 训练
在 HW01 目录运行（注意用 module 方式运行）：
bash中
conda activate ml-tu
cd /mnt/d/ML22/ml2022-spring/assignments/HW01_regression
python -m src.train --config configs/default.yaml


输出会包含 device（cuda/cpu）、valid MSE 以及 early stop 信息，并在 outputs/ 下生成：

outputs/mlp.pt

### 5.4 推理并生成提交文件
python -m src.infer --config configs/default.yaml
生成：

outputs/submission.csv

检查表头:id,tested_positive


---

## 6. 踩坑记录（必读）

## 6.1 Kaggle 提交 status=error

原因：提交文件第二列列名写成了训练标签列 tested_positive.4

正确做法：提交列必须是 tested_positive

## 6.2 train/test 特征维度不一致（117 vs 116）

原因：train 里把 id 当作特征了，而 test 推理时丢掉了 id

解决：固定 feature_cols = test.columns - {id}，并且 train/test 均只使用该 feature_cols

## 7. 下一步改进方向

特征工程：尝试对部分特征做 clipping / log transform

模型：更深/更宽 MLP、BatchNorm、或使用学习率调度

训练：K-fold、不同随机种子集成