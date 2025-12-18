# HWxx - Topic

## 任务说明
- 目标：
- 数据：
- 评价指标：

## 方法与实现
- Baseline：
- 我的改动：
- 关键超参：

## 结果
- 指标/分数：
- 训练曲线/可视化：
- 误差分析：

## 复现
```bash
# 环境
source ../../tools/env.sh
python -m venv ../../.venv
source ../../.venv/bin/activate
pip install -r ../../requirements/base.txt

# 训练
python src/train.py

# 推理/生成提交
python src/infer.py
