# ML2022 Spring（李宏毅）学习与作业工程化整理

目标：
- 系统学习 NTU 李宏毅 Machine Learning 2022 Spring
- 完成 15 个作业（Colab 跑训练 + 本地整理）
- 输出中文笔记与作业报告（保留关键英文术语）

工作方式：
- 本地：WSL2（仓库位于 D 盘），仅做轻量测试与代码整理
- 训练：Google Colab（GPU），跑通后导出到本地归档
- 版本管理：Git + GitHub（SSH）

## 目录结构
- `docs/00_setup/`：环境与工作流（WSL2 / Colab / Kaggle）
- `docs/01_lecture_notes/`：课程笔记（中文为主）
- `assignments/`：15 个作业（每个作业一个独立子目录）
- `requirements/`：依赖（base + 作业增量）
- `tools/`：脚本（缓存重定向、导出等）
- `common/`：跨作业复用工具

## 进度
- [ ] HW01 Regression
- [ ] HW02 Classification
- [ ] HW03 CNN
- [ ] HW04 Self-attention
- [ ] HW05 Transformer
- [ ] HW06 GAN
- [ ] HW07 BERT
- [ ] HW08 Autoencoder
- [ ] HW09 Explainable AI
- [ ] HW10 Attack
- [ ] HW11 Adaptation
- [ ] HW12 RL
- [ ] HW13 Compression
- [ ] HW14 Life-long Learning
- [ ] HW15 Meta Learning

## 本地快速开始
```bash
cd /mnt/d/ML22/ml2022-spring
source tools/env.sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/base.txt
