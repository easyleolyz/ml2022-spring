### 7.3.1 新建文件
在 `docs/00_setup/` 新建：`kaggle_workflow.md`

### 7.3.2 粘贴并保存
```md
# Kaggle 工作流（本项目标准）

## 重要原则
- `kaggle.json` 属于密钥文件：严禁提交到 Git（仓库 .gitignore 已忽略）
- `submission.csv` 等提交文件默认不进 Git（建议放 `assignments/HWxx_*/outputs/`）

## 推荐流程
1) 在 Kaggle 网站生成 API Token（得到 kaggle.json）
2) 在 Colab 中配置 Kaggle API（放到私密位置/Drive 私密目录）
3) 在 Colab 生成 submission.csv 后直接提交
4) 把分数记录到对应作业 README：
- 分数（public/private）
- 使用的模型与关键超参
- 复现步骤与环境说明