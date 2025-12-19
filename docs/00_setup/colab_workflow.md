### 7.2.1 新建文件
同样在 `docs/00_setup/` 新建：`colab_workflow.md`

### 7.2.2 粘贴并保存
```md
# Colab 工作流（本项目标准）

## 原则
- Colab 负责：完整训练（GPU）、跑通 Notebook、记录环境
- 本地负责：把 Notebook 代码抽成 `src/*.py`，并写作业 README

## 标准流程
1) 切换 Runtime 为 GPU（需要训练时）
2) 挂载 Google Drive：保存数据 / checkpoint / notebook 副本
3) 跑通后导出：
   - 下载 `.ipynb` 到 `assignments/HWxx_*/colab/`
   - 下载 `.py` 或手动整理为脚本到 `assignments/HWxx_*/src/`
4) 记录可复现环境（在 Colab 最后执行）：
```bash
pip freeze > requirements_lock.txt