# LUOJIA

LUOJIA 是一个基于 LangGraph 的政策问答项目。这个最小实现用五条虚构的设备借用规定演示从混合检索到带原文引用的回答。默认离线运行。

## 运行

使用 Python 3.11 或更高版本。在仓库根目录安装依赖并试跑。

```bash
python -m pip install -r requirements.txt
python -m demo ask "How long can I borrow equipment?"
python -m demo evaluate
```

首个问题返回 72 小时的借用期限和一次 48 小时的续借规则，引用 `LAB-LOAN`。评测覆盖借用、押金、归还和缺少证据四种情形。

## 会话与记忆

```bash
python -m demo remember steps --owner sample
python -m demo ask "How long can I borrow equipment?" --owner sample --session first
python -m demo ask "Tell me more" --owner sample --session first
python -m demo ask "Where do I return equipment after hours?" --owner sample --session second --json
```

同一会话可以恢复最近的咨询，`steps` 偏好可以跨会话用于编号排版。状态写到本机的 `.luojia/`。带 `--json` 的回答会提供 `trace_id`，用 `python -m demo trace <trace_id>` 查看节点事件。

## 读源码

| 文件 | 实现 |
| --- | --- |
| [agent_workflow.py](demo/agent_workflow.py) | 用 `StateGraph` 编排检索、上下文、记忆召回、生成、校验与发布。`Command` 控制分支及一次校验修订。 |
| [retrieval.py](demo/retrieval.py) | 将关键词与本地哈希向量得分融合排序，用 LangGraph 缓存复用结果。 |
| [explanation.py](demo/explanation.py) | 从召回片段选取引用，逐句核对原文，再渲染最终答案。 |
| [memory.py](demo/memory.py) | `SqliteSaver` 保留最近 8 轮会话。`SqliteStore` 按用户保存向量化答复偏好，有效期 30 天。 |
| [tracing.py](demo/tracing.py) | 保存节点阶段、耗时、校验和发布状态。 |

要改用 OpenAI 兼容的模型接口，在本机环境中同时设置 `LLM_API_KEY`、`LLM_BASE_URL` 和 `LLM_MODEL`。问答会向配置的服务发送当前问题和召回的虚构条款。服务端输出通过原文引用校验后才用于回答。
