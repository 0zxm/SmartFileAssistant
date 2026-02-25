# FileAutoClassify & Semantic Search

智能文件整理、聚类与检索工具。PyQt 前端整合本地 HuggingFace 向量库、BM25 语义检索与多种自动分类策略，并可接入 SiliconFlow / OpenRouter 模型获得更精准的问答与归档体验。

## 功能速览
- **多策略分类**：按扩展名、按文件名语义聚类、混合 AI 分类三合一，可预览后批量落盘。
- **RAG 检索**：向量检索 + BM25 混合召回，问题以 `?` 结尾时自动触发 LLM 生成答案。
- **多格式解析**：antiword、PyMuPDF、Docx2txt、Markdown/Text 等统一切片入库，自动补充 `file_name`、`file_dir` 元数据。
- **线程化 UI**：索引、分类、回答均在独立线程执行，避免阻塞界面。

## 快速开始
1. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```
2. 准备环境变量（参考下节）并在当前终端加载：
   - Linux/macOS
     ```bash
     cp .env.example .env
     source .env
     ```
   - Windows PowerShell
     ```powershell
     Copy-Item .env.example .env
     Get-Content .env | ForEach-Object {
       if ($_ -match '^(?<k>[^#=]+)=(?<v>.*)$') {
         Set-Item -Path Env:$($Matches.k.Trim()) -Value $Matches.v.Trim()
       }
     }
     ```
3. 启动桌面端
   ```bash
   python main.py
   ```
4. 在 GUI 中使用“搜索”“初始化”“分类文件”按钮完成索引、检索与分类操作。

## 分类模式
- **按文件类型**：仅依据扩展名，将常见格式（pdf/doc/docx/md/txt 等）搬运到对应目录，速度最快。
- **按文件名（语义聚类）**：调用 SiliconFlow LLM 读取采样文本与文件名，自动生成语义类目并移动文件，支持执行前人工确认。
- **混合 AI 分类**：对需要深度理解的文件（`SUFFIX_NEED2_CLASSIFY`）调用 `classify_by_ai`，其余走扩展名策略，同时并行两个线程提升吞吐。

## RAG 检索流程
1. **初始化**：点击“初始化”选择资料目录，`RagComponent.build_from_folder()` 会加载->清洗->切块->写入 Chroma，并自动刷新 BM25 检索器。
2. **搜索**：输入关键词后检索相似片段，UI 会展示可点击的路径、目录以及内容摘要。
3. **生成回答**：若查询以 `?` 或 `？` 结尾，后台线程会把候选文档交给 SiliconFlow LLM，并在 Markdown 视图中合并“AI 回答”与“参考文档”。

## 环境变量说明
| 变量名 | 说明 | 是否必填 | 默认值 |
| --- | --- | --- | --- |
| `SILICONFLOW_API_KEY` | SiliconFlow 账号密钥，供聊天/向量接口共用 | ✅ | 无 |
| `SILICONFLOW_CHAT_MODEL` | 聊天模型，如 `deepseek-ai/DeepSeek-V2.5` | ⛔ | `deepseek-ai/DeepSeek-V2.5` |
| `SILICONFLOW_CHAT_ENDPOINT` | 聊天接口地址 | ⛔ | `https://api.siliconflow.cn/v1/chat/completions` |
| `SILICONFLOW_EMBEDDING_MODEL` | 嵌入模型，如 `Qwen/Qwen3-Embedding-8B` | ⛔ | `Qwen/Qwen3-Embedding-8B` |
| `SILICONFLOW_EMBEDDING_URL` | 嵌入接口地址 | ⛔ | `https://api.siliconflow.cn/v1/embeddings` |
| `OPENROUTER_API_KEY` | OpenRouter 密钥（若使用） | ⚪ | 无 |
| `OPENROUTER_MODEL` | OpenRouter 模型名 | ⚪ | `nex-agi/deepseek-v3.1-nex-n1:free` |

> `.env` 已加入 `.gitignore`，请勿提交真实密钥；也可直接配置系统环境变量。

## 项目结构速览
- `main.py`：PyQt 启动入口。
- `ui.py`：主界面逻辑，负责搜索、RAG 展示与分类入口。
- `rag_component.py`：构建/检索逻辑，含向量库、BM25 与 LLM 回调。
- `ai_cloud_plugin.py`：SiliconFlow/OpenRouter API 适配。
- `file_read.py`：多格式文件读取与 antiword 封装。
- `embedding/`：包含 SiliconFlow 接口示例、嵌入工具脚本。
- `refactored_project/`：拆分后的 client/server 版本，可作为进一步工程化的参考。

## 常见问题
- **`.doc` 无法解析**：安装 antiword 并在 `file_read.py` 中确认 `ANTIWORD_CMD` 路径；或将可执行文件所在目录加入 `PATH`。
- **`.env` 未生效**：新的终端窗口不会自动继承变量，请在每个终端执行 `source .env`/PowerShell 脚本，或写入系统级环境变量。
- **向量入库过慢**：留意终端日志；可调低 `TEXT_SPEilTER_CHUNK_SIZE` 或减少一次加载的文件数，必要时关闭 `use_siliconflow_embeddings` 走本地模型。
- **Chroma 路径冲突**：默认落在 `./chroma_db`，可在 `constants.py` 中修改 `CHROMA_DB_PATH` 后重新初始化。

## 进一步开发
- 想要命令行 / FastAPI 形态，可参考 `refactored_project/server` 与 `refactored_project/client_refactored`。
- 欢迎通过 Issue/PR 提交新的分类策略、检索 reranker 或 UI 交互改进，共同完善该工具。

