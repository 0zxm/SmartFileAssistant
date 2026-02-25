# AI 协作日志

## 2026-02-24
- **User**：要求创建协作日志。
- **AI**：建立 `ai.md` 并记录后续需同步日志。

- **User**：反馈 AI 分类无法识别 `.doc`。
- **AI**：更新扩展名映射与 `SUFFIX_NEED2_CLASSIFY`，并新增 `.doc` 读取实现（Word COM）。

- **User**：指出 `ai.md` 格式混乱，难以分辨角色。
- **AI**：按日期-角色格式重排日志，明确区分 User/AI 记录。
- **User**：询问是否可在不依赖 Word 的情况下优雅解析 `.doc`。
- **AI**：调研并给出基于 `antiword`、`textract`、`mammoth` 等纯 Python/第三方方案的建议。
- **User**：指定采用 antiword 方案，并需依赖安装说明。
 - **AI**：将 `.doc` 读取逻辑改为 antiword subprocess，并准备安装指引。
- **User**：要求在代码中预设 antiword 路径 `C:\Users\Canway\Downloads\antiword`。
- **AI**：在 `file_read.py` 默认使用该路径，并允许 `ANTIWORD_CMD` 覆盖。

- **User**：抱怨 `file_read.py` 的 import 顺序“太丑”。
- **AI**：按标准库/第三方分组重排 import，保持可读性。

- **User**：运行批量加载报错“Could not detect encoding”读取多份 `.doc`。
- **AI**：在 `rag_component.py` 新增 `AntiwordDocLoader`，用 `FileReader.read_doc_func()` 统一处理 `.doc` 并接入 `DirectoryLoader`。

- **User**：觉得向量入库太慢，询问加速方法。
- **AI**：给出 GPU/批处理/并行嵌入等多种优化建议。

- **User**：要求将上述建议翻译成中文。
- **AI**：输出中文版优化说明，便于查阅。

- **User**：追问写入耗时主要耗在何处，以及是否受持久化影响。
- **AI**：解释主要瓶颈是本地 `HuggingFaceEmbeddings` 向量化，与是否持久化关系不大，并建议记录细化耗时。

- **User**：希望切换到 SiliconFlow 在线向量（`Qwen/Qwen3-Embedding-8B`）且保留本地方案。
- **AI**：在 `ai_cloud_plugin.py` 创建可复用的 `SiliconFlowEmbedding`，并让 `RagComponent` 通过 `use_siliconflow_embeddings` 选项切换远程或本地嵌入。

- **User**：提供 SiliconFlow 嵌入 API 返回示例，要求适配解析。
- **AI**：更新 `SiliconFlowEmbedding` 以按 `index` 对齐返回的 embedding，校验 `data` 结构并在异常时抛出错误。

- **User**：要求后续每次对话都记录进 `ai.md`。
- **AI**：确认遵循该约定，保持对话摘要同步。

- **User**：发现 BM25 混合检索结果缺少文件名/路径信息。
- **AI**：从 Chroma 读取文档及元数据构建 BM25，使 `file_name`/`file_dir` 在 UI 中可见，并在 BM25 不可用时安全跳过。

- **User**：调用 SiliconFlow 嵌入接口遇到 502 错误，批量入库失败。
- **AI**：为 `SiliconFlowEmbedding` 增加重试/指数退避逻辑，可在 5xx 时自动等待并重试，必要时提示改用本地嵌入。

- **User**：追问混合检索结果为何曾缺少文件名/路径和可点击链接。
- **AI**：解释旧实现只用 `BM25Retriever.from_texts()`，未带入 Chroma 的 `metadatas`，导致 UI 只能看到裸文本；现已改为 `from_documents()` 并补齐元数据即可恢复显示及链接。

## 2026-02-25
- **User**：要求实现“按文件名”分类模式，使用 SiliconFlow LLM 自动聚类所有文件并在线程中执行。
- **AI**：新增 LLM 聚类线程与 UI 入口，引入批量聚类提示词，按 LLM 返回的类别批量移动文件并提供预览确认。
- **User**：反馈 LLM 聚类仅按文件格式分组，期望根据语义（学校/公司等）输出类别。
- **AI**：加强聚类 Prompt 规则，明确禁止按扩展名命名类别，提示 LLM 优先依据内容主题生成“校园/公司/项目”等语义标签。
- **User**：要求提升分类效果、扩大 LLM 的文件内容采样范围。
- **AI**：重写 `get_token()` 抽样逻辑，增大首/中/尾采样句数与字符上限，为分类与聚类 LLM 提供更长的上下文摘要。
- **User**：希望通过 `.env` + `os.getenv()` 管理 API Key，方便开源。
- **AI**：移除硬编码密钥，改为读取环境变量，新增 `.env` 模板与 README 指南，并更新示例脚本。
- **User**：要求进一步完善 README 文档。
- **AI**：重写 README，补充快速开始、分类模式、RAG 流程、环境变量表、目录导航与常见问题等内容。
