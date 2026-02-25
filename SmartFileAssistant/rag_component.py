import os
import time
from textwrap import shorten
try:
    import torch
except ImportError:  # pragma: no cover - GPU 依赖可能不存在
    torch = None
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever  
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyMuPDFLoader,
    UnstructuredPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from langchain_community.document_loaders.base import BaseLoader
from langchain_unstructured import UnstructuredLoader   # 自动识别所有文件格式


from ai_cloud_plugin import SiliconFlowAI, SiliconFlowEmbedding
from constants import BGE_ZH_V15_MODEL_PATH, CHROMA_DB_PATH, RAG_LLM_PROMPT_TEMPLATE, TEXT_SPEilTER_CHUNK_SIZE
from file_read import FileReader
from util import get_unique_id, clean_document_content


class AntiwordDocLoader(BaseLoader):
    """将 .doc 文件转换为 Document，避免错误的文本编码探测。"""

    _reader = FileReader()

    def __init__(self, file_path: str, **_: dict):
        self.file_path = file_path

    def load(self) -> list:
        try:
            content = self._reader.read_doc_func(self.file_path)
        except Exception as exc:
            print(f"Error loading file {self.file_path}: {exc}")
            return []

        if not content:
            print(f"Warning: {self.file_path} 经过 antiword 转换后为空")
            return []

        return [Document(page_content=content, metadata={"source": self.file_path})]

class RagComponent():
    
    def __init__(
        self,
        model=BGE_ZH_V15_MODEL_PATH,
        db_path=CHROMA_DB_PATH,
        k=3,
        device: str = None,
        use_siliconflow_embeddings: bool = False,
        siliconflow_model: str = "Qwen/Qwen3-Embedding-8B",
        siliconflow_api_key: str = None,
        siliconflow_batch_size: int = 16,
    ):
        """
        @Args:
            model: 模型名称或者本地路径
            db_path: 向量数据库本地持久化路径
            device: 指定 embedding 模型运行设备（cuda/cpu），默认自动选择
            use_siliconflow_embeddings: 是否切换到 SiliconFlow 在线向量
            siliconflow_model: 在线模型名称
            siliconflow_api_key: 显式传入 API Key，默认读取环境变量
            siliconflow_batch_size: 在线向量请求批量大小
        """
        self.use_siliconflow_embeddings = use_siliconflow_embeddings
        if self.use_siliconflow_embeddings:
            self.device = "siliconflow"
            print(f"⚙️ 正在使用 SiliconFlow 在线向量服务: {siliconflow_model}")
            self.embedding_model = SiliconFlowEmbedding(
                api_key=siliconflow_api_key,
                model=siliconflow_model,
                batch_size=siliconflow_batch_size,
            )
        else:
            self.device = device or self._auto_select_device()
            if self.device.startswith("cuda"):
                print(f"⚙️ 已启用 GPU 加速: {self.device}")
            else:
                print("⚙️ 正在使用 CPU 运行 embedding 模型")
            # TODO 这里只考虑到了本地的情况
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs={
                    "device": self.device,
                    "local_files_only": True,
                    "trust_remote_code": True
                },
                encode_kwargs={
                    "batch_size" : 64,
                    "normalize_embeddings": True
                }
            )
        self.vector_db = Chroma(
            embedding_function=self.embedding_model,
            persist_directory=db_path,
            client_settings=Settings(
                persist_directory="./chroma_db",
                anonymized_telemetry=False,  # 关闭遥测，略提升速度
                is_persistent=True  # 显式指定持久化，避免隐式判断
            )
        )        
        self.k = k  # 统一设置k值，向量/BM25保持一致
        self._init_or_update_bm25()

    def _init_or_update_bm25(self):
        """
        初始化/更新BM25检索器：
        1. 启动时：加载Chroma持久化的历史文档初始化BM25
        2. 新增文档后：同步更新BM25，保证和向量库数据一致
        自动适配空库/有历史数据场景，统一设置k值
        """
        # 重新获取Chroma中所有文档及其元数据（历史+新增，保证数据一致）
        snapshot = self.vector_db.get(include=["documents", "metadatas"])
        documents = snapshot.get("documents") or []
        metadatas = snapshot.get("metadatas") or []

        if documents:
            enriched_docs = []
            for idx, content in enumerate(documents):
                metadata = metadatas[idx] if idx < len(metadatas) else {}
                enriched_docs.append(Document(page_content=content, metadata=metadata or {}))

            self.all_docs = enriched_docs
            # 初始化/更新BM25，保留元数据供 UI 展示文件名/路径
            self.bm25_retriever = BM25Retriever.from_documents(enriched_docs)
            self.bm25_retriever.k = self.k
        else:
            # 空库时，BM25设为None，后续检索时判空即可
            self.all_docs = []
            self.bm25_retriever = None

    def build_from_folder(self, folder_path: str):
        """
        从指定文件夹加载、处理文档，并构建/更新向量数据库。
        @Args:
            folder_path: 包含知识库文档的文件夹路径。
        @Returns:
            一个包含处理统计信息的元组: (total_docs, total_chunks)
        """
        # --- 1. 加载文档 ---
        print(f"开始从文件夹加载文档: {folder_path}")
        total_start = time.time()
        step_marker = total_start

        all_documents = []
        
        # 定义文件类型及其对应的加载器
        loader_mapping = {
            "**/*.pdf": (PyMuPDFLoader, {}),
            "**/*.docx": (Docx2txtLoader, {}),
            # "**/*.doc":(TextLoader, {"autodetect_encoding": True}),
            "**/*.doc": (AntiwordDocLoader, {}),
            "**/*.md": (TextLoader, {"autodetect_encoding": True}),
            "**/*.txt": (TextLoader, {"autodetect_encoding": True}),
        }
        
        # 分别加载每种类型的文件
        for glob_pattern, (loader_cls, load_kwargs) in loader_mapping.items():
            try:
                loader = DirectoryLoader(
                    path=folder_path,
                    glob=glob_pattern,
                    loader_cls=loader_cls,
                    show_progress=True,
                    use_multithreading=True, # TODO 多线程有句柄风险
                    silent_errors=True,
                    loader_kwargs=load_kwargs,
                )
                docs = loader.load()
                if docs:
                    print(f"-加载 {glob_pattern}: {len(docs)} 个文档")
                    all_documents.extend(docs)
            except Exception as e:
                print(f"警告: 加载 {glob_pattern} 时出错: {str(e)}")
                continue
        
        if not all_documents:
            print("警告：在文件夹中未找到可处理的文档。")
            return 0, 0

        step_marker = self._log_step_time("加载文件", step_marker)
        
        total_docs = len(all_documents)
        print(f"加载到的文档数量: {len(all_documents)}")
        for i, doc in enumerate(all_documents):
            print(f"{i+1}. {doc.metadata.get('source')} | 内容长度: {len(doc.page_content)}")

        for doc in all_documents:
            # 从source中获取文件绝对路径
            file_abs_path = doc.metadata.get('source', '')
            if not file_abs_path:
                print(f"警告：文档缺少source元数据，无法关联文件名，元数据={doc.metadata}")
                continue
            file_name = os.path.basename(file_abs_path)  # 核心：提取纯文件名
            file_type = os.path.splitext(file_name)[-1].lstrip('.')  # 提取文件后缀，去掉点
            file_dir = os.path.dirname(file_abs_path)  # 可选：文件所在目录
            # 新增元数字段到doc.metadata
            doc.metadata['file_name'] = file_name    # 纯文件名，UI显示用
            doc.metadata['file_type'] = file_type    # 文件类型，后续可按类型适配分割参数
            doc.metadata['file_dir'] = file_dir      # 可选：文件所在目录

        step_marker = self._log_step_time("补充元数据", step_marker)

        print("开始清洗文档内容，移除多余空行和空格...")
        cleaned_docs = []
        for doc in all_documents:
            doc.page_content = clean_document_content(doc.page_content)
            if doc.page_content:
                cleaned_docs.append(doc)
        valid_docs = len(cleaned_docs)
        if valid_docs < total_docs:
            print(f"清洗完成，过滤掉 {total_docs - valid_docs} 个空文档，剩余 {valid_docs} 个有效文档")

        step_marker = self._log_step_time("内容清洗", step_marker)

        # --- 2. 切分文档 ---
        documents = cleaned_docs  # 将合并后的文档赋值给documents变量
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=TEXT_SPEilTER_CHUNK_SIZE,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", " "],
            keep_separator=True,
            length_function=len
        )
        print("开始切分文档...")
        split_docs = text_splitter.split_documents(documents)

        step_marker = self._log_step_time("递归切分", step_marker)

        MIN_CHUNK_LENGTH = 350  # 设定最小块长度，保证展示内容够长
        merged_docs = []
        temp_doc = None  # 临时存储同一文件下待合并的短块

        for doc in split_docs:
            doc_source = doc.metadata.get('source')
            if temp_doc is None:
                temp_doc = doc
            else:
                temp_source = temp_doc.metadata.get('source')
                if doc_source != temp_source:
                    merged_docs.append(temp_doc)
                    temp_doc = doc
                else:
                    temp_doc = Document(
                        page_content=f"{temp_doc.page_content}\n{doc.page_content}",
                        metadata=temp_doc.metadata
                    )

            if temp_doc is not None and len(temp_doc.page_content) >= MIN_CHUNK_LENGTH:
                merged_docs.append(temp_doc)
                temp_doc = None

        if temp_doc is not None:
            merged_docs.append(temp_doc)

        split_docs = merged_docs

        step_marker = self._log_step_time("块合并", step_marker)

        total_chunks = len(split_docs)
        print(f"切分完成，共得到 {total_chunks} 个文本块。")
        # --- 3. 添加到向量数据库 ---
        print(f"开始将 {total_chunks} 个文本块添加到向量数据库...")
        self._add_documents(split_docs)
        step_marker = self._log_step_time("向量写入", step_marker)
        print("✅ 向量数据库更新完成。")
        
        self._init_or_update_bm25()
        step_marker = self._log_step_time("BM25 更新", step_marker)

        total_cost = time.time() - total_start
        print(f"📊 初始化总耗时 {total_cost:.2f}s")

        return total_docs, total_chunks

    def _add_documents(self, document_list):
        """分批写入向量，并在批次间短暂让出 GIL。"""
        if not document_list:
            return
        ids = [get_unique_id(doc, idx) for idx, doc in enumerate(document_list)]
        existing_ids = self._get_existing_ids(ids)
        if existing_ids:
            print(f"跳过 {len(existing_ids)} 个已存在的文本块，避免重复入库。")

        docs_to_add = []
        ids_to_add = []
        for doc, doc_id in zip(document_list, ids):
            if doc_id in existing_ids:
                continue
            docs_to_add.append(doc)
            ids_to_add.append(doc_id)

        if not docs_to_add:
            print("没有新的文本块需要写入，跳过存储阶段。")
            return

        BATCH_SIZE = 32
        total_added = 0
        for start in range(0, len(docs_to_add), BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_docs = docs_to_add[start:end]
            batch_ids = ids_to_add[start:end]
            self.vector_db.add_documents(documents=batch_docs, ids=batch_ids)
            total_added += len(batch_docs)
            time.sleep(0.16)  # 让出GIL，避免长期占用造成UI卡顿

        print(f"新增 {total_added} 个文本块完成持久化。")

    def _get_existing_ids(self, candidate_ids, batch_size=256):
        if not candidate_ids:
            return set()
        existing = set()
        for start in range(0, len(candidate_ids), batch_size):
            batch = candidate_ids[start:start + batch_size]
            try:
                response = self.vector_db._collection.get(ids=batch)
            except Exception as exc:
                print(f"警告：查询已存在向量ID失败({exc})，跳过该批次。")
                continue
            existing.update(response.get("ids", []))
        return existing

    def _auto_select_device(self) -> str:
        if torch is None:
            return "cpu"
        try:
            if torch.cuda.is_available():
                return "cuda"
        except Exception as exc:
            print(f"警告：检测 GPU 可用性时出错({exc})，回退到 CPU。")
        return "cpu"

    def _log_step_time(self, label: str, start_time: float) -> float:
        duration = time.time() - start_time
        print(f"⏱️ 步骤[{label}]耗时 {duration:.2f}s")
        return time.time()

    def _similarity_search(self, query):
        """
        执行相似性搜索并返回结果。
        """
        print(f"🚀 正在执行相似性搜索，查询: '{query}'") 
        # TODO 混合检索
        # combined_results = self.vector_db.similarity_search(query, k) + self.bm25_retriever.invoke(query)
        results = self.vector_db.similarity_search_with_score(query, self.k)
        bm25_sc_results = []
        if self.bm25_retriever is not None:
            bm25_results = self.bm25_retriever.invoke(query)
            bm25_sc_results = [(doc, 0.0) for doc in bm25_results]
        combined_results = results + bm25_sc_results
        # TODO 重排序
        # from langchain_community.retrievers import CrossEncoderReranker  
        # reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-base")  
        # final_results = reranker.rerank(query, self.vector_db.similarity_search(query, k=10))
        return combined_results
    

    def _format_reference_docs(self, reference_docs):
        """格式化参考文档，便于提示词引用。"""
        formatted = []
        for idx, (doc, score) in enumerate(reference_docs, 1):
            metadata = getattr(doc, 'metadata', {}) or {}
            source_path = metadata.get('source', '')
            file_name = metadata.get('file_name') or (os.path.basename(source_path) if source_path else '未命名文档')
            snippet_source = doc.page_content.strip()
            if snippet_source:
                snippet = shorten(snippet_source.replace('\n', ' '), width=TEXT_SPEilTER_CHUNK_SIZE+1, placeholder='…')
            else:
                snippet = '（该片段为空）'
            formatted.append(f"[doc_{idx}] 《{file_name}》 | 置信度: {score:.4f}\n{snippet}")
        return "\n\n".join(formatted)

    def generate_answer_with_llm(self, reference_docs: list, user_query: str):
        ai = SiliconFlowAI()
        formatted_docs = self._format_reference_docs(reference_docs)
        prompt = RAG_LLM_PROMPT_TEMPLATE.format(reference_docs=formatted_docs, user_query=user_query)
        print(prompt, " ----------------")
        resp = ai.chat_with_ai(prompt)
        return resp
