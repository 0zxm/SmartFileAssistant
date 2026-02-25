import sys
import os
from pathlib import Path
try:
    import markdown  # 仅在可用时用于渲染
except ImportError:  # pragma: no cover - 环境可能未安装
    markdown = None
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QFileDialog,
    QMessageBox,
    QTextEdit,
    QHBoxLayout,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QTextBrowser
)
from PyQt5.QtCore import QUrl
from classify import classify_by_ai, classify_by_extension
from constants import QSS_TEXT, SUFFIX_NEED2_CLASSIFY
from work_thread import FileClassifyThread, LLMAutoClusterThread
from index_thread import IndexFileThread
from common_thread import CommonThread
from rag_component import RagComponent


class ClassificationPreviewDialog(QDialog):
    def __init__(self, summary: str, detail: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("确认分类")
        layout = QVBoxLayout()

        summary_label = QLabel(summary)
        summary_label.setWordWrap(True)
        layout.addWidget(summary_label)

        if detail:
            detail_browser = QTextBrowser()
            detail_browser.setPlainText(detail)
            detail_browser.setMinimumWidth(420)
            detail_browser.setMinimumHeight(200)
            layout.addWidget(detail_browser)

        button_box = QDialogButtonBox(QDialogButtonBox.Yes | QDialogButtonBox.No)
        button_box.button(QDialogButtonBox.Yes).setText("确认执行")
        button_box.button(QDialogButtonBox.No).setText("取消")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

class FileClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.classify_index = 2  # 默认使用混合ai分类
        self.classify_thread = None
        self.ai_answer_thread = None
        self._latest_doc_preview = ""
        self._latest_result_count = 0
        # TODO 没有考虑线程安全
        self.rag_component = RagComponent(k=3, use_siliconflow_embeddings=True)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('文件自动分类与搜索')
        self.setGeometry(100, 100, 400, 300)

        # 创建布局
        self.layout = QVBoxLayout()

        # 输入框
        self.search_input = QLineEdit(self)
        self.search_input.setPlaceholderText('输入搜索关键词')
        self.search_input.returnPressed.connect(self.search_files)
        self.layout.addWidget(self.search_input)

        self.rag_h_layout = QHBoxLayout()
        # 搜索按钮
        self.search_button = QPushButton('搜索', self)
        self.search_button.clicked.connect(self.search_files)
        self.rag_h_layout.addWidget(self.search_button)
        
        # 初始化按钮
        self.init_button = QPushButton('初始化', self)
        self.init_button.clicked.connect(self.init_rag_vector)
        self.rag_h_layout.addWidget(self.init_button)
        self.layout.addLayout(self.rag_h_layout)

        # 标签显示结果
        self.result_label = QLabel('搜索结果:', self)
        self.layout.addWidget(self.result_label)
        self.result_text_edit = QTextBrowser(self)
        self.result_text_edit.setReadOnly(True)
        self.result_text_edit.setOpenLinks(False)
        self.result_text_edit.anchorClicked.connect(self._open_reference_link)
        self.result_text_edit.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.layout.addWidget(self.result_text_edit)

        # 分类模组
        self.classify_layout = QHBoxLayout()
        self.classify_mode_combobox = QComboBox(self)
        self.classify_mode_combobox.addItems(["按文件类型", "按文件名", "混合ai分类"])
        self.classify_mode_combobox.setCurrentIndex(2)
        self.classify_mode_combobox.currentIndexChanged.connect(self.change_classify_mode)
        
        self.classify_button = QPushButton('分类文件', self)
        self.classify_button.clicked.connect(self.classify_func)
        self.classify_layout.addWidget(self.classify_button)
        self.classify_layout.addWidget(self.classify_mode_combobox)
        self.layout.addLayout(self.classify_layout)

        self.setLayout(self.layout)
        self.setStyleSheet(QSS_TEXT)

    def change_classify_mode(self, index):
        self.classify_index = index

    def classify_func(self):
        if self.classify_index == 0:
            self.extension_classify_files()
        elif self.classify_index == 1:
            self.filename_classify_files()
        elif self.classify_index == 2:
            self.mix_ai_classify_files()

    def search_files(self):
        keyword = self.search_input.text().strip()
        if not keyword:
            self.result_label.setText('请先输入关键词')
            self.result_text_edit.clear()
            return

        results = self.rag_component._similarity_search(keyword)
        if not results:
            self.result_label.setText('未检索到匹配内容')
            self.result_text_edit.clear()
            return

        self._latest_result_count = len(results)
        doc_blocks = ["### 参考文档预览"]
        for idx, (doc, score) in enumerate(results, 1):
            doc_blocks.append(self._format_doc_preview(idx, doc, score))
        doc_section = "\n\n".join(doc_blocks)
        self._latest_doc_preview = doc_section

        if keyword.endswith(("?", "？")):
            placeholder = "### AI 回答\n> 正在生成，请稍候..."
            combined = "\n\n".join([placeholder, doc_section])
            self._render_markdown(combined)
            self.result_label.setText(f'检索完成，共{len(results)}条候选，AI回答生成中…')
            self._start_ai_answer_thread(results, keyword)
        else:
            self._render_markdown(doc_section)
            self.result_label.setText(f'检索完成，共{len(results)}条候选')
            self.ai_answer_thread = None
        

    def init_rag_vector(self):
        folder = QFileDialog.getExistingDirectory(self, "请选择要索引的文件夹")
        if not folder:
            self.result_label.setText("操作取消。") 
            return
        self.init_button.setEnabled(False)
        self.result_label.setText(f"开始处理文件夹: {folder}...这可能需要几分钟")
        QApplication.processEvents()

        self.active_threads_count = 1
        self.index_file_thread = IndexFileThread(self.rag_component, folder)
        self.index_file_thread.progress.connect(self.update_result)
        self.index_file_thread.finished.connect(self.index_finished)
        self.index_file_thread.error.connect(self.update_result)
        self.index_file_thread.start()
        

    def extension_classify_files(self):
        folder = QFileDialog.getExistingDirectory(self, "选择要分类的文件夹")
        if not folder:
            return 
    
        if self.classify_thread and self.classify_thread.isRunning():
            QMessageBox.warning(self, "提示", "分类任务正在进行中，请等待！")
            return
        
        self.active_threads_count = 1
        self.classify_thread = FileClassifyThread(
            folder,
            classify_func=classify_by_extension,
            require_confirmation=False
        )
        self._bind_classify_thread_signals(self.classify_thread)

        self.classify_thread.start()

    def filename_classify_files(self):
        folder = QFileDialog.getExistingDirectory(self, "选择要分类的文件夹")
        if not folder:
            return

        if self.classify_thread and self.classify_thread.isRunning():
            QMessageBox.warning(self, "提示", "分类任务正在进行中，请等待！")
            return

        self.active_threads_count = 1
        self.classify_thread = LLMAutoClusterThread(
            folder=folder,
            require_confirmation=True
        )
        self._bind_classify_thread_signals(self.classify_thread)
        self.classify_thread.start()
               
    def mix_ai_classify_files(self):
        folder = QFileDialog.getExistingDirectory(self, "选择要分类的文件夹")
        if not folder:
            return 
    
        if self.classify_thread and self.classify_thread.isRunning():
            QMessageBox.warning(self, "提示", "分类任务正在进行中，请等待！")
            return
        
        content_files = []
        suffix_files = []
        for path, subdirs, files in os.walk(folder):
                for file in files:
                    full_file_path = os.path.join(path, file)
                    _, ext = os.path.splitext(file)
                    ext = ext.lower()
                    if ext in SUFFIX_NEED2_CLASSIFY:
                        content_files.append(full_file_path)
                    else:
                        suffix_files.append(full_file_path)

        self.active_threads_count = 2       # 初始化活动线程计数器

        self.classify_thread1 = FileClassifyThread(
            folder=folder,
            classify_func=classify_by_ai,
            file_list=content_files,
            require_confirmation=True
        )
        self.classify_thread2 = FileClassifyThread(
            folder=folder,
            classify_func=classify_by_extension,
            file_list=suffix_files,
            require_confirmation=False
        )
        
        self._bind_classify_thread_signals(self.classify_thread1)
        self._bind_classify_thread_signals(self.classify_thread2)

        self.classify_thread1.start()
        self.classify_thread2.start()

     # ---------------------- 主线程UI更新槽函数 ----------------------

    def update_result(self, text):
        """更新分类结果文本（主线程执行）"""
        self.result_text_edit.setText(text)

    def classify_finished(self, folder):
        """
        分类完成提示（主线程执行）
        只有当所有线程都跑完时，才提示用户。
        """

        self.last_finished_folder = folder
        self.active_threads_count -= 1
        if self.active_threads_count == 0:
            # 真正的结束逻辑放在这里
            # final_output_folder = os.path.abspath(os.path.join(self.last_finished_folder, os.pardir)) + "/output"
            #刷新一次最终的树状图
            # self.result_label.setText(print_file_tree(final_output_folder))
            QMessageBox.information(\
                self, "分类完成", f'所有分类任务执行完毕: {self.last_finished_folder}')
            
            # TODO 线程回收，可以用一个列表存储所有工作线程循环回收
            self.classify_thread = None
            print("=== 所有线程已结束 ===")

    def index_finished(self, msg):
        """
        文件切片索引完成提示
        """
        self.active_threads_count -= 1
        if self.active_threads_count == 0:
            QMessageBox.information(self, "文件切片索引完成", msg)
            if self.index_file_thread and self.index_file_thread.isRunning():
                self.index_file_thread.quit()    # 请求线程事件循环停止
                self.index_file_thread.wait()    # 等待线程完全终止
            self.index_file_thread = None
            self.init_button.setEnabled(True)
            print("=== 所有线程已结束 ===")

    def _format_doc_preview(self, index, doc, score):
        metadata = getattr(doc, 'metadata', {}) or {}
        source_path = metadata.get('source', '')
        file_name = metadata.get('file_name') or (os.path.basename(source_path) if source_path else '未命名文档')
        file_dir = metadata.get('file_dir') or (os.path.dirname(source_path) if source_path else '未知路径')
        snippet_source = doc.page_content.strip()
        if snippet_source:
            preview_text = snippet_source.replace('\n', ' ')
            # 直接按字符截断，避免中文无空格时被过度缩短
            snippet = preview_text[:320] + ('...' if len(preview_text) > 320 else '')
        else:
            snippet = '（该片段为空）'
        snippet = snippet.replace('```', '` ` `')
        link_target = self._build_file_link(source_path)
        if link_target:
            dir_info = file_dir or '未知路径'
            path_line = (
                f"路径: [打开文件]({link_target})  \n"
                f"所在目录: {dir_info}  \n"
            )
        else:
            path_line = f"路径: {file_dir or '未知路径'}  \n"
        return (
            f"**[doc_{index}] {file_name}**  \n"
            f"{path_line}"
            f"分数: {score:.4f}  \n"
            f"内容片段：\n```text\n{snippet}\n```"
        )

    def _build_file_link(self, source_path):
        if not source_path:
            return ""
        try:
            normalized_path = Path(source_path).resolve()
        except OSError:
            return ""
        return normalized_path.as_uri()

    def _bind_classify_thread_signals(self, thread):
        thread.result_signal.connect(self.update_result)
        thread.finish_signal.connect(self.classify_finished)
        thread.preview_signal.connect(self.handle_classify_preview)

    def handle_classify_preview(self, payload):
        thread = self.sender()
        summary = "确认执行文件分类操作？"
        detail = ""
        if isinstance(payload, dict):
            summary = payload.get("summary", summary)
            detail = payload.get("detail", "")
        elif isinstance(payload, str):
            detail = payload

        dialog = ClassificationPreviewDialog(summary, detail, self)
        user_choice = dialog.exec_()

        approved = user_choice == QDialog.Accepted
        if thread and hasattr(thread, "set_confirmation_result"):
            thread.set_confirmation_result(approved)

    def _start_ai_answer_thread(self, reference_docs, keyword):
        thread = CommonThread(
            self.rag_component.generate_answer_with_llm,
            reference_docs,
            keyword
        )
        self.ai_answer_thread = thread
        thread.finish_signal.connect(self._handle_ai_answer)
        thread.error_signal.connect(self._handle_ai_answer_error)
        thread.start()

    def _handle_ai_answer(self, answer_text):
        thread = self.sender()
        if thread is not self.ai_answer_thread:
            if thread:
                thread.deleteLater()
            return

        formatted_answer = (answer_text or "").strip()
        sections = [f"### AI 回答\n{formatted_answer}"]
        if self._latest_doc_preview:
            sections.append(self._latest_doc_preview)
        self._render_markdown("\n\n".join(sections))
        self.result_label.setText(f'检索完成，共{self._latest_result_count}条候选，AI回答已生成')

        if thread:
            thread.deleteLater()
        self.ai_answer_thread = None

    def _handle_ai_answer_error(self, error_text):
        thread = self.sender()
        if thread is not self.ai_answer_thread:
            if thread:
                thread.deleteLater()
            return

        cleaned_error = (error_text or "未知错误").strip().replace('```', '` ` `')
        error_section = "### AI 回答\n生成失败，请稍后重试。\n\n```text\n{}\n```".format(cleaned_error)
        sections = [error_section]
        if self._latest_doc_preview:
            sections.append(self._latest_doc_preview)
        self._render_markdown("\n\n".join(sections))
        self.result_label.setText('AI回答生成失败，请重试')

        if thread:
            thread.deleteLater()
        self.ai_answer_thread = None

    def _open_reference_link(self, url: QUrl):
        if not url:
            return

        target_url = url
        if url.isLocalFile():
            local_path = url.toLocalFile()
            if not os.path.exists(local_path):
                QMessageBox.warning(self, "文件不存在", f"找不到路径: {local_path}")
                return
            target_url = QUrl.fromLocalFile(local_path)

        QDesktopServices.openUrl(target_url)

    def _render_markdown(self, markdown_text: str):
        """优先使用原生 Markdown 渲染，不支持时回退。"""
        if hasattr(self.result_text_edit, "setMarkdown"):
            self.result_text_edit.setMarkdown(markdown_text)
            return
        if markdown:
            html_text = markdown.markdown(markdown_text)
            self.result_text_edit.setHtml(html_text)
        else:
            self.result_text_edit.setPlainText(markdown_text)
            
