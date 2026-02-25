import os
from PyQt5.QtCore import QThread, pyqtSignal


class IndexFileThread(QThread):
    """
    RAG索引工作者
    - 运行在单独的线程中以避免GUI冻结。
    - 使用信号与主线程通信。
    """
    # 定义信号
    # finished 信号：当任务成功完成时发出
    # 参数：(处理的文件数, 生成的向量块数)
    finished = pyqtSignal(str)
    
    # error 信号：当任务发生异常时发出
    # 参数：(错误信息字符串)
    error = pyqtSignal(str)
    
    # progress 信号：用于在处理过程中更新状态文本
    # 参数：(进度消息字符串)
    progress = pyqtSignal(str)

    def __init__(self, rag_component, folder_path):
        super().__init__()
        self.rag_component = rag_component
        self.folder_path = folder_path

    def run(self):
        """执行耗时的索引任务"""
        try:
            # 为简单起见，我们只在开始和结束时处理
            self.progress.emit(f"正在处理: {self.folder_path}...")

            total_docs, total_chunks = self.rag_component.build_from_folder(self.folder_path)
            
            # 任务完成，发出 finished 信号
            self.finished.emit(f"目标文件夹: {self.folder_path}，成功处理{total_docs}个文件, 生成了{total_chunks}索引块")
            
        except Exception as e:
            import traceback
            error_msg = f"错误: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)