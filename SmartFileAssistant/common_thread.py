import os
from PyQt5.QtCore import QThread, pyqtSignal

class CommonThread(QThread):
    finish_signal = pyqtSignal(str) # 耗时任务的处理结果
    error_signal = pyqtSignal(str)  # 任务出错时的错误信息

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        """
        线程执行的核心逻辑（耗时操作放这里）
        """
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finish_signal.emit(result)
        except Exception as e:
            self.error_signal.emit(f"{self.func.__name__}函数执行错误，参数{self.args}-{self.kwargs}：\n{str(e)}")