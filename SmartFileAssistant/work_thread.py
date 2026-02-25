import os
import re
import threading
from PyQt5.QtCore import QThread, pyqtSignal

from util import make_move_file_to, print_file_tree
from my_exception import NotAllowedArgsException
from llm_cluster import LLMFileClusterPlanner

class FileClassifyThread(QThread):
    # 定义信号
    result_signal = pyqtSignal(str)
    finish_signal = pyqtSignal(str)
    preview_signal = pyqtSignal(object)

    def __init__(self, folder, classify_func, file_list: list = None, require_confirmation: bool = False):
        super().__init__()
        self.folder = folder  # 要分类的文件夹路径，假如传入了file_list参数的时候为output文件夹的上级
        self.classify_func = classify_func # 针对某个文件输出分类结果目录名的方法
        self.file_list = file_list  # 要分类的文件名列表，一般来说和folder参数二选一
        self.require_confirmation = require_confirmation
        self._confirm_event = threading.Event() if require_confirmation else None
        self._confirm_approved = False

    def run(self):
        """
        线程执行的核心逻辑（耗时操作放这里）
        
        对（某个文件夹/文件名列表）做分类操作，调用构造函数传入的分类函数
        """
        try:
            # 1. 发送进度信号：开始分类
            self.result_signal.emit(f'正在处理文件夹: {self.folder}')
            
            # 2. 执行文件分类逻辑（原有的耗时代码）
            result_map = {}

            if self.file_list and self.folder:
                for f in self.file_list:
                    classify_res = self.classify_func(f)
                    if classify_res not in result_map:
                        result_map[classify_res] = []
                    result_map[classify_res].append(f)
            elif not self.file_list and self.folder:
                for path, subdirs, files in os.walk(self.folder):
                    for file in files:
                        full_file_path = os.path.join(path, file)
                        classify_res = self.classify_func(full_file_path)
                        if classify_res not in result_map:
                            result_map[classify_res] = []
                        result_map[classify_res].append(full_file_path)
            else:
                raise NotAllowedArgsException(f"func-FileClassifyThread.fun()")
            
            if self.require_confirmation:
                preview_payload = self._build_preview_payload(result_map)
                self.preview_signal.emit(preview_payload)
                self.result_signal.emit("分类结果已生成，等待用户确认...")
                self._confirm_event.wait()
                if not self._confirm_approved:
                    self.result_signal.emit("用户取消了本次分类操作。")
                    self.finish_signal.emit(self.folder)
                    return

            # 3. 移动文件到分类文件夹
            par_dir = os.path.abspath(os.path.join(self.folder, os.pardir))
            output_folder = os.path.join(par_dir, "output")
            for key, value in result_map.items():
                dest_folder = os.path.join(output_folder, key)
                make_move_file_to(value, dest_folder)
            
            # 4. 发送结果信号：文件树文本
            self.result_signal.emit(print_file_tree(output_folder))
            
            # 5. 发送完成信号：分类完成
            self.finish_signal.emit(self.folder)
        
        except Exception as e:
            # 异常处理：把错误信息通过进度信号传递
            self.result_signal.emit(f"分类出错: {str(e)}")

    def set_confirmation_result(self, approved: bool):
        if not self.require_confirmation or not self._confirm_event:
            return
        self._confirm_approved = approved
        self._confirm_event.set()

    def _build_preview_payload(self, result_map):
        total_files = sum(len(files) for files in result_map.values())
        lines = [f"共{total_files}个文件待移动，覆盖{len(result_map)}个分类："]
        for category, files in result_map.items():
            lines.append(f"- {category}（{len(files)} 个）")
            sample_files = files[:3]
            for sample in sample_files:
                lines.append(f"    · {os.path.basename(sample)}")
            if len(files) > len(sample_files):
                lines.append("    · ...")
        detail_text = "\n".join(lines)
        summary = f"是否按照上述结果移动{total_files}个文件？"
        return {"summary": summary, "detail": detail_text}


class LLMAutoClusterThread(QThread):
    result_signal = pyqtSignal(str)
    finish_signal = pyqtSignal(str)
    preview_signal = pyqtSignal(object)

    def __init__(self, folder: str, require_confirmation: bool = True):
        super().__init__()
        self.folder = folder
        self.require_confirmation = require_confirmation
        self._confirm_event = threading.Event() if require_confirmation else None
        self._confirm_approved = False
        self._planner = LLMFileClusterPlanner(folder)

    def run(self):
        try:
            file_paths = self._collect_files()
            if not file_paths:
                self.result_signal.emit("未在所选文件夹下找到任何文件。")
                self.finish_signal.emit(self.folder)
                return
            self.result_signal.emit(f"已收集{len(file_paths)}个文件，正在调用LLM进行聚类……")
            cluster_result = self._planner.cluster_files(file_paths)
            if self.require_confirmation:
                self.preview_signal.emit(cluster_result.preview_payload)
                self.result_signal.emit("分类结果已生成，等待用户确认…")
                self._confirm_event.wait()
                if not self._confirm_approved:
                    self.result_signal.emit("用户取消了本次分类操作。")
                    self.finish_signal.emit(self.folder)
                    return

            output_folder = self._apply_result(cluster_result.category_map)
            tree_text = print_file_tree(output_folder)
            self.result_signal.emit(tree_text)
            self.finish_signal.emit(self.folder)
        except Exception as exc:
            self.result_signal.emit(f"分类出错: {exc}")

    def set_confirmation_result(self, approved: bool):
        if not self.require_confirmation or not self._confirm_event:
            return
        self._confirm_approved = approved
        self._confirm_event.set()

    def _collect_files(self):
        collected = []
        for root, _subdirs, files in os.walk(self.folder):
            for file_name in files:
                collected.append(os.path.join(root, file_name))
        return collected

    def _apply_result(self, category_map):
        par_dir = os.path.abspath(os.path.join(self.folder, os.pardir))
        output_folder = os.path.join(par_dir, "output")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        used_names = set()
        for raw_category, files in category_map.items():
            if not files:
                continue
            safe_name = self._sanitize_category_name(raw_category, used_names)
            dest_folder = os.path.join(output_folder, safe_name)
            make_move_file_to(files, dest_folder)
        return output_folder

    def _sanitize_category_name(self, name: str, used_names: set[str]) -> str:
        safe = re.sub(r'[<>:"/\\|?*]+', '_', (name or "LLM未命名类别").strip())
        safe = safe or "LLM未命名类别"
        candidate = safe
        suffix = 1
        while candidate in used_names:
            suffix += 1
            candidate = f"{safe}_{suffix}"
        used_names.add(candidate)
        return candidate