import os
import shutil
import hashlib
import re
from langchain_core.documents import Document

# 生成唯一ID的简易函数
def get_unique_id(doc: Document, chunk_index: int):
    source = doc.metadata.get('source', 'unknown')
    
    # 组合标识：文件路径 + 块索引 + 内容哈希
    unique_string = f"{source}::chunk_{chunk_index}::{doc.page_content}"
    
    return hashlib.md5(unique_string.encode('utf-8')).hexdigest()

def make_move_file_to(src_path_li, dest_folder):
    # 将所有源文件拷贝到目标文件夹
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    for src_path in src_path_li:
        filename = os.path.basename(src_path)
        dest_path = os.path.join(dest_folder, filename)
        shutil.copy2(src_path, dest_path)
    return True


def print_file_tree(
    path: str,
    prefix: str = "",
    ignore: list = [],
    is_last: bool = True,
):
    """
    手动打印目录树形结构
    :param path: 目标目录路径
    :param prefix: 前缀（控制缩进和线条）
    :param ignore: 忽略的文件/目录名
    :param is_last: 是否是当前目录的最后一个项
    """
    ignore = ignore or [".git", "__pycache__", "*.pyc"]
    # 过滤忽略项
    name = os.path.basename(path)
    for pattern in ignore:
        if pattern.startswith("*") and name.endswith(pattern[1:]):
            return ""
        if name == pattern:
            return ""

    # 打印当前项的前缀和名称
    connector = "└── " if is_last else "├── "
    current_line = (f"{prefix}{connector}{name}\n")
    # print(current_line[:-1])

    # 如果是目录，递归处理子项
    if os.path.isdir(path):
        # 更新前缀（控制子项的缩进线条）
        new_prefix = prefix + ("    " if is_last else "│   ")
        items = os.listdir(path)
        # 过滤忽略项
        items = [item for item in items if not any(
            item == p or (p.startswith("*") and item.endswith(p[1:]))
            for p in ignore
        )]
        # 遍历子项
        for i, item in enumerate(items):
            item_path = os.path.join(path, item)
            child_lines = print_file_tree(
                path=item_path,
                prefix=new_prefix,
                ignore=ignore,
                is_last=(i == len(items) - 1)
            )
            current_line += child_lines 
        return current_line
    else:
        return current_line

def clean_document_content(text: str) -> str:
    """
    清洗文档文本内容，移除多余空行、连续空格，保留正常排版
    :param text: 原始文档文本
    :return: 清洗后的干净文本
    """
    if not text:
        return ""
    # 1. 统一换行符并去掉多余回车
    text = text.replace("\r", "")

    # 2. 压缩多余空行：最多保留一个空行表示段落
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # 3. 去掉每行首尾空白，并过滤掉纯空行
    lines = [line.strip() for line in text.split("\n")]
    non_empty_lines = [line for line in lines if line]

    # 4. 按段落重新拼接，保留单个换行作为软分隔
    clean_text = "\n".join(non_empty_lines)

    # 5. 将连续空格/制表符压缩为单空格
    clean_text = re.sub(r"[ \t]+", " ", clean_text)

    return clean_text.strip()

# 调用示例：打印当前目录树，忽略.git和.pyc文件
if __name__ == "__main__":
    result = print_file_tree(
        path=r"python\毕设-文件自动分类语义搜索\test", 
        ignore=[".git", "__pycache__", "*.pyc"])
    print("\n", result)
