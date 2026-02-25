import os
import re

from constants import EXTENTION_MAP
from ai_cloud_plugin import SiliconFlowAI, OpenRouterAI
from constants import FILE_CLASSIFY_PROMPT_
from tokenization import _read_file_content, get_token

def classify_by_extension(file_path):
    """
    根据文件扩展名将文件分类。

    Args:
        file_path (str): 要分类的文件路径。

    Returns:
        str: 文件的类别。
    """
    # 获取文件的扩展名
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # 遍历映射字典，找出文件的类别
    for category, extensions in EXTENTION_MAP.items():
        if ext in extensions:
           return category
    return "其他"


def classify_by_ai(file_path):
    if os.path.exists(file_path):
        content_ = _read_file_content(file_path)
        ai_input = get_token(content_)
        ai_input = '================'.join(ai_input)

        # ai = OpenRouterAI(model="nvidia/nemotron-3-nano-30b-a3b:free",system_prompt=FILE_CLASSIFY_PROMPT_)
        ai = SiliconFlowAI(system_prompt=FILE_CLASSIFY_PROMPT_)
        resp = ai.chat_with_ai(ai_input)
        print(resp)
        pattern = r"(.*?)：(.*?)"
        match = re.search(pattern, resp)
        if match:
            category = match.group(1).strip()
        else:
            category = "其他"
        return category
