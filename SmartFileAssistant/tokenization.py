"""将文件或者文本内容变成token单元，下一个部件接LLM云插件"""
import os
import re
from typing import List

from ai_cloud_plugin import SiliconFlowAI
from constants import FILE_CLASSIFY_PROMPT_, FR, SUFFIX_NEED2_CLASSIFY
from my_exception import NotSupportedSuffixException

def _read_file_content(file_path: str) -> str:
    """读取文件内容"""
    file_ext = os.path.splitext(file_path)[1].lower()
    content = ""
    if file_ext not in SUFFIX_NEED2_CLASSIFY:
        raise NotSupportedSuffixException("不支持分类操作的文件类型")
    try:
        content = getattr(FR,f"read_{file_ext[1:]}_func")(file_path)
    except Exception as e:
        print(f"读取文件失败：{file_path}，错误：{str(e)}")
    
    content = content.replace("\r", "").replace("\t", " ")
    content = "\n".join([line.strip() for line in content.splitlines() if line.strip()])
    return content


def smart_sentence_split(text: str) -> List[str]:
    """增强版智能分句（处理引号、省略号、标点粘连）"""
    # 1. 预处理：标记引号内容，避免内部错误分割
    text = re.sub(r'“([^”]+)”', r' QUOTE_\1_QUOTE ', text)
    text = re.sub(r'‘([^’]+)’', r' SINGLE_QUOTE_\1_SINGLE_QUOTE ', text)
    
    # 2. 合并中文/英文省略号为统一分隔符
    sentence_delimiters = r'([。！？!?；;\n]|…{1,2})+'
    parts = re.split(sentence_delimiters, text)
    
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sentence = parts[i].strip()
        delimiter = parts[i + 1] if (i + 1 < len(parts)) else ""
        # 保留分隔符后的空格（如英文句号后接空格）
        sentences.append(f"{sentence}{delimiter.rstrip()}")
    
    # 3. 恢复引号并处理末尾句子
    sentences = [s.replace("QUOTE_", "“").replace("_QUOTE", "”") for s in sentences]
    sentences = [s.replace("SINGLE_QUOTE_", "‘").replace("_SINGLE_QUOTE", "’") for s in sentences]
    
    if parts and len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())
    
    return [s for s in sentences if s.strip()]


def get_token(text: str, max_chunk_chars: int = 1600):
        """将分句内容抽样成大段摘要，默认提供更多上下文"""
        sentences = smart_sentence_split(text)
        if not sentences:
            return []

        n = len(sentences)
        if n <= 30:
            return [_clip_chunk(' '.join(sentences), max_chunk_chars)]

        def _clamp(value: int, min_value: int, max_value: int) -> int:
            return max(min_value, min(max_value, value))

        start_count = _clamp(int(n * 0.3), 6, 40)
        end_count = _clamp(int(n * 0.3), 6, 40)
        middle_pool = sentences[start_count:n - end_count] or sentences
        middle_count = _clamp(int(n * 0.4), 12, 60)

        def _sample_middle(pool: List[str], target: int) -> List[str]:
            if len(pool) <= target:
                return pool
            step = len(pool) / target
            idx = 0.0
            sampled = []
            while len(sampled) < target:
                sampled.append(pool[int(idx)])
                idx += step
                if int(idx) >= len(pool):
                    break
            return sampled

        start_chunk = _clip_chunk(' '.join(sentences[:start_count]), max_chunk_chars)
        middle_chunk = _clip_chunk(' '.join(_sample_middle(middle_pool, middle_count)), max_chunk_chars)
        end_chunk = _clip_chunk(' '.join(sentences[-end_count:]), max_chunk_chars)
        return [start_chunk, middle_chunk, end_chunk]


def _clip_chunk(text: str, limit: int) -> str:
    return text[:limit] if limit and len(text) > limit else text
        



if __name__ == "__main__":
    TEST_FILE = r"test\毕设.md"
    
    if os.path.exists(TEST_FILE):
        print("\n" + "="*60)
        print("单文件分类测试")
        print("="*60)
        content_ = _read_file_content(TEST_FILE)
        ai_input = get_token(content_)
        ai_input = '================'.join(ai_input)

        ai = SiliconFlowAI(system_prompt=FILE_CLASSIFY_PROMPT_)
        resp = ai.chat_with_ai(ai_input)
        print(resp)

        
