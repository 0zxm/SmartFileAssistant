import os
import time
import json
import requests
from abc import ABC, abstractmethod
from typing import Optional

from constants import TEST_PROMPT_


def _require_env(value: Optional[str], env_name: str) -> str:
    """Ensure sensitive configs come from explicit args or environment."""
    resolved = value or os.getenv(env_name)
    if resolved:
        return resolved
    raise RuntimeError(
        f"环境变量 {env_name} 未配置，请在 .env 中填写并执行 'source .env' 再运行程序。"
    )

class LLMManager(ABC):
    """LLM管理器抽象基类"""
    base_url = ""

    def __init__(self, api_key: str = None, model: str = None) -> None:
        self.api_key = api_key
        self.model = model
        self.messages = []
    
    @abstractmethod
    def chat_with_ai(self, user_input: str = None) -> str:
        """与AI对话的抽象方法"""
        pass
    
    @abstractmethod
    def reset_conversation(self) -> None:
        """重置对话历史"""
        pass


class OpenRouterAI(LLMManager):
    """OpenRouter AI 实现类"""
    base_url = "https://openrouter.ai/api/v1/chat/completions"
    models = [
        "moonshotai/kimi-k2:free",
        "nex-agi/deepseek-v3.1-nex-n1:free",
        "deepseek/deepseek-r1-0528:free"
    ]
    def __init__(
        self, 
        api_key: str = None,
        model: str = None,
        system_prompt=None
    ) -> None:
        resolved_key = _require_env(api_key, "OPENROUTER_API_KEY")
        resolved_model = model or os.getenv("OPENROUTER_MODEL") or self.models[0]
        super().__init__(resolved_key, resolved_model)
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        if system_prompt:
            self.set_system_prompt(system_prompt)
    
    def _send_request(self, content: str) -> str:
        """发送请求到OpenRouter API"""
        self.messages.append({"role": "user", "content": content})
        
        try:
            response = requests.post(
                url=self.base_url,
                headers=self.headers,
                json={"model": self.model, "messages": self.messages}
            )
            
            if response.status_code == 200:
                ai_reply = response.json()["choices"][0]["message"]["content"]
                self.messages.append({"role": "assistant", "content": ai_reply})
                return ai_reply
            else:
                error_msg = f"请求失败: {response.text}"
                print(error_msg)
                # 移除失败的用户消息
                self.messages.pop()
                return None
                
        except Exception as e:
            print(f"请求异常: {str(e)}")
            # 移除失败的用户消息
            if self.messages and self.messages[-1]["role"] == "user":
                self.messages.pop()
            return None
    
    def chat_with_ai(self, user_input: str = None) -> str:
        """
        Args:
            user_input: 用户输入内容，如果为None则进入交互模式
        """
        if user_input is None:
            # 交互模式
            self._interactive_mode()
            return None
        else:
            # 单次对话模式
            return self._send_request(user_input)
    
    def _interactive_mode(self) -> None:
        """交互式对话模式"""
        print("OpenRouter 交互版（输入 'exit' 退出）")
        while True:
            user_input = input("你: ")
            if user_input.lower() == "exit":
                break
            
            ai_reply = self._send_request(user_input)
            if ai_reply:
                print("AI:", ai_reply)
    
    def reset_conversation(self) -> None:
        """重置对话历史"""
        self.messages = []
        print("对话历史已重置")
    
    def set_system_prompt(self, system_prompt: str) -> None:
        """设置系统提示词"""
        # 如果已有系统提示词，先移除
        if self.messages and self.messages[0]["role"] == "system":
            self.messages.pop(0)
        # 在开头插入新的系统提示词
        self.messages.insert(0, {"role": "system", "content": system_prompt})

    def reset_conversation(self, keep_system_prompt: bool = True) -> None:
        if keep_system_prompt and self.messages and self.messages[0]["role"] == "system":
            system_msg = self.messages[0]
            self.messages = [system_msg]
        else:
            self.messages = []
        print("对话历史已重置")


class SiliconFlowAI(LLMManager):
    """硅基流动 AI 实现类"""
    base_url = "https://api.siliconflow.cn/v1/chat/completions"
    models = [
        'ai-org/GLM-4.6',
        'Qwen/Qwen3-8B',
        'Qwen/Qwen3-14B',
        'Qwen/Qwen3-32B',
        'Qwen/Qwen3-30B-A3B',
        'Qwen/Qwen3-235B-A22B',
        'tencent/Hunyuan-A13B-Instruct',
        'zai-org/GLM-4.5V',
        'deepseek-ai/DeepSeek-V3.1-Terminus',
        'Pro/deepseek-ai/DeepSeek-V3.1-Terminus',
    ]

    def __init__(
        self, 
        api_key: str = None,
        model: str = None,
        system_prompt: str = None,
        enable_thinking: bool = False,
        stream: bool = True,
        max_tokens:int = 4096
    ) -> None:
        resolved_key = _require_env(api_key, "SILICONFLOW_API_KEY")
        resolved_model = model or os.getenv("SILICONFLOW_CHAT_MODEL") or "deepseek-ai/DeepSeek-V2.5"
        super().__init__(resolved_key, resolved_model)
        self.base_url = os.getenv("SILICONFLOW_CHAT_ENDPOINT", self.base_url)
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}"
        }
        self.enable_thinking = enable_thinking
        self.stream = stream
        self.max_tokens = max_tokens
        if system_prompt:
            self.set_system_prompt(system_prompt)

    
    def _send_request(self, content: str) -> str:
        """发送请求到硅基流动 API"""
        self.messages.append({"role": "user", "content": content})
        
        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": self.stream,
            "enable_thinking": self.enable_thinking,
            "max_tokens" : self.max_tokens
        }
        
        try:
            response = requests.post(
                url=self.base_url,
                json=payload,
                headers=self.headers,
                stream=self.stream
            )
            
            if response.status_code == 200:
                if self.stream:
                    return self._handle_stream_response(response)
                else:
                    return self._handle_normal_response(response)
            else:
                error_msg = f"请求失败，状态码：{response.status_code}"
                print(error_msg)
                # 移除失败的用户消息
                self.messages.pop()
                return None
                
        except Exception as e:
            print(f"请求异常: {str(e)}")
            # 移除失败的用户消息
            if self.messages and self.messages[-1]["role"] == "user":
                self.messages.pop()
            return None
    
    def _handle_stream_response(self, response) -> str:
        """处理流式返回"""
        full_content = ""
        full_reasoning_content = ""
        
        for chunk in response.iter_lines():
            if chunk:
                chunk_str = chunk.decode('utf-8').replace('data: ', '')
                if chunk_str != "[DONE]":
                    try:
                        chunk_data = json.loads(chunk_str)
                        delta = chunk_data['choices'][0].get('delta', {})
                        
                        # 处理普通内容
                        content = delta.get('content', '')
                        if content:
                            print(content, end="", flush=True)
                            full_content += content
                        
                        # 处理推理内容（如果enable_thinking=True）
                        reasoning_content = delta.get('reasoning_content', '')
                        if reasoning_content:
                            print(reasoning_content, end="", flush=True)
                            full_reasoning_content += reasoning_content
                    except json.JSONDecodeError:
                        continue
        
        print()
        
        # 将AI回复添加到历史
        assistant_message = {"role": "assistant", "content": full_content}
        if full_reasoning_content:
            assistant_message["reasoning_content"] = full_reasoning_content
        self.messages.append(assistant_message)
        
        return full_content

    def set_system_prompt(self, system_prompt: str) -> None:
        """设置系统提示词"""
        # 如果已有系统提示词，先移除
        if self.messages and self.messages[0]["role"] == "system":
            self.messages.pop(0)
        # 在开头插入新的系统提示词
        self.messages.insert(0, {"role": "system", "content": system_prompt})
    
    def _handle_normal_response(self, response) -> str:
        """处理非流式返回"""
        ai_reply = response.json()["choices"][0]["message"]["content"]
        print("AI:", ai_reply)
        self.messages.append({"role": "assistant", "content": ai_reply})
        return ai_reply
    
    def chat_with_ai(self, user_input: str = None) -> str:
        """
        与AI对话
        
        Args:
            user_input: 用户输入内容，如果为None则进入交互模式
            
        Returns:
            AI的回复内容
        """
        if user_input is None:
            # 交互模式
            self._interactive_mode()
            return None
        else:
            # 单次对话模式
            return self._send_request(user_input)
    
    def _interactive_mode(self) -> None:
        """交互式对话模式"""
        print("硅基流动 交互版（输入 'exit' 退出）")
        while True:
            user_input = input("\n你: ")
            if user_input.lower() == "exit":
                break
            
            print("AI: ", end="", flush=True)
            self._send_request(user_input)
    
    def reset_conversation(self, keep_system_prompt: bool = True) -> None:
        """
        重置对话历史
        
        Args:
            keep_system_prompt: 是否保留系统提示词
        """
        if keep_system_prompt and self.messages and self.messages[0]["role"] == "system":
            system_msg = self.messages[0]
            self.messages = [system_msg]
        else:
            self.messages = []
        print("对话历史已重置")
    
    
    def set_thinking_mode(self, enable_thinking: bool) -> None:
        """设置是否启用思考模式"""
        self.enable_thinking = enable_thinking
        print(f"思考模式已{'启用' if enable_thinking else '禁用'}")


class SiliconFlowEmbedding:
    """SiliconFlow 向量服务轻量封装，供RAG组件复用"""

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        # model: str = "Qwen/Qwen3-Embedding-8B",
        batch_size: int = 16,
        timeout: int = 60,
        endpoint: str = None,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        self.api_key = _require_env(api_key, "SILICONFLOW_API_KEY")
        self.model = model or os.getenv("SILICONFLOW_EMBEDDING_MODEL", "BAAI/bge-m3")
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self.endpoint = endpoint or os.getenv(
            "SILICONFLOW_EMBEDDING_URL",
            "https://api.siliconflow.cn/v1/embeddings",
        )
        self.max_retries = max(1, max_retries)
        self.retry_backoff = max(0.0, retry_backoff)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "accept": "application/json",
                "content-type": "application/json",
                "authorization": f"Bearer {self.api_key}",
            }
        )

    @staticmethod
    def _batch_items(items: list[str], batch_size: int):
        for idx in range(0, len(items), batch_size):
            yield items[idx : idx + batch_size]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts, "dimensions": 1024}
        last_error = None
        response = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.post(self.endpoint, json=payload, timeout=self.timeout)
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                last_error = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                retriable = status is None or status >= 500
                if not retriable or attempt == self.max_retries:
                    raise RuntimeError(f"SiliconFlow 向量请求失败: {exc}") from exc
                wait_seconds = self.retry_backoff * attempt
                print(f"⚠️ SiliconFlow 向量接口异常({status}), {wait_seconds:.1f}s 后重试 ({attempt}/{self.max_retries})")
                time.sleep(wait_seconds)
        if response is None:
            raise RuntimeError("SiliconFlow 向量请求失败：无法获得有效响应。")
        payload_json = response.json()
        data = payload_json.get("data", [])
        if not isinstance(data, list):
            raise RuntimeError("SiliconFlow 返回格式异常：data字段缺失或类型不正确")
        ordered_embeddings = [None] * len(texts)
        fallback_embeddings = []
        use_index_order = True
        for item in data:
            embedding = item.get("embedding")
            if embedding is None:
                raise RuntimeError("SiliconFlow 响应缺少embedding字段")
            fallback_embeddings.append(embedding)
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(texts):
                ordered_embeddings[idx] = embedding
            else:
                use_index_order = False
        if use_index_order and all(vec is not None for vec in ordered_embeddings):
            embeddings = ordered_embeddings
        else:
            embeddings = fallback_embeddings
        if len(embeddings) != len(texts):
            raise RuntimeError("SiliconFlow返回的向量数量与请求不一致")
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        for batch in self._batch_items(texts, self.batch_size):
            embeddings.extend(self._embed_batch(batch))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        if not text:
            return []
        return self.embed_documents([text])[0]


# 使用示例
if __name__ == "__main__":
    # 方式1: 创建实例并进入交互模式
    # ai = OpenRouterAI(system_prompt=TEST_PROMPT_)
    ai = SiliconFlowAI(system_prompt=TEST_PROMPT_)
    # ai.chat_with_ai()  # 交互模式
    
    # 方式2: 单次对话
    # ai = OpenRouterAI()
    # response = ai.chat_with_ai("你好，请介绍一下自己")
    # print("AI回复:", response)
    
    # 方式3: 自定义配置
    # ai = OpenRouterAI(
    #     api_key="your-api-key",
    #     model="your-model"
    # )
    # ai.set_system_prompt("你是一个专业的编程助手")
    # response = ai.chat_with_ai("如何学习Python?")
