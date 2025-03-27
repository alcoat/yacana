from typing import Callable

from .modelSettings import ModelSettings
from .OllamaAgent import OllamaAgent
from .OpenAiAgent import OpenAiAgent
from .inference import ServerType


class Agent:

    def __new__(cls, name: str, model_name: str, system_prompt: str | None = None, endpoint: str = "http://127.0.0.1:11434",
                api_token: str = "", server_type=ServerType.OLLAMA, headers=None, model_settings: ModelSettings = None, streaming_callback: Callable | None = None):
        if server_type == ServerType.OPENAI:
            return OpenAiAgent(name, model_name, system_prompt=system_prompt, endpoint=endpoint, api_token=api_token, headers=headers, model_settings=model_settings, streaming_callback=streaming_callback)
        elif server_type == ServerType.OLLAMA:
            return OllamaAgent(name, model_name, system_prompt=system_prompt, endpoint=endpoint, headers=headers, model_settings=model_settings, streaming_callback=streaming_callback)
        elif server_type == ServerType.VLLM:
            raise NotImplemented()
            #return VLLMAgent(name, model_name, system_prompt, endpoint, api_token, server_type, headers, model_settings, streaming_callback=streaming_callback)
        raise ValueError("Unknown server type")

    def __init__(self, name: str, model_name: str, system_prompt: str | None = None, endpoint: str = "http://127.0.0.1:11434",
                 api_token: str = "", server_type=ServerType.OLLAMA, headers=None, model_settings: ModelSettings = None) -> None:
        pass
