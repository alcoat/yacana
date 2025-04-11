import logging
from typing import Callable
from enum import Enum

from .model_settings import OllamaModelSettings, OpenAiModelSettings
from .ollama_agent import OllamaAgent
from .open_ai_agent import OpenAiAgent
from .generic_agent import GenericAgent


class ServerType(Enum):
    OLLAMA = 1
    VLLM = 2
    OPENAI = 3


class Agent(GenericAgent):

    def __new__(cls, name: str, model_name: str, system_prompt: str | None = None, endpoint: str = "http://127.0.0.1:11434",
                api_token: str = "", server_type=ServerType.OLLAMA, headers=None, model_settings: OllamaModelSettings | OpenAiModelSettings = None, streaming_callback: Callable | None = None):
        logging.info("Deprecation notice: Use specialized Agents class instead. For instance OllamaAgent() or OpenAiAgent(), etc")
        if server_type == ServerType.OPENAI:
            return OpenAiAgent(name, model_name, system_prompt=system_prompt, endpoint=endpoint, api_token=api_token, headers=headers, model_settings=model_settings, streaming_callback=streaming_callback)
        elif server_type == ServerType.OLLAMA:
            return OllamaAgent(name, model_name, system_prompt=system_prompt, endpoint=endpoint, headers=headers, model_settings=model_settings, streaming_callback=streaming_callback)
        elif server_type == ServerType.VLLM:
            raise NotImplemented()
            #return VLLMAgent(name, model_name, system_prompt, endpoint, api_token, server_type, headers, model_settings, streaming_callback=streaming_callback)
        raise ValueError("Unknown server type")

    def __init__(self, name: str, model_name: str, system_prompt: str | None = None, endpoint: str = "http://127.0.0.1:11434",
                 api_token: str = "", server_type=ServerType.OLLAMA, headers=None, model_settings: OllamaModelSettings | OpenAiModelSettings = None) -> None:
        pass
