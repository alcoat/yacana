from typing import List, Type, Callable, Dict
from pydantic import BaseModel

from .history import GenericMessage
from .generic_agent import GenericAgent
from .tool import Tool
from .model_settings import OpenAiModelSettings # @todo is it really the same ?


class VllmAgent(GenericAgent):

    def __init__(self, name: str, model_name: str, system_prompt: str | None = None, endpoint: str | None = None,
                 api_token: str = "", headers=None, model_settings: OpenAiModelSettings = None, streaming_callback: Callable | None = None, runtime_config: Dict | None = None, **kwargs) -> None:
        """
        Returns a new Agent
        :param name: str : Name of the agent. Can be used during conversations. Use something short and meaningful that doesn't contradict the system prompt
        :param model_name: str : Name of the LLM model that will be sent to the inference server. For instance 'llama:3.1" or 'mistral:latest' etc
        :param system_prompt: str : Defines the way the LLM will behave. For instance set the SP to "You are a pirate" to have it talk like a pirate.
        :param endpoint: str : By default will look for Ollama endpoint on your localhost. If you are using a VM with GPU then update this to the remote URL + port.
        :param api_token: str : The API token used for authentication with the inference server.
        :param headers: dict : Custom headers to be sent with the inference request. If None, an empty dictionary will be used.
        :param model_settings: ModelSettings : All settings that Ollama currently supports as model configuration. This needs to be tested with other inference servers. This allows modifying deep behavioral patterns of the LLM.
        """
        model_settings = OpenAiModelSettings() if model_settings is None else model_settings
        super().__init__(name, model_name, model_settings, system_prompt=system_prompt, endpoint=endpoint, api_token=api_token, headers=headers, streaming_callback=streaming_callback, runtime_config=runtime_config, history=kwargs.get("history", None))

    def _interact(self, task: str, tools: List[Tool], json_output: bool, structured_output: Type[BaseModel] | None, medias: List[str] | None, streaming_callback: Callable | None) -> GenericMessage:

        raise NotImplemented("WIP")