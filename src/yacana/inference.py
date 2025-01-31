from enum import Enum
from abc import ABC, abstractmethod
import openai
from ollama import Client
from openai import OpenAI
from typing import List

from openai.types.chat import ChatCompletionToolChoiceOptionParam

from .tool import Tool
from .exceptions import IllogicalConfiguration


class ServerType(Enum):
    OLLAMA = 1
    VLLM = 2
    OPENAI = 3


class Inference(ABC):
    @abstractmethod
    def go(self, model_name: str, history: list, endpoint: str, api_token: str, model_settings: dict, stream: bool, json_output: bool, headers: dict, tools: List[Tool] | None = None):
        pass


class OllamaInference(Inference):
    def go(self, model_name: str, history: list, endpoint: str, api_token: str, model_settings: dict, stream: bool, json_output: bool, headers: dict, tools: List[Tool] | None = None):
        client = Client(host=endpoint, headers=headers)
        response = client.chat(model=model_name,
                               messages=history,
                               format=("json" if json_output else ""),
                               stream=stream,
                               options=model_settings
                               )
        return response['message']['content']


class VllmInference(Inference):
    def go(self, model_name: str, history: list, endpoint: str, api_token: str, model_settings: dict, stream: bool, json_output: bool, headers: dict, tools: List[Tool] | None = None):
        raise NotImplemented("VLLM Inference is not implemented yet")


class OpenAIInference(Inference):
    def go(self, model_name: str, history: list, endpoint: str, api_token: str, model_settings: dict, stream: bool, json_output: bool, headers: dict, tools: List[Tool] | None = None):
        refusal: bool = False
        # Extracting all json schema from tools so it can be passed to the OpenAI API
        all_function_calling_json = [tool._openai_function_schema for tool in tools] if tools else []
        tool_choice = self._find_right_tool_choice_option(tools)

        response_format = {"type": "json_object"} if json_output else {}

        client = OpenAI(
            api_key=api_token,
        )

        # @todo pour stream faudrait du code spécifique donc je ne vois pas bien comment on pourrait le faire
        completion = client.chat.completions.create(
            model=model_name,
            messages=history,
            stream=stream,
            response_format=response_format,
            tools=all_function_calling_json,
            tool_choice=tool_choice
        )
        refusal = False if completion.choices[0].message.refusal is None else True # Utile uniquement lorsqu'on utilise structured output. Permet de savoir si le modèle a refusé de répondre ou non pendant qu'on lui demande du JSON.

        print(completion.choices)
        print(completion.choices[0].message)
        print(completion.choices[0].message.content)

        return completion.choices[0].message.content

    def _find_right_tool_choice_option(self, tools: List[Tool] | None) -> ChatCompletionToolChoiceOptionParam:
        """
        iterate over all tools, then:
        If they are all optional == False then the final mode is "required"
        If they are all optional == True then the final mode is "auto"
        If there is a mix of required and optional, then raise IllogicalConfiguration() with a message explaining that mixing required and optional tools is not allowed for OpenAI.
        """
        all_optional = all(tool.optional for tool in tools)
        all_required = all(not tool.optional for tool in tools)

        if all_optional:
            return ChatCompletionToolChoiceOptionParam.auto
        elif all_required:
            return ChatCompletionToolChoiceOptionParam.required
        else:
            raise IllogicalConfiguration("OpenAI does not allow mixing required and optional tools.")

class InferenceFactory:
    @staticmethod
    def get_inference(server_type: ServerType) -> Inference:
        if server_type == ServerType.OLLAMA:
            return OllamaInference()
        elif server_type == ServerType.VLLM:
            return VllmInference()
        elif server_type == ServerType.OPENAI:
            return OpenAIInference()
        else:
            raise ValueError("Unsupported server type")