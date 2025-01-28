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
        function_calling_json = [tool._openai_function_schema for tool in tools] if tools else []
        all_optional = all(tool.optional for tool in tools)
        all_required = all(not tool.optional for tool in tools)

        if all_optional:
            tool_choice = ChatCompletionToolChoiceOptionParam.auto
        elif all_required:
            tool_choice = ChatCompletionToolChoiceOptionParam.required
        else:
            raise IllogicalConfiguration("OpenAI does not allow mixing required and optional tools.")
        """
        on tourne sur tous les tools.
        Si ils sont tous optionnal = False donc le mode final est "required"
        si ils sont tous à optional = True alors le mode final est "auto"
        si y a un mixte de required et optional alors on raise IllogicalConfiguration() avec comme message que OpenAI ne permet pas de mixer des tools required et optional.
        """
        response_format = {"type": "json_object"} if json_output else {}

        client = OpenAI(
            api_key=api_token,
        )

        completion = client.chat.completions.create(
            model=model_name,
            messages=history,
            stream=stream,
            response_format=response_format,
            tools=function_calling_json,
            tool_choice=tool_choice
        )
        refusal = False if completion.choices[0].message.refusal is None else True # Utile uniquement lorsqu'on utilise structured output. Permet de savoir si le modèle a refusé de répondre ou non pendant qu'on lui demande du JSON.

        print(completion.choices)
        print(completion.choices[0].message)
        print(completion.choices[0].message.content)

        return completion.choices[0].message.content


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