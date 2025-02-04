from enum import Enum
from abc import ABC, abstractmethod
import openai
from ollama import Client
from openai import OpenAI
from typing import List, Type, Dict, Any, T, Literal

from openai.types.chat import ChatCompletionToolChoiceOptionParam
from pydantic import BaseModel

from .tool import Tool
from .exceptions import IllogicalConfiguration, TaskCompletionRefusal


class ServerType(Enum):
    OLLAMA = 1
    VLLM = 2
    OPENAI = 3


class Inference(ABC):
    @abstractmethod
    def go(self, model_name: str, history: list, endpoint: str, api_token: str, model_settings: dict, stream: bool, json_output: bool, structured_output: Type[T] | None, headers: dict, tools: List[Tool] | None) -> T | str:
        pass


class OllamaInference(Inference):

    def get_expected_output_format(self, json_output: bool, structured_output: Type[BaseModel] | None) -> dict[str, Any] | str:
        if structured_output:
            return structured_output.model_json_schema()
        elif json_output:
            return 'json'
        else:
            return ''

    def go(self, model_name: str, history: list, endpoint: str, api_token: str, model_settings: dict, stream: bool, json_output: bool, structured_output: Type[T] | None, headers: dict, tools: List[Tool] | None) -> (str, T):
        client = Client(host=endpoint, headers=headers)
        #print("valeur de ca = ", self.get_expected_output_format(json_output, structured_output))
        response = client.chat(model=model_name,
                               messages=history,
                               format=self.get_expected_output_format(json_output, structured_output),
                               stream=stream,
                               options=model_settings
                               )
        print("message = ", response['message']['content'])
        print(T)
        print(type(T))
        if structured_output is None:
            return response['message']['content'], None
        else:
            return response['message']['content'], structured_output.model_validate_json(response['message']['content'])


class VllmInference(Inference):
    def go(self, model_name: str, history: list, endpoint: str, api_token: str, model_settings: dict, stream: bool, json_output: bool, structured_output: Type[T] | None, headers: dict, tools: List[Tool] | None) -> (str, T):
        raise NotImplemented("VLLM Inference is not implemented yet")


class OpenAIInference(Inference):
    def go(self, model_name: str, history: list, endpoint: str, api_token: str, model_settings: dict, stream: bool, json_output: bool, structured_output: Type[T] | None, headers: dict, tools: List[Tool] | None) -> (str, T):

        print("fu tools", tools)
        # Extracting all json schema from tools, so it can be passed to the OpenAI API
        all_function_calling_json = [tool._openai_function_schema for tool in tools] if tools else []

        tool_choice_option = self._find_right_tool_choice_option(tools)
        if structured_output is not None:
            response_format = structured_output
        elif json_output is True:
            response_format = {"type": "json_object"}  # This is not the structured output feature, but only "best effort" to get a JSON object (as string)
        else:
            response_format = None

        client = OpenAI(
            api_key=api_token,
        )

        # @todo pour stream faudrait du code spécifique donc je ne vois pas bien comment on pourrait le faire
        # @todo modelsettings

        params = {
            "model": model_name,
            "messages": history,
            **({"response_format": response_format} if response_format is not None else {}),
            **({"tools": all_function_calling_json} if len(all_function_calling_json) > 0 else {}),
            **({"tool_choice": tool_choice_option} if len(all_function_calling_json) > 0 else {})

        }
        # **({"stream": stream} if structured_output is None else {})
        if structured_output is None:
            completion = client.chat.completions.create(**params)
            return completion.choices[0].message.content, None
        else:
            print(f"model_name: {model_name}, history: {history}, endpoint: {endpoint}, api_token: {api_token}, model_settings: {model_settings}, stream: {stream}, json_output: {json_output}, structured_output: {structured_output}, headers: {headers}, tools: {tools}")
            completion = client.beta.chat.completions.parse(**params)
            if completion.choices[0].message.refusal is not None:
                raise TaskCompletionRefusal(completion.choices[0].message.refusal)  # Refusal is only available for structured output and doesn't work very well
            return completion.choices[0].message.content, completion.choices[0].message.parsed

    def _find_right_tool_choice_option(self, tools: List[Tool] | None) -> Literal["none", "auto", "required"]:
        """
        iterate over all tools, then:
        If they are all optional == False then the final mode is "required"
        If they are all optional == True then the final mode is "auto"
        If there is a mix of required and optional, then raise IllogicalConfiguration() with a message explaining that mixing required and optional tools is not allowed for OpenAI.
        """
        if tools is None:
            return "none"

        all_optional = all(tool.optional for tool in tools)
        all_required = all(not tool.optional for tool in tools)

        if all_optional:
            return "auto"
        elif all_required:
            return "required"
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