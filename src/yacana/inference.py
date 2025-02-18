import json
import uuid
from enum import Enum
from abc import ABC, abstractmethod
import openai
from ollama import Client
from openai import OpenAI
from typing import List, Type, Dict, Any, Literal, T

from openai.types.chat import ChatCompletionToolChoiceOptionParam
from pydantic import BaseModel

from .inferenceOutput import ChatOutput, StructuredOutput, InferenceOutput, ToolCallingOutput
from .tool import Tool
from .exceptions import IllogicalConfiguration, TaskCompletionRefusal


class ServerType(Enum):
    OLLAMA = 1
    VLLM = 2
    OPENAI = 3


class InferenceOutputType(Enum):
    CHAT = 1
    STRUCTURED_OUTPUT = 2
    TOOL_CALLING = 3

"""
class InferenceOutput:
    def __init__(self, raw_llm_response: str, structured_output: Type[T] | None, message_content: str | None = None, tool_call_id: str | None = None):
        self.raw_llm_response: str = raw_llm_response
        self.structured_output: Type[T] = structured_output
        self.message_content: str | None = message_content
        self.tool_call_id: str | None = tool_call_id

    def __str__(self):
        return self.raw_llm_response
"""

class Inference(ABC):
    @abstractmethod
    def go(self, model_name: str, history: list, endpoint: str, api_token: str, model_settings: dict, stream: bool, json_output: bool, structured_output: Type[T] | None, headers: dict, tools: List[Tool] | None, images: List[str] | None = None) -> InferenceOutput:
        pass


class OllamaInference(Inference):

    @staticmethod
    def _get_expected_output_format(json_output: bool, structured_output: Type[BaseModel] | None) -> dict[str, Any] | str:
        if structured_output:
            return structured_output.model_json_schema()
        elif json_output:
            return 'json'
        else:
            return ''

    def go(self, model_name: str, history: list, endpoint: str, api_token: str, model_settings: dict, stream: bool, json_output: bool, structured_output: Type[T] | None, headers: dict, tools: List[Tool] | None, images: List[str] | None = None) -> InferenceOutput:
        client = Client(host=endpoint, headers=headers)
        response = client.chat(model=model_name,
                               messages=history,
                               format=OllamaInference._get_expected_output_format(json_output, structured_output),
                               stream=stream,
                               options=model_settings,
                               images=images
                               )
        if structured_output is None:
            return ChatOutput(raw_llm_response=response['message']['content'], message_content=response['message']['content']) #InferenceOutput(raw_llm_response=response['message']['content'], structured_output=None, tool_call_id=None)
        else:
            return StructuredOutput(raw_llm_response=response['message']['content'], structured_output=structured_output.model_validate_json(response['message']['content'])) #InferenceOutput(raw_llm_response=response['message']['content'], structured_output=structured_output.model_validate_json(response['message']['content']), tool_call_id=None)


class VllmInference(Inference):
    def go(self, model_name: str, history: list, endpoint: str, api_token: str, model_settings: dict, stream: bool, json_output: bool, structured_output: Type[T] | None, headers: dict, tools: List[Tool] | None, images: List[str] | None = None) -> InferenceOutput:
        raise NotImplemented("VLLM Inference is not implemented yet")


class OpenAIInference(Inference):
    def go(self, model_name: str, history: list, endpoint: str, api_token: str, model_settings: dict, stream: bool, json_output: bool, structured_output: Type[T] | None, headers: dict, tools: List[Tool] | None, images: List[str] | None = None) -> InferenceOutput:

        # Extracting all json schema from tools, so it can be passed to the OpenAI API
        all_function_calling_json = [tool._openai_function_schema for tool in tools] if tools else []

        #print("ca devrait etre du json = ", all_function_calling_json)

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
        # @todo faut gérer les choices autres [0]

        #print("current history = ", history)

        params = {
            "model": model_name,
            "messages": history,
            **({"response_format": response_format} if response_format is not None else {}),
            **({"tools": all_function_calling_json} if len(all_function_calling_json) > 0 else {}),
            **({"tool_choice": tool_choice_option} if len(all_function_calling_json) > 0 else {})
        }
        print("tool choice = ", tool_choice_option)
        print("----")
        print("current params = ", json.dumps(params, indent=2))
        print("----")
        if structured_output is None:
            completion = client.chat.completions.create(**params)
            if tools is not None and len(tools) > 0:
                print("retour de l'autre noob = ", completion.choices)
                function_calling_answer = []

                #print("huuuuuuummm", completion.model_dump_json())

                for tool_call in completion.choices[0].message.tool_calls:
                    function_calling_answer.append({
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments)
                        }
                    })
                print("saloperie = ", json.loads(completion.model_dump_json())["choices"][0]["message"])
                # We return the function calling as a JSON string and there is no structured output involved
                return ToolCallingOutput(raw_llm_response=json.dumps(json.loads(completion.model_dump_json())["choices"][0]["message"]), tool_call_id=tool_call.id)
                # -> InferenceOutput(raw_llm_response=json.dumps(json.loads(completion.model_dump_json())["choices"][0]["message"]), structured_output=None, tool_call_id=tool_call.id) # @todo PB ICI ! Le tool_call c'est une instance de boucle... Donc quand il y a pls fonction qui sont retournées elles ont chacune leur tool_call_id. Et j'aurais besoin de savoir quel tool call correspond à quel id pour ensuite pouvoir uploder la réponse du tool dans l'historique. En gros l'id du tool doit matcher la fonction.
            else:
                # No tools were given, so we return the classic completion and no structured output is involved
                return ChatOutput(raw_llm_response=completion.choices[0].message.content, message_content=completion.choices[0].message.content) #InferenceOutput(raw_llm_response=completion.choices[0].message.content, structured_output=None, tool_call_id=None)
        else:  # Using structured output
            print(f"model_name: {model_name}, history: {history}, endpoint: {endpoint}, api_token: {api_token}, model_settings: {model_settings}, stream: {stream}, json_output: {json_output}, structured_output: {structured_output}, headers: {headers}, tools: {tools}")
            completion = client.beta.chat.completions.parse(**params)
            if completion.choices[0].message.refusal is not None:
                raise TaskCompletionRefusal(completion.choices[0].message.refusal)  # Refusal is only available for structured output and doesn't work very well
            # We return the structured output as raw text and as structured output (but no tool calling is involved)
            return StructuredOutput(raw_llm_response=completion.choices[0].message.content, structured_output=completion.choices[0].message.parsed) #InferenceOutput(raw_llm_response=completion.choices[0].message.content, structured_output=completion.choices[0].message.parsed, is_function_calling=False)

    def _find_right_tool_choice_option(self, tools: List[Tool] | None) -> Literal["none", "auto", "required"]:
        """
        iterate over all tools, then:
        If they are all 'optional == False' then the final mode is "required"
        If they are all 'optional == True' then the final mode is "auto"
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