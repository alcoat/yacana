import json
import logging
import os
from enum import Enum
from abc import ABC, abstractmethod
from ollama import Client, ChatResponse
from openai import OpenAI
from typing import List, Type, Any, Literal, T, Dict
from collections.abc import Iterator
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel

from .history import HistorySlot, Message, MessageRole, ToolCall
from .tool import Tool
from .exceptions import IllogicalConfiguration, TaskCompletionRefusal


class ServerType(Enum):
    OLLAMA = 1
    VLLM = 2
    OPENAI = 3

"""
class InferenceOutputType(Enum):
    CHAT = 1
    STRUCTURED_OUTPUT = 2
    TOOL_CALLING = 3


class InferenceOutput:
    def __init__(self, raw_llm_response: str, structured_output: Type[T] | None, message_content: str | None = None, tool_call_id: str | None = None):
        self.raw_llm_response: str = raw_llm_response
        self.structured_output: Type[T] = structured_output
        self.message_content: str | None = message_content
        self.tool_call_id: str | None = tool_call_id

    def __str__(self):
        return self.raw_llm_response
"""


class InferenceServer(ABC):
    @abstractmethod
    def go(self, model_name: str, history: list, endpoint: str, api_token: str, model_settings: dict, stream: bool, json_output: bool, structured_output: Type[T] | None, headers: dict, tools: List[Tool] | None = None, images: List[str] | None = None) -> HistorySlot:
        pass


class OllamaInference(InferenceServer):

    @staticmethod
    def _get_expected_output_format(json_output: bool, structured_output: Type[BaseModel] | None) -> dict[str, Any] | str:
        if structured_output:
            return structured_output.model_json_schema()
        elif json_output:
            return 'json'
        else:
            return ''

    def _response_to_json(self, response: Any) -> str:
        try:
            result: Dict[str, Any] = {
                'model': getattr(response, 'model', None),
                'created_at': getattr(response, 'created_at', None),
                'done': getattr(response, 'done', None),
                'done_reason': getattr(response, 'done_reason', None),
                'total_duration': getattr(response, 'total_duration', None),
                'load_duration': getattr(response, 'load_duration', None),
                'prompt_eval_count': getattr(response, 'prompt_eval_count', None),
                'prompt_eval_duration': getattr(response, 'prompt_eval_duration', None),
                'eval_count': getattr(response, 'eval_count', None),
                'eval_duration': getattr(response, 'eval_duration', None),
            }

            # Extract 'message' if present
            message = getattr(response, 'message', None)
            if message is not None:
                result['message'] = {
                    'role': getattr(message, 'role', None),
                    'content': getattr(message, 'content', None),
                    'images': getattr(message, 'images', None),
                    'tool_calls': getattr(message, 'tool_calls', None),
                }

            # Return the JSON string representation
            return json.dumps(result, indent=4)
        except Exception as e:
            raise TypeError(f"Failed to convert response to JSON: {e}")

    def go(self, model_name: str, history: list, endpoint: str, api_token: str, model_settings: dict, stream: bool, json_output: bool, structured_output: Type[T] | None, headers: dict, tools: List[Tool] | None = None, images: List[str] | None = None) -> HistorySlot:
        history_slot = HistorySlot()
        client = Client(host=endpoint, headers=headers)
        response = client.chat(model=model_name,
                               messages=history,
                               format=OllamaInference._get_expected_output_format(json_output, structured_output),
                               stream=stream,
                               options=model_settings,
                               )
        if structured_output is None:
            history_slot.add_message(Message(MessageRole.ASSISTANT, response['message']['content'], tool_call_id="", is_yacana_builtin=True))
        else:
            history_slot.add_message(Message(MessageRole.ASSISTANT, str(response['message']['content']), structured_output=structured_output.model_validate_json(response['message']['content'])))

        history_slot.set_raw_llm_json(self._response_to_json(response))
        return history_slot

class VllmInference(InferenceServer):
    def go(self, model_name: str, history: list, endpoint: str, api_token: str, model_settings: dict, stream: bool, json_output: bool, structured_output: Type[T] | None, headers: dict, tools: List[Tool] | None = None, images: List[str] | None = None) -> HistorySlot:
        raise NotImplemented("VLLM Inference is not implemented yet")


class OpenAIInference(InferenceServer):

    def is_structured_output(self, choice: Choice) -> bool:
        return hasattr(choice.message, "parsed") and choice.message.parsed is not None

    def is_tool_calling(self, choice: Choice) -> bool:
        return hasattr(choice.message, "tool_calls") and choice.message.tool_calls is not None and len(choice.message.tool_calls) > 0

    def is_common_chat(self, choice: Choice) -> bool:
        return hasattr(choice.message, "content") and choice.message is not None

    def go(self, model_name: str, history: list, endpoint: str, api_token: str, model_settings: dict, stream: bool, json_output: bool, structured_output: Type[T] | None, headers: dict, tools: List[Tool] | None = None, images: List[str] | None = None) -> HistorySlot:

        print(f"inference : model_name: {model_name}, history: {history}, endpoint: {endpoint}, api_token: {api_token}, model_settings: {model_settings}, stream: {stream}, json_output: {json_output}, structured_output: {structured_output}, headers: {headers}, tools: {str(tools)}")
        # Extracting all json schema from tools, so it can be passed to the OpenAI API
        all_function_calling_json = [tool._openai_function_schema for tool in tools] if tools else []

        tool_choice_option = self._find_right_tool_choice_option(tools)
        response_format = self._find_right_output_format(structured_output, json_output)

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
        print("tool choice = ", tool_choice_option)
        print("----")
        print("current params = ", json.dumps(params, indent=2))
        print(f"model_name: {model_name}, history: {history}, endpoint: {endpoint}, api_token: {api_token}, model_settings: {model_settings}, stream: {stream}, json_output: {json_output}, structured_output: {structured_output}, headers: {headers}")
        print("----")

        history_slot = HistorySlot()
        if structured_output is None:
            completion = client.chat.completions.create(**params)
        else:  # Using structured output
            completion = client.beta.chat.completions.parse(**params)

        history_slot.set_raw_llm_json(completion.model_dump_json())

        print("Résultat de l'inférence quelle quelle soit = ")
        print(completion.model_dump_json(indent=2))
        logging.debug("Inference output: %s", completion.model_dump_json(indent=2))

        for choice in completion.choices:
            print("boucle !")

            if self.is_structured_output(choice):
                print("This is a structured_output answer.")
                logging.debug("Response assessment is structured output")
                if choice.message.refusal is not None:
                    raise TaskCompletionRefusal(choice.message.refusal)  # Refusal key is only available for structured output but also doesn't work very well
                history_slot.add_message(Message(MessageRole.ASSISTANT, choice.message.content, structured_output=choice.message.parsed, is_yacana_builtin=True))

            elif self.is_tool_calling(choice):
                print("This is a tool_calling answer.")
                logging.debug("Response assessment is tool calling")
                tool_calls: List[ToolCall] = []  # @todo on pourait peut etre renomer ToolCall en InferencedToolCall pour montrer que c'est le résultat d'une inférence et pas un truc qu'on donne au départ. A voir pour le nom.
                for tool_call in choice.message.tool_calls:
                    tool_calls.append(ToolCall(tool_call.id, tool_call.function.name, json.loads(tool_call.function.arguments)))
                    print("tool info = ", tool_call.id, tool_call.function.name, tool_call.function.arguments)
                history_slot.add_message(Message(MessageRole.ASSISTANT, None, tool_calls=tool_calls, is_yacana_builtin=True))

            elif self.is_common_chat(choice):
                print("this is a classic chat answer.")
                logging.debug("Response assessment is classic chat answer")
                history_slot.add_message(Message(MessageRole.ASSISTANT, choice.message.content, is_yacana_builtin=True))
            else:
                raise ValueError("Unknown response from OpenAI API") # @todo error custom

        return history_slot

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

    def _find_right_output_format(self, structured_output: Type[T] | None, json_output: bool) -> Any:
        if structured_output is not None:
            return structured_output
        elif json_output is True:
            return {"type": "json_object"}  # This is NOT the "structured output" feature, but only "best effort" to get a JSON object (as string)
        else:
            return None


class InferenceFactory:
    @staticmethod
    def get_inference(server_type: ServerType) -> InferenceServer:
        if server_type == ServerType.OLLAMA:
            return OllamaInference()
        elif server_type == ServerType.VLLM:
            return VllmInference()
        elif server_type == ServerType.OPENAI:
            return OpenAIInference()
        else:
            raise ValueError("Unsupported server type")