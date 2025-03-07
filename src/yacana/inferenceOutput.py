from abc import ABC, abstractmethod
from enum import Enum
from typing import Type, TypeVar, Any, List

from src.yacana import Message

T = TypeVar("T")


class InferenceOutputType(Enum):
    CHAT = 1
    STRUCTURED_OUTPUT = 2
    TOOL_CALLING = 3


class InferenceOutput(ABC):
    def __init__(self, raw_llm_response: str):
        self._raw_llm_response: str = raw_llm_response
        self._messages: List[Message] = []

    @property
    def raw_llm_response(self) -> str:
        return self._raw_llm_response

    @raw_llm_response.setter
    def raw_llm_response(self, value: str):
        self._raw_llm_response = value

    @property
    def messages(self) -> List[Message]:
        return self._messages

    @messages.setter
    def messages(self, value: List[Message]):
        self._messages = value


class InferenceProcessor(ABC):

    @abstractmethod
    def process_choice(self, choice: Any) -> List[Message]:
        raise NotImplementedError("Subclasses should implement this method.")


class ChatOutput(InferenceOutput):

    def __init__(self, raw_llm_response: str, processor: InferenceProcessor):
        super().__init__(raw_llm_response)
        self.messages: List[Message] = processor.process_choice(raw_llm_response)


class StructuredOutput(InferenceOutput):
    def __init__(self, raw_llm_response: str, structured_output: Type[T]):
        super().__init__(raw_llm_response)
        self.structured_output: Type[T] = structured_output

    @property
    def inference_type(self) -> InferenceOutputType:
        return InferenceOutputType.STRUCTURED_OUTPUT


class ToolCallingOutput(InferenceOutput):
    def __init__(self, raw_llm_response: str, tool_call_id: str):
        super().__init__(raw_llm_response)
        self.tool_call_id: str = tool_call_id

    @property
    def inference_type(self) -> InferenceOutputType:
        return InferenceOutputType.TOOL_CALLING
