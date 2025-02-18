from abc import ABC, abstractmethod
from enum import Enum
from typing import Type, TypeVar

T = TypeVar("T")


class InferenceOutputType(Enum):
    CHAT = 1
    STRUCTURED_OUTPUT = 2
    TOOL_CALLING = 3


class InferenceOutput(ABC):
    def __init__(self, raw_llm_response: str):
        self.raw_llm_response: str = raw_llm_response

    def __str__(self):
        return self.raw_llm_response

    @property
    @abstractmethod
    def inference_type(self) -> InferenceOutputType:
        raise NotImplementedError("Subclasses must implement inference_type")


class ChatOutput(InferenceOutput):
    def __init__(self, raw_llm_response: str, message_content: str):
        super().__init__(raw_llm_response)
        self.message_content: str = message_content

    @property
    def inference_type(self) -> InferenceOutputType:
        return InferenceOutputType.CHAT


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
