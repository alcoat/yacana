from enum import Enum
from abc import ABC, abstractmethod
import openai
from ollama import Client
from openai import OpenAI


class ServerType(Enum):
    OLLAMA = 1
    VLLM = 2
    OPENAI = 3


class Inference(ABC):
    @abstractmethod
    def go(self, model_name: str, history: list, endpoint: str, api_token: str, model_settings: dict, stream: bool, json_output: bool, headers: dict):
        pass


class OllamaInference(Inference):
    def go(self, model_name: str, history: list, endpoint: str, api_token: str, model_settings: dict, stream: bool, json_output: bool, headers: dict):
        client = Client(host=endpoint, headers=headers)
        response = client.chat(model=model_name,
                               messages=history,
                               format=("json" if json_output else ""),
                               stream=stream,
                               options=model_settings
                               )
        return response['message']['content']


class VllmInference(Inference):
    def go(self, model_name: str, history: list, endpoint: str, api_token: str, model_settings: dict, stream: bool, json_output: bool, headers: dict):
        raise NotImplemented("VLLM Inference is not implemented yet")


class OpenAIInference(Inference):
    def go(self, model_name: str, history: list, endpoint: str, api_token: str, model_settings: dict, stream: bool, json_output: bool, headers: dict):
        refusal: bool = False
        response_format = {"type": "json_object"} if json_output else {}

        client = OpenAI(
            api_key=api_token,
        )

        completion = client.chat.completions.create(
            model=model_name,
            messages=history,
            stream=stream,
            response_format=response_format
        )
        refusal = False if completion.choices[0].message.refusal is None else True # Utile uniquement lorsqu'on utilise structured output

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
