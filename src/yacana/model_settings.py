import json
import logging
from typing import List, Dict, Any
from abc import ABC, abstractmethod


class ModelSettings(ABC):

    _registry = {}

    def __init__(self):
        self._initial_values = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ModelSettings._registry[cls.__name__] = cls

    def _export(self) -> Dict:
        members = self.__dict__.copy()
        members["type"] = self.__class__.__name__
        members.pop("_initial_values", None)
        print("members = ", members)
        return members

    @staticmethod
    def create_instance(members: Dict):
        cls_name = members.pop("type")
        cls = ModelSettings._registry.get(cls_name)
        return cls(**members)

    def get_settings(self) -> dict:
        """
        Returns a dictionary of all the settings and their current values, excluding None values.

        Returns
        -------
        dict
            A dictionary containing the machine settings that have been set.
        """

        return {key: value for key, value in self.__dict__.items() if value is not None and not key.startswith("_")}

    def reset(self) -> None:
        """
        Reset all properties to their initial values
        @return: None
        """
        for key, value in self._initial_values.items():
            setattr(self, f"{key}", value)


class OllamaModelSettings(ModelSettings):
    """Class to encapsulate the settings available into the inference server.
    Note that these are all recognised by Ollama but may have no effect when using other inference servers.
    
    """

    def __init__(self,
                 mirostat: int = None,
                 mirostat_eta: float = None,
                 mirostat_tau: float = None,
                 num_ctx: int = None,
                 num_gqa: int = None,
                 num_gpu: int = None,
                 num_thread: int = None,
                 repeat_last_n: int = None,
                 repeat_penalty: float = None,
                 temperature: float = None,
                 seed: int = None,
                 stop: List[str] = None,
                 tfs_z: float = None,
                 num_predict: int = None,
                 top_k: int = None,
                 top_p: float = None,
                 **kwargs) -> None:
        """
        @param mirostat: Like a volume control for the machine’s “creativity.” (Example: 0 is off, 1 is on, 2 is extra on)
        @param mirostat_eta: Adjusts how quickly the machine learns from what it’s currently talking about. (Example: 0.1)
        @param mirostat_tau: Helps decide if the machine should stick closely to the topic. (Example: 5.0)
        @param num_ctx: Determines how much of the previous conversation the machine can remember at once. (Example: 4096)
        @param num_gqa: Like tuning how many different tasks the machine can juggle at once. (Example: 8)
        @param num_gpu: Sets how many “brains” (or parts of the computer’s graphics card) the machine uses. (Example: 50)
        @param num_thread: Determines how many separate conversations or tasks the machine can handle at the same time. (Example: 8)
        @param repeat_last_n: How much of the last part of the conversation to try not to repeat. (Example: 64)
        @param repeat_penalty: A nudge to encourage the machine to come up with something new if it starts repeating itself. (Example: 1.1)
        @param temperature: Controls how “wild” or “safe” the machine’s responses are. (Example: 0.7)
        @param seed: Sets up a starting point for generating responses. (Example: 42)
        @param stop: Tells the machine when to stop talking, based on certain cues or keywords. (Example: "AI assistant:")
        @param tfs_z: Aims to reduce randomness in the machine’s responses. (Example: 2.0)
        @param num_predict: Limits how much the machine can say in one go. (Example: 128)
        @param top_k: Limits the machine’s word choices to the top contenders. (Example: 40)
        @param top_p: Works with top_k to fine-tune the variety of the machine’s responses. (Example: 0.9)
        """
        super().__init__()
        # Initialize all properties
        self.mirostat = mirostat
        self.mirostat_eta = mirostat_eta
        self.mirostat_tau = mirostat_tau
        self.num_ctx = num_ctx
        self.num_gqa = num_gqa
        self.num_gpu = num_gpu
        self.num_thread = num_thread
        self.repeat_last_n = repeat_last_n
        self.repeat_penalty = repeat_penalty
        self.temperature = temperature
        self.seed = seed
        self.stop = stop
        self.tfs_z = tfs_z
        self.num_predict = num_predict
        self.top_k = top_k
        self.top_p = top_p

        # Store the initial values for resetting
        self._initial_values = {key: value for key, value in self.__dict__.items() if not key.startswith("_")}




class OpenAiModelSettings(ModelSettings):
    """Class to encapsulate the settings available into the inference server.
    Note that these are all recognised by Ollama but may have no effect when using other inference servers.

    """

    def __init__(self,
                 audio: Any = None,
                 frequency_penalty: float = None,
                 logit_bias: Dict = None,
                 logprobs: bool = None,
                 max_completion_tokens: int = None,
                 metadata: Dict = None,
                 modalities: List[str] = None,
                 n: int = None,
                 prediction: Any = None,
                 presence_penalty: float = None,
                 reasoning_effort: str = None,
                 seed: int = None,
                 service_tier: str = None,
                 stop: str | List = None,
                 store: bool = None,
                 stream_options: Any = None,
                 temperature: float = None,
                 top_logprobs: int = None,
                 top_p: float = None,
                 user: str = None,
                 web_search_options: Any = None,
                 **kwargs) -> None:
        """
        @param audio: Parameters for audio output. Required when audio output is requested with modalities: ["audio"]
        @param frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
        @param logit_bias: Modify the likelihood of specified tokens appearing in the completion. Accepts a JSON object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.
        @param logprobs: Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message.
        @param max_completion_tokens: An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens.
        @param metadata: Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format, and querying for objects via API or the dashboard. Keys are strings with a maximum length of 64 characters. Values are strings with a maximum length of 512 characters.
        @param modalities: Output types that you would like the model to generate. Most models are capable of generating text, which is the default: ["text"]. The gpt-4o-audio-preview model can also be used to generate audio. To request that this model generate both text and audio responses, you can use: ["text", "audio"]
        @param n: How many chat completion choices to generate for each input message. Note that you will be charged based on the number of generated tokens across all of the choices. Keep n as 1 to minimize costs.
        @param prediction: Configuration for a Predicted Output, which can greatly improve response times when large parts of the model response are known ahead of time. This is most common when you are regenerating a file with only minor changes to most of the content.
        @param presence_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
        @param reasoning_effort: o-series models only. Constrains effort on reasoning for reasoning models. Currently supported values are low, medium, and high. Reducing reasoning effort can result in faster responses and fewer tokens used on reasoning in a response.
        @param seed: This feature is in Beta. If specified, our system will make a best effort to sample deterministically, such that repeated requests with the same seed and parameters should return the same result. Determinism is not guaranteed, and you should refer to the system_fingerprint response parameter to monitor changes in the backend.
        @param service_tier: Specifies the latency tier to use for processing the request. This parameter is relevant for customers subscribed to the scale tier service: 1) If set to 'auto', and the Project is Scale tier enabled, the system will utilize scale tier credits until they are exhausted. 2) If set to 'auto', and the Project is not Scale tier enabled, the request will be processed using the default service tier with a lower uptime SLA and no latency guarentee. 3) If set to 'default', the request will be processed using the default service tier with a lower uptime SLA and no latency guarentee. 4) When not set, the default behavior is 'auto'. When this parameter is set, the response body will include the service_tier utilized.
        @parama stop: Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence.
        @param store: Whether or not to store the output of this chat completion request for use in our model distillation or evals products.
        @param stream_options: Options for streaming response.
        @param temperature: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or top_p but not both.
        @param top_logprobs: An integer between 0 and 20 specifying the number of most likely tokens to return at each token position, each with an associated log probability. logprobs must be set to true if this parameter is used.
        @param top_p: An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. We generally recommend altering this or temperature but not both.
        @param user: A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
        @param web_search_options: This tool searches the web for relevant results to use in a response.
        """
        super().__init__()
        # Initialize all properties
        self.audio = audio
        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias
        self.logprobs = logprobs
        self.max_completion_tokens = max_completion_tokens
        self.metadata = metadata
        self.modalities = modalities
        self.n = n
        self.prediction = prediction
        self.presence_penalty = presence_penalty
        self.reasoning_effort = reasoning_effort
        self.seed = seed
        self.service_tier = service_tier
        self.stop = stop
        self.store = store
        self.stream_options = stream_options
        self.temperature = temperature
        self.top_logprobs = top_logprobs
        self.top_p = top_p
        self.user = user
        self.web_search_options = web_search_options


        # Store the initial values for resetting
        self._initial_values = {key: value for key, value in self.__dict__.items() if not key.startswith("_")}

    def reset(self) -> None:
        """
        Reset all properties to their initial values
        @return: None
        """
        for key, value in self._initial_values.items():
            setattr(self, f"{key}", value)

