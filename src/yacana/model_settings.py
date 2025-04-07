import logging
from typing import List, Dict, Any


class OllamaModelSettings:
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
                 stop: str = None,
                 tfs_z: float = None,
                 num_predict: int = None,
                 top_k: int = None,
                 top_p: float = None) -> None:
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
        # Initialize all properties
        self._mirostat = mirostat
        self._mirostat_eta = mirostat_eta
        self._mirostat_tau = mirostat_tau
        self._num_ctx = num_ctx
        self._num_gqa = num_gqa
        self._num_gpu = num_gpu
        self._num_thread = num_thread
        self._repeat_last_n = repeat_last_n
        self._repeat_penalty = repeat_penalty
        self._temperature = temperature
        self._seed = seed
        self._stop = stop
        self._tfs_z = tfs_z
        self._num_predict = num_predict
        self._top_k = top_k
        self._top_p = top_p

        # Store the initial values for resetting
        self._initial_values = {
            'mirostat': mirostat,
            'mirostat_eta': mirostat_eta,
            'mirostat_tau': mirostat_tau,
            'num_ctx': num_ctx,
            'num_gqa': num_gqa,
            'num_gpu': num_gpu,
            'num_thread': num_thread,
            'repeat_last_n': repeat_last_n,
            'repeat_penalty': repeat_penalty,
            'temperature': temperature,
            'seed': seed,
            'stop': stop,
            'tfs_z': tfs_z,
            'num_predict': num_predict,
            'top_k': top_k,
            'top_p': top_p,
        }

    # Getter and setter for mirostat
    @property
    def mirostat(self) -> int:
        return self._mirostat

    @mirostat.setter
    def mirostat(self, value: int) -> None:
        self._mirostat = value

    # Getter and setter for mirostat_eta
    @property
    def mirostat_eta(self) -> float:
        return self._mirostat_eta

    @mirostat_eta.setter
    def mirostat_eta(self, value: float) -> None:
        self._mirostat_eta = value

    # Getter and setter for mirostat_tau
    @property
    def mirostat_tau(self) -> float:
        return self._mirostat_tau

    @mirostat_tau.setter
    def mirostat_tau(self, value: float) -> None:
        self._mirostat_tau = value

    # Getter and setter for num_ctx
    @property
    def num_ctx(self) -> int:
        return self._num_ctx

    @num_ctx.setter
    def num_ctx(self, value: int) -> None:
        self._num_ctx = value

    # Getter and setter for num_gqa
    @property
    def num_gqa(self) -> int:
        return self._num_gqa

    @num_gqa.setter
    def num_gqa(self, value: int) -> None:
        self._num_gqa = value

    # Getter and setter for num_gpu
    @property
    def num_gpu(self) -> int:
        return self._num_gpu

    @num_gpu.setter
    def num_gpu(self, value: int) -> None:
        self._num_gpu = value

    # Getter and setter for num_thread
    @property
    def num_thread(self) -> int:
        return self._num_thread

    @num_thread.setter
    def num_thread(self, value: int) -> None:
        self._num_thread = value

    # Getter and setter for repeat_last_n
    @property
    def repeat_last_n(self) -> int:
        return self._repeat_last_n

    @repeat_last_n.setter
    def repeat_last_n(self, value: int) -> None:
        self._repeat_last_n = value

    # Getter and setter for repeat_penalty
    @property
    def repeat_penalty(self) -> float:
        return self._repeat_penalty

    @repeat_penalty.setter
    def repeat_penalty(self, value: float) -> None:
        self._repeat_penalty = value

    # Getter and setter for temperature
    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        self._temperature = value

    # Getter and setter for seed
    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._seed = value

    # Getter and setter for stop
    @property
    def stop(self) -> str:
        return self._stop

    @stop.setter
    def stop(self, value: str) -> None:
        self._stop = value

    # Getter and setter for tfs_z
    @property
    def tfs_z(self) -> float:
        return self._tfs_z

    @tfs_z.setter
    def tfs_z(self, value: float) -> None:
        self._tfs_z = value

    # Getter and setter for num_predict
    @property
    def num_predict(self) -> int:
        return self._num_predict

    @num_predict.setter
    def num_predict(self, value: int) -> None:
        self._num_predict = value

    # Getter and setter for top_k
    @property
    def top_k(self) -> int:
        return self._top_k

    @top_k.setter
    def top_k(self, value: int) -> None:
        self._top_k = value

    # Getter and setter for top_p
    @property
    def top_p(self) -> float:
        return self._top_p

    @top_p.setter
    def top_p(self, value: float) -> None:
        self._top_p = value

    def reset(self) -> None:
        """
        Reset all properties to their initial values
        @return: None
        """
        for key, value in self._initial_values.items():
            setattr(self, f"_{key}", value)

    def get_settings(self) -> dict:
        """
        Returns a dictionary of all the settings and their current values, excluding None values.

        Returns
        -------
        dict
            A dictionary containing the machine settings that have been set.
        """
        settings = {
            "mirostat": self.mirostat,
            "mirostat_eta": self.mirostat_eta,
            "mirostat_tau": self.mirostat_tau,
            "num_ctx": self.num_ctx,
            "num_gqa": self.num_gqa,
            "num_gpu": self.num_gpu,
            "num_thread": self.num_thread,
            "repeat_last_n": self.repeat_last_n,
            "repeat_penalty": self.repeat_penalty,
            "temperature": self.temperature,
            "seed": self.seed,
            "stop": self.stop,
            "tfs_z": self.tfs_z,
            "num_predict": self.num_predict,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }
        return {key: value for key, value in settings.items() if value is not None}


class OpenAiModelSettings:
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
                 web_search_options: Any = None) -> None:
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
        # Initialize all properties
        self._audio = audio
        self._frequency_penalty = frequency_penalty
        self._logit_bias = logit_bias
        self._logprobs = logprobs
        self._max_completion_tokens = max_completion_tokens
        self._metadata = metadata
        self._modalities = modalities
        self._n = n
        self._prediction = prediction
        self._presence_penalty = presence_penalty
        self._reasoning_effort = reasoning_effort
        self._seed = seed
        self._service_tier = service_tier
        self._stop = stop
        self._store = store
        self._stream_options = stream_options
        self._temperature = temperature
        self._top_logprobs = top_logprobs
        self._top_p = top_p
        self._user = user
        self._web_search_options = web_search_options


        # Store the initial values for resetting
        self._initial_values = {
            'audio': audio,
            'frequency_penalty': frequency_penalty,
            'logit_bias': logit_bias,
            'logprobs': logprobs,
            'max_completion_tokens': max_completion_tokens,
            'metadata': metadata,
            'modalities': modalities,
            'n': n,
            'prediction': prediction,
            'presence_penalty': presence_penalty,
            'reasoning_effort': reasoning_effort,
            'seed': seed,
            'service_tier': service_tier,
            'stop': stop,
            'store': store,
            'stream_options': stream_options,
            'temperature': temperature,
            'top_logprobs': top_logprobs,
            'top_p': top_p,
            'user': user,
            'web_search_options': web_search_options
        }

    # Getter and setter for audio
    @property
    def audio(self) -> Any:
        return self._audio

    @audio.setter
    def audio(self, value: Any) -> None:
        self._audio = value
        logging.warning("Setting this may raise errors if the Agent is exported as this value is probably not serializable.")

    # Getter and setter for logit_bias
    @property
    def logit_bias(self) -> Dict:
        return self._logit_bias

    @logit_bias.setter
    def logit_bias(self, value: Dict) -> None:
        self._logit_bias = value

    # Getter and setter for logprobs
    @property
    def logprobs(self) -> bool:
        return self._logprobs

    @logprobs.setter
    def logprobs(self, value: bool) -> None:
        self._logprobs = value

    # Getter and setter for max_completion_tokens
    @property
    def max_completion_tokens(self) -> int:
        return self._max_completion_tokens

    @max_completion_tokens.setter
    def max_completion_tokens(self, value: int) -> None:
        self._max_completion_tokens = value

    # Getter and setter for metadata
    @property
    def metadata(self) -> Dict:
        return self._metadata

    @metadata.setter
    def metadata(self, value: Dict) -> None:
        self._metadata = value
        logging.warning("Setting this may raise errors if the Agent is exported as this value is probably not serializable.")

    # Getter and setter for modalities
    @property
    def modalities(self) -> List[str]:
        return self._modalities

    @modalities.setter
    def modalities(self, value: List[str]) -> None:
        self._modalities = value

    # Getter and setter for n
    @property
    def n(self) -> int:
        return self._n

    @n.setter
    def n(self, value: int) -> None:
        self._n = value

    # Getter and setter for prediction
    @property
    def prediction(self) -> Any:
        return self._prediction

    @prediction.setter
    def prediction(self, value: Any) -> None:
        self._prediction = value
        logging.warning("Setting this may raise errors if the Agent is exported as this value is probably not serializable.")

    # Getter and setter for reasoning_effort
    @property
    def reasoning_effort(self) -> str:
        return self._reasoning_effort

    @reasoning_effort.setter
    def reasoning_effort(self, value: str) -> None:
        self._reasoning_effort = value

    # Getter and setter for service_tier
    @property
    def service_tier(self) -> str:
        return self._service_tier

    @service_tier.setter
    def service_tier(self, value: str) -> None:
        self._service_tier = value

    # Getter and setter for stop
    @property
    def stop(self) -> str | List:
        return self._stop

    @stop.setter
    def stop(self, value: str | List) -> None:
        self._stop = value

    # Getter and setter for store
    @property
    def store(self) -> bool:
        return self._store

    @store.setter
    def store(self, value: bool) -> None:
        self._store = value

    # Getter and setter for stream_options
    @property
    def stream_options(self) -> Any:
        return self._stream_options

    @stream_options.setter
    def stream_options(self, value: Any) -> None:
        self._stream_options = value
        logging.warning("Setting this may raise errors if the Agent is exported as this value is probably not serializable.")

    # Getter and setter for top_logprobs
    @property
    def top_logprobs(self) -> int:
        return self._top_logprobs

    @top_logprobs.setter
    def top_logprobs(self, value: int) -> None:
        self._top_logprobs = value

    # Getter and setter for user
    @property
    def user(self) -> str:
        return self._user

    @user.setter
    def user(self, value: str) -> None:
        self._user = value

    # Getter and setter for web_search_options
    @property
    def web_search_options(self) -> Any:
        return self._web_search_options

    @web_search_options.setter
    def web_search_options(self, value: Any) -> None:
        self._web_search_options = value
        logging.warning("Setting this may raise errors if the Agent is exported as this value is probably not serializable.")

    # Getter and setter for frequency_penalty
    @property
    def frequency_penalty(self) -> float:
        return self._frequency_penalty

    @frequency_penalty.setter
    def frequency_penalty(self, value: float) -> None:
        self._frequency_penalty = value

    # Getter and setter for presence_penalty
    @property
    def presence_penalty(self) -> float:
        return self._presence_penalty

    @presence_penalty.setter
    def presence_penalty(self, value: float) -> None:
        self._presence_penalty = value

    # Getter and setter for seed
    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._seed = value

    # Getter and setter for temperature
    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        self._temperature = value

    # Getter and setter for top_p
    @property
    def top_p(self) -> float:
        return self._top_p

    @top_p.setter
    def top_p(self, value: float) -> None:
        self._top_p = value

    def reset(self) -> None:
        """
        Reset all properties to their initial values
        @return: None
        """
        for key, value in self._initial_values.items():
            setattr(self, f"_{key}", value)

    def get_settings(self) -> dict:
        """
        Returns a dictionary of all the settings and their current values, excluding None values.

        Returns
        -------
        dict
            A dictionary containing the machine settings that have been set.
        """
        settings = {
            "audio": self.audio,
            "frequency_penalty": self.frequency_penalty,
            "logit_bias": self.logit_bias,
            "logprobs": self.logprobs,
            "max_completion_tokens": self.max_completion_tokens,
            "metadata": self.metadata,
            "modalities": self.modalities,
            "n": self.n,
            "prediction": self.prediction,
            "presence_penalty": self.presence_penalty,
            "reasoning_effort": self.reasoning_effort,
            "seed": self.seed,
            "service_tier": self.service_tier,
            "stop": self.stop,
            "store": self.store,
            "stream_options": self.stream_options,
            "temperature": self.temperature,
            "top_logprobs": self.top_logprobs,
            "top_p": self.top_p,
            "user": self.user,
            "web_search_options": self.web_search_options
        }
        return {key: value for key, value in settings.items() if value is not None}