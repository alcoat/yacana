import json
import logging
from abc import ABC, abstractmethod
from typing import List, Type, T, Callable, Dict
from pydantic import BaseModel

from .history import History, GenericMessage, MessageRole, Message
from .logging_config import LoggerManager
from .model_settings import ModelSettings
from .tool import Tool

logger = logging.getLogger(__name__)


class GenericAgent(ABC):
    """ Representation of an LLM. This class gives ways to interact with the LLM that is being assign to it.
    However, an agent should not be controlled directly but assigned to a Task(). When the task is marked as solved
    then the agent will interact with the prompt inside the task and output an answer. This class is more about
    configuring the agent than interacting with it.

    Attributes
    ----------
    name : str
        Name of the agent. Can be used during conversations. Use something short and meaningful that doesn't contradict the system prompt
    model_name : str
        Name of the LLM model that will be sent to the inference server. For instance 'llama:3.1" or 'mistral:latest' etc
    system_prompt : str
        Defines the way the LLM will behave. For instance set the SP to "You are a pirate" to have it talk like a pirate.
    model_settings: ModelSettings
        All settings that Ollama currently supports as model configuration. This needs to be tested with other inference servers. This allows modifying deep behavioral patterns of the LLM.
    endpoint: str
        By default will look for Ollama endpoint on your localhost. If you are using a VM with GPU then update this to the remote URL + port.
    history: History
        The whole conversation that is sent to the inference server. It contains the alternation of message between the prompts in the task that are given to the LLM and its answers.

    Methods
    ----------
    simple_chat(custom_prompt: str = "> ", stream: bool = True) -> None
    export_to_file(self, file_path: str) -> None

    ClassMethods
    ----------
    get_agent_from_state(file_path: str) -> 'Agent'
    """

    _registry = {}

    def __init__(self, name: str, model_name: str, model_settings: ModelSettings, system_prompt: str | None = None, endpoint: str | None = None,
                 api_token: str = "", headers=None, streaming_callback: Callable | None = None, runtime_config: Dict | None = None, history: History | None = None, task_runtime_config: Dict | None = None) -> None:
        """
        Returns a new Agent
        :param name: str : Name of the agent. Can be used during conversations. Use something short and meaningful that doesn't contradict the system prompt
        :param model_name: str : Name of the LLM model that will be sent to the inference server. For instance 'llama:3.1" or 'mistral:latest' etc
        :param system_prompt: str : Defines the way the LLM will behave. For instance set the SP to "You are a pirate" to have it talk like a pirate.
        :param endpoint: str : By default will look for Ollama endpoint on your localhost. If you are using a VM with GPU then update this to the remote URL + port.
        :param api_token: str : The API token used for authentication with the inference server.
        :param server_type: ServerType : The type of server to use for inference. Options are ServerType.OLLAMA, ServerType.VLLM, and ServerType.OPENAI.
        :param headers: dict : Custom headers to be sent with the inference request. If None, an empty dictionary will be used.
        :param model_settings: ModelSettings : All settings that Ollama currently supports as model configuration. This needs to be tested with other inference servers. This allows modifying deep behavioral patterns of the LLM.
        """

        self.name: str = name
        self.model_name: str = model_name
        self.system_prompt: str | None = system_prompt
        if model_settings is None:
            raise ValueError("model_settings cannot be None. Please provide a valid ModelSettings instance.")
        self.model_settings: ModelSettings = model_settings
        self.api_token: str = api_token
        self.headers = {} if headers is None else headers
        self.endpoint: str | None = endpoint
        self.streaming_callback: Callable | None = streaming_callback
        self.runtime_config = runtime_config if runtime_config is not None else {}
        self.task_runtime_config = task_runtime_config if task_runtime_config is not None else {}

        self.history: History = history if history is not None else History()
        if self.system_prompt is not None and history is None:
            self.history.add_message(Message(MessageRole.SYSTEM, system_prompt))

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        GenericAgent._registry[cls.__name__] = cls

    def export_to_file(self, file_path: str, save_token=True) -> None:
        """
        Exports the current agent configuration to a file. This contains all the agents data and history.
        This means that you can use the @get_agent_from_state method to load this agent back again and continue where
        you left off.
        WARNING : THIS WILL LEAK API_KEYS !!!
        @param file_path: str: Path of the file in which you wish the data to be saved. Specify the path + filename. Be wary when using relative path.
        @return:
        """
        if save_token is True:
            logging.warning("Saving the agent state will leak API keys and other sensitive information to the destination file. Set 'save_token' to False to avoid this.")
        members_as_dict = self.__dict__.copy()
        members_as_dict["type"] = self.__class__.__name__
        members_as_dict["model_settings"] = self.model_settings._export() #self.model_settings.get_settings()
        members_as_dict["history"] = self.history._export()

        with open(file_path, 'w') as file:
            json.dump(members_as_dict, file, indent=4)
        if self.streaming_callback is not None:
            logging.info("Streaming callbacks cannot be exported. Please reassign the streaming callback after loading the agent from state.")
        logging.info("Agent state successfully exported to %s", file_path)

    @classmethod
    def import_from_file(cls, file_path: str) -> 'GenericAgent':
        """
        Loads the state previously exported from the @export_state() method. This will return an Agent in the same state
        as it was before it was saved allowing you to resume the agent conversation even after the program has exited.
        @param file_path: str : The path from the file from which to load the Agent.
        @return: Agent : A newly created Agent that is a copy from disk of a previously exported agent
        """
        with open(file_path, 'r') as file:
            members: Dict = json.load(file)

        cls_name = members.pop("type")
        members["model_settings"] = ModelSettings.create_instance(members["model_settings"])
        members["history"] = History.create_instance(members["history"])
        cls = GenericAgent._registry.get(cls_name)
        return cls(**members)

    @abstractmethod
    def _interact(self, task: str, tools: List[Tool], json_output: bool, structured_output: Type[BaseModel] | None, medias: List[str] | None, streaming_callback: Callable | None, task_runtime_config: Dict | None) -> GenericMessage:
        raise NotImplemented(f"This method must be subclassed by the child class. It starts the inference using given parameters.")

