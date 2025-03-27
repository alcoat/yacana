import json
import logging
from abc import ABC, abstractmethod
from typing import List, Type, T, Callable
from pydantic import BaseModel

from .history import History, GenericMessage, MessageRole, Message, OllamaTextMessage
from .logging_config import LoggerManager
from .modelSettings import ModelSettings
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
    export_state(self, file_path: str) -> None

    ClassMethods
    ----------
    get_agent_from_state(file_path: str) -> 'Agent'
    """

    def __init__(self, name: str, model_name: str, system_prompt: str | None = None, endpoint: str | None = None,
                 api_token: str = "", headers=None, model_settings: ModelSettings = None, streaming_callback: Callable | None = None) -> None:
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
        self.model_settings: ModelSettings = ModelSettings() if model_settings is None else model_settings
        self.api_token: str = api_token
        self.headers = {} if headers is None else headers
        self.endpoint: str | None = endpoint
        self.streaming_callback: Callable | None = streaming_callback

        self.history: History = History()
        if self.system_prompt is not None:
            self.history.add_message(Message(MessageRole.SYSTEM, system_prompt))


    def simple_chat(self, custom_prompt: str = "> ", stream: bool = True) -> None:
        """
        Use for testing but this is not how the framework is intended to be used. It creates a simple chatbot that
        keeps track of the history.
        @param custom_prompt: str : Set the prompt style for user input
        @param stream: bool : If set to True you will see the result of your output as the LLM generates tokens instead of waiting for it to complete.
        @return: None
        """
        LoggerManager.set_log_level(None)
        print("Type 'quit' then enter to exit.")

        while True:
            user_query: str = input(custom_prompt)
            if user_query == "quit":
                break
            llm_response: str = self._chat(self.history, user_query, stream=stream)
            if stream is True:
                complete_llm_answer: List[str] = []
                for chunk in llm_response:
                    print(chunk['message']['content'], end='', flush=True)
                    complete_llm_answer.append(chunk['message']['content'])
                print("")
                self.history.add_message(OllamaTextMessage(MessageRole.ASSISTANT, "".join(complete_llm_answer), is_yacana_builtin=True))
            else:
                print(llm_response)

    def export_state(self, file_path: str) -> None:
        """
        Exports the current agent configuration to a file. This contains all the agents data and history.
        This means that you can use the @get_agent_from_state method to load this agent back again and continue where
        you left off.
        WARNING : THIS WILL LEAK API_KEYS !!!
        @param file_path: str: Path of the file in which you wish the data to be saved. Specify the path + filename. Be wary when using relative path.
        @return:
        """
        final: dict = {
            "name": self.name,
            "model_name": self.model_name,
            "system_prompt": self.system_prompt,
            "model_settings": self.model_settings.get_settings(),
            "api_token": self.api_token,
            "server_type": self.server_type.name,
            "custom_headers": self.headers,
            "endpoint": self.endpoint,
            "history": self.history._export()
        }
        with open(file_path, 'w') as file:
            json.dump(final, file, indent=4)

    @classmethod
    def get_agent_from_state(cls, file_path: str) -> 'GenericAgent':
        """
        Loads the state previously exported from the @export_state() method. This will return an Agent in the same state
        as it was before it was saved allowing you to resume the agent conversation even after the program has exited.
        @param file_path: str : The path from the file from which to load the Agent.
        @return: Agent : A newly created Agent that is a copy from disk of a previously exported agent
        """
        with open(file_path, 'r') as file:
            state = json.load(file)

        model_settings = ModelSettings(**state['model_settings'])
        history = History()
        history._load_as_dict(state['history'])

        agent = cls(
            name=state['name'],
            model_name=state['model_name'],
            system_prompt=state.get('system_prompt'),
            endpoint=state['endpoint'],
            api_token=state['api_token'],
            server_type=state['server_type'],
            headers=state['custom_headers'],
            model_settings=model_settings
        )

        agent.history = history
        return agent

    @abstractmethod
    def _interact(self, task: str, tools: List[Tool], json_output: bool, structured_output: Type[BaseModel] | None, medias: List[str] | None, streaming_callback: Callable | None) -> GenericMessage:
        raise NotImplemented(f"This method must be subclassed by the child class. It starts the inference using given parameters.")





"""
    def _chat(self, history: History, task: str | None, medias: List[str] | None = None, json_output=False, structured_output: Type[T] | None = None, save_to_history: bool = True, stream: bool = False, tools: List[Tool] | None = None) -> str | Iterator:
        #if task is not None:
        #    #message = GenericMessage(MessageRole.USER, task, medias=medias)
        #    if save_to_history is True:
        #        #history.add_message(message)
        #        pass
        #    else:
        #        history_save = copy.deepcopy(history)
        #        #history_save.add_message(message)
        
        if task is not None:
            logging.info(f"[PROMPT][To: {self.name}]: {task}")
        inference: InferenceServer = InferenceFactory.get_inference(self.server_type)
        history_slot: HistorySlot = inference.go(model_name=self.model_name,
                                                 task=task,
                                                 history=history if save_to_history is True else copy.deepcopy(history),
                                                 endpoint=self.endpoint,
                                                 api_token=self.api_token,
                                                 model_settings=self.model_settings.get_settings(),
                                                 stream=stream,
                                                 json_output=(True if json_output is True else False),
                                                 structured_output=structured_output,
                                                 headers=self.headers,
                                                 tools=tools,
                                                 medias=medias
                                                 )
        if stream is True:
            return response.raw_llm_response  # Only for the simple_chat() method and is of no importance.
        logging.info(f"[AI_RESPONSE][From: {self.name}]: {history_slot.get_message().get_best_visual_form()}")
        if save_to_history is True:
            history.add_slot(history_slot) #add_message(Message(MessageRole.ASSISTANT, response.raw_llm_response, structured_output=response.structured_output, tool_call_id=response.tool_call_id))
        return history_slot.get_message().content #response.raw_llm_response
"""
