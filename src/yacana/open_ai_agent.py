import copy
import json
import logging
from json import JSONDecodeError
from openai import OpenAI, Stream
from typing import List, Type, Any, Literal, T, Dict, Callable
from collections.abc import Iterator
from openai.types.chat.chat_completion import Choice, ChatCompletion
from pydantic import BaseModel

from openai.types.chat import ChatCompletionChunk

from .generic_agent import GenericAgent
from .model_settings import OpenAiModelSettings
from .utils import Dotdict
from .exceptions import MaxToolErrorIter, ToolError, IllogicalConfiguration, TaskCompletionRefusal
from .history import OpenAIToolCallingMessage, HistorySlot, GenericMessage, MessageRole, ToolCall, OpenAIFunctionCallingMessage, OpenAITextMessage, OpenAIMediaMessage, History, OllamaUserMessage, OpenAIStructuredOutputMessage, OpenAIUserMessage
from .tool import Tool

logger = logging.getLogger(__name__)


class OpenAiAgent(GenericAgent):
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

    def __init__(self, name: str, model_name: str, system_prompt: str | None = None, endpoint: str | None = None,
                 api_token: str = "", headers=None, model_settings: OpenAiModelSettings = None, streaming_callback: Callable | None = None, runtime_config: Dict | None = None, **kwargs) -> None:
        """
        Returns a new Agent
        :param name: str : Name of the agent. Can be used during conversations. Use something short and meaningful that doesn't contradict the system prompt
        :param model_name: str : Name of the LLM model that will be sent to the inference server. For instance 'llama:3.1" or 'mistral:latest' etc
        :param system_prompt: str : Defines the way the LLM will behave. For instance set the SP to "You are a pirate" to have it talk like a pirate.
        :param endpoint: str : By default will look for Ollama endpoint on your localhost. If you are using a VM with GPU then update this to the remote URL + port.
        :param api_token: str : The API token used for authentication with the inference server.
        :param headers: dict : Custom headers to be sent with the inference request. If None, an empty dictionary will be used.
        :param model_settings: ModelSettings : All settings that Ollama currently supports as model configuration. This needs to be tested with other inference servers. This allows modifying deep behavioral patterns of the LLM.
        """
        model_settings = OpenAiModelSettings() if model_settings is None else model_settings
        if not isinstance(model_settings, OpenAiModelSettings):
            raise IllogicalConfiguration("model_settings must be an instance of OpenAiModelSettings.")
        super().__init__(name, model_name, model_settings, system_prompt=system_prompt, endpoint=endpoint, api_token=api_token, headers=headers, streaming_callback=streaming_callback, runtime_config=runtime_config, history=kwargs.get("history", None), task_runtime_config=kwargs.get("task_runtime_config", None))
        if self.api_token == "":
            logging.warning("OpenAI requires an API token to be set.")


    def _use_other_tool(self, local_history: History) -> bool:
        tool_continue_prompt = "Now that the tool responded do you need to make another tool call ? Explain why and what are the remaining steps are if any."
        ai_tool_continue_answer: str = self._chat(local_history, tool_continue_prompt)

        # Syncing with global history
        self.history.add_message(OllamaUserMessage(MessageRole.USER, tool_continue_prompt, tags=["yacana_builtin"]))
        self.history.add_message(OllamaUserMessage(MessageRole.ASSISTANT, ai_tool_continue_answer, tags=["yacana_builtin"]))

        tool_confirmation_prompt = "To summarize your previous answer in one word. Do you need to make another tool call ? Answer ONLY by 'yes' or 'no'."
        ai_tool_continue_answer: str = self._chat(local_history, tool_confirmation_prompt,
                                                  save_to_history=False)

        if "yes" in ai_tool_continue_answer.lower():
            logging.info("Continuing tool calls loop\n")
            return True
        else:
            logging.info("Exiting tool calls loop\n")
            return False

    def _call_openai_tool(self, tool: Tool, function_args: Dict) -> str:
        max_call_error: int = tool.max_call_error
        max_custom_error: int = tool.max_custom_error
        tool_output: str = ""

        while True:
            try:
                tool_output: str = tool.function_ref(**function_args)
                if tool_output is None:
                    tool_output = f"Tool {tool.tool_name} was called successfully. It didn't return anything."
                else:
                    tool_output = str(tool_output)
                logging.info(f"[TOOL_RESPONSE][{tool.tool_name}]: {tool_output}\n")
                break
            except (ToolError, TypeError, JSONDecodeError) as e:  # @todo catcher plus large ?
                if type(e) is ToolError or type(e) is JSONDecodeError:
                    logging.warning(f"Tool '{tool.tool_name}' raised an error\n")
                    max_custom_error -= 1
                    tool_output = e.message
                elif type(e) is TypeError:
                    logging.warning(f"Yacana failed to call tool '{tool.tool_name}' correctly based on the LLM output\n")
                    tool_output = str(e)
                    max_call_error -= 1

                if max_custom_error < 0:
                    raise MaxToolErrorIter(
                        f"Too many errors were raise by the tool '{tool.tool_name}'. Stopping after {tool.max_custom_error} errors. You can change the maximum errors a tool can raise in the Tool constructor with @max_custom_error.")
                if max_call_error < 0:
                    raise MaxToolErrorIter(
                        f"Too many errors occurred while trying to call the python function by Yacana (tool name: {tool.tool_name}). Stopping after {tool.max_call_error} errors. You can change the maximum call error in the Tool constructor with @max_call_error.")
                self._chat(self.history, f"The tool returned an error: `{tool_output}`\nUsing this error message, fix the JSON you generated.")
        return tool_output

    def _update_tool_definition(self, tools: List[Tool]) -> None:
        tools: List[Tool] = [] if tools is None else tools
        for tool in tools:
            if tool._openai_function_schema is None:
                tool._function_to_json_with_pydantic()

    def _interact(self, task: str, tools: List[Tool], json_output: bool, structured_output: Type[BaseModel] | None, images: List[str] | None, streaming_callback: Callable | None = None, task_runtime_config: Dict | None = None) -> GenericMessage:
        self._update_tool_definition(tools)
        self.task_runtime_config = task_runtime_config if task_runtime_config is not None else {}

        if len(tools) == 0:  # @todo implicitement si tu mets des tools du coup ca coupe l'herbe sous le pied à l'image car ca n'ira pas là. Pour le moment les 2 sont mutuellement exclusif.
            self._chat(self.history, task, medias=images, json_output=json_output, structured_output=structured_output, streaming_callback=streaming_callback)
        elif len(tools) > 0:
            self._chat(self.history, task, medias=images, json_output=json_output, structured_output=structured_output, tools=tools)
            if isinstance(self.history.get_last_message(), OpenAIFunctionCallingMessage):
                for tool_call in self.history.get_last_message().tool_calls:
                    tool = next((tool for tool in tools if tool.tool_name == tool_call.name), None)
                    if tool is None:
                        raise ValueError(f"Tool {tool_call.name} not found in tools list")  # @todo Autre chose qu'un valueError, genre une classe custom ?
                    print("found ", tool.tool_name)
                    tool_output: str = self._call_openai_tool(tool, tool_call.arguments)
                    self.history.add_message(OpenAIToolCallingMessage(tool_output, tool_call.call_id, tags=["yacana_builtin"]))
                    #self.history.add_message(GenericMessage(MessageRole.TOOL, tool_output, tool_call_id=tool_call.call_id, tags=["yacana_builtin"]))  # @todo nb 5 & 6
                logging.info(f"[PROMPT][To: {self.name}]: Retrying with original task and tools answer: '{task}'")
                self._chat(self.history, None, medias=images, json_output=json_output, structured_output=structured_output, streaming_callback=streaming_callback)
            """
            else:
                print("No tool calls even though tools were provided !!")
                self.history.add_message(Message(MessageRole.ASSISTANT))
                # @todo c'est ici le pb. Si chatGPT a choisit de ne pas utiliser de tool finalement alors c'est juste de l'inférence classique et faut choper choice[0].content
            """
        return self.history.get_last_message()

    def _chat(self, history: History, task: str | None, medias: List[str] | None = None, json_output=False, structured_output: Type[T] | None = None, save_to_history: bool = True, tools: List[Tool] | None = None, streaming_callback: Callable | None = None) -> str | Iterator:
        """
        if task is not None:
            #message = GenericMessage(MessageRole.USER, task, medias=medias)
            if save_to_history is True:
                #history.add_message(message)
                pass
            else:
                history_save = copy.deepcopy(history)
                #history_save.add_message(message)
        """
        #if task is not None:
        #    logging.info(f"[PROMPT][To: {self.name}]: {task}")
        #inference: InferenceServer = InferenceFactory.get_inference(self.server_type)
        history_slot: HistorySlot = self._go(task=task,
                                             history=history if save_to_history is True else copy.deepcopy(history),
                                             json_output=(True if json_output is True else False),
                                             structured_output=structured_output,
                                             tools=tools,
                                             medias=medias,
                                             streaming_callback=streaming_callback
                                             )
        #if stream is True:
        #    return response.raw_llm_response  # Only for the simple_chat() method and is of no importance.
        logging.info(f"[AI_RESPONSE][From: {self.name}]: {history_slot.get_message().get_as_pretty()}")
        if save_to_history is True:
            history.add_slot(history_slot)  #add_message(Message(MessageRole.ASSISTANT, response.raw_llm_response, structured_output=response.structured_output, tool_call_id=response.tool_call_id))
        return history_slot.get_message().content  #response.raw_llm_response

    def is_structured_output(self, choice: Choice) -> bool:
        return hasattr(choice.message, "parsed") and choice.message.parsed is not None

    def is_tool_calling(self, choice: Choice) -> bool:
        return hasattr(choice.message, "tool_calls") and choice.message.tool_calls is not None and len(choice.message.tool_calls) > 0

    def is_common_chat(self, choice: Choice) -> bool:
        return hasattr(choice.message, "content") and choice.message is not None

    def _dispatch_chunk_if_streaming(self, completion: ChatCompletion | Stream[ChatCompletionChunk], streaming_callback: Callable | None):
        if streaming_callback is None:
            return completion
        all_chunks = ""
        for chunk in completion:
            if chunk.choices[0].delta.refusal in (False, None):
                if chunk.choices[0].delta.content is not None:
                    all_chunks += chunk.choices[0].delta.content
                    streaming_callback(chunk.choices[0].delta.content)
            else:
                raise TaskCompletionRefusal("Got a refusal from the LLM. This is not supported in streaming mode.")
        return Dotdict({
            "choices": [
                {
                    "message": {
                        "content": all_chunks,
                    }
                }
            ]
        })

    def _go(self, task: str | None, history: History, json_output: bool, structured_output: Type[T] | None, tools: List[Tool] | None = None, medias: List[str] | None = None, streaming_callback: Callable | None = None) -> HistorySlot:
        if task is not None:
            logging.info(f"[PROMPT][To: {self.name}]: {task}")
            #if medias is not None:
            #    history.add_message(OpenAIMediaMessage(MessageRole.USER, task, medias, tags=["yacana_builtin"]))
            #else:
            #    history.add_message(OpenAITextMessage(MessageRole.USER, task, tags=["yacana_builtin"]))
            history.add_message(OpenAIUserMessage(MessageRole.USER, task, tags=["yacana_builtin"], medias=medias, structured_output=structured_output))
        #print(f"inference : model_name: {self.model_name}, history: {history}, endpoint: {self.endpoint}, api_token: {self.api_token}, model_settings: {self.model_settings.get_settings()}, stream: {str(streaming_callback)}, json_output: {json_output}, structured_output: {structured_output}, headers: {self.headers}, tools: {str(tools)}")
        # Extracting all json schema from tools, so it can be passed to the OpenAI API
        all_function_calling_json = [tool._openai_function_schema for tool in tools] if tools else []

        tool_choice_option = self._find_right_tool_choice_option(tools)
        response_format = self._find_right_output_format(structured_output, json_output)

        client = OpenAI(
            api_key=self.api_token,
            base_url=self.endpoint
        )


        # @todo tests multi turn
        # @todo plus de tests multimodal

        params = {
            "model": self.model_name,
            "messages": history.get_messages_as_dict(),
            **({"stream": True} if streaming_callback is not None else {}),
            **({"response_format": response_format} if response_format is not None else {}),
            **({"tools": all_function_calling_json} if len(all_function_calling_json) > 0 else {}),
            **({"tool_choice": tool_choice_option} if len(all_function_calling_json) > 0 else {}),
            **self.model_settings.get_settings(),
            **self.runtime_config,
            **self.task_runtime_config
        }
        print("tool choice = ", tool_choice_option)
        print("----")
        #print("current params = ", json.dumps(params, indent=2))
        print("params = ", params)
        print("----")

        history_slot = HistorySlot()
        if structured_output is not None:
            print("before request")
            response = client.beta.chat.completions.parse(**params)
            print("after request")
        else:
            response = client.chat.completions.create(**params)
            response = self._dispatch_chunk_if_streaming(response, streaming_callback)

        self.task_runtime_config = {}
        history_slot.set_raw_llm_json(response.model_dump_json())

        print("Résultat de l'inférence quelle quelle soit = ")
        print(response.model_dump_json(indent=2))
        logging.debug("Inference output: %s", response.model_dump_json(indent=2))

        for choice in response.choices:

            if self.is_structured_output(choice):
                print("This is a structured_output answer.")
                logging.debug("Response assessment is structured output")
                if choice.message.refusal is not None:
                    raise TaskCompletionRefusal(choice.message.refusal)  # Refusal key is only available for structured output but also doesn't work very well
                history_slot.add_message(OpenAIStructuredOutputMessage(MessageRole.ASSISTANT, choice.message.content, choice.message.parsed, tags=["yacana_builtin"]))

            elif self.is_tool_calling(choice):
                print("This is a tool_calling answer.")
                logging.debug("Response assessment is tool calling")
                tool_calls: List[ToolCall] = []  # @todo on pourait peut etre renomer ToolCall en InferencedToolCall pour montrer que c'est le résultat d'une inférence et pas un truc qu'on donne au départ. A voir pour le nom.
                for tool_call in choice.message.tool_calls:
                    tool_calls.append(ToolCall(tool_call.id, tool_call.function.name, json.loads(tool_call.function.arguments)))
                    print("tool info = ", tool_call.id, tool_call.function.name, tool_call.function.arguments)
                history_slot.add_message(OpenAIFunctionCallingMessage(tool_calls, tags=["yacana_builtin"]))

            elif self.is_common_chat(choice):
                print("this is a classic chat answer.")
                logging.debug("Response assessment is classic chat answer")
                history_slot.add_message(OpenAITextMessage(MessageRole.ASSISTANT, choice.message.content, tags=["yacana_builtin"]))
            else:
                raise ValueError("Unknown response from OpenAI API")  # @todo error custom

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
