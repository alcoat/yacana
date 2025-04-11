import copy
from json import JSONDecodeError
import json
import logging
from ollama import Client
from typing import List, Type, Any, T, Dict, Callable, Mapping
from collections.abc import Iterator
from pydantic import BaseModel

from .generic_agent import GenericAgent
from .model_settings import OllamaModelSettings
from .utils import Dotdict
from .exceptions import MaxToolErrorIter, ToolError, IllogicalConfiguration, TaskCompletionRefusal
from .history import HistorySlot, GenericMessage, MessageRole, History, OllamaUserMessage, OllamaStructuredOutputMessage, OllamaTextMessage
from .tool import Tool

logger = logging.getLogger(__name__)


class OllamaAgent(GenericAgent):
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

    def __init__(self, name: str, model_name: str, system_prompt: str | None = None, endpoint: str = "http://127.0.0.1:11434", headers=None, model_settings: OllamaModelSettings = None, streaming_callback: Callable | None = None, runtime_config: Dict | None = None, **kwargs) -> None:
        """
        Returns a new Agent
        :param name: str : Name of the agent. Can be used during conversations. Use something short and meaningful that doesn't contradict the system prompt
        :param model_name: str : Name of the LLM model that will be sent to the inference server. For instance 'llama:3.1" or 'mistral:latest' etc
        :param system_prompt: str : Defines the way the LLM will behave. For instance set the SP to "You are a pirate" to have it talk like a pirate.
        :param endpoint: str : By default will look for Ollama endpoint on your localhost. If you are using a VM with GPU then update this to the remote URL + port.
        :param server_type: ServerType : The type of server to use for inference. Options are ServerType.OLLAMA, ServerType.VLLM, and ServerType.OPENAI.
        :param headers: dict : Custom headers to be sent with the inference request. If None, an empty dictionary will be used.
        :param model_settings: ModelSettings : All settings that Ollama currently supports as model configuration. This needs to be tested with other inference servers. This allows modifying deep behavioral patterns of the LLM.
        """
        model_settings = OllamaModelSettings() if model_settings is None else model_settings
        if not isinstance(model_settings, OllamaModelSettings):
            raise IllogicalConfiguration("model_settings must be an instance of OllamaModelSettings.")
        super().__init__(name, model_name, model_settings, system_prompt=system_prompt, endpoint=endpoint, api_token="", headers=headers, streaming_callback=streaming_callback, runtime_config=runtime_config)

    def _choose_tool_by_name(self, local_history: History, tools: List[Tool]) -> Tool:
        max_tool_name_use_iter: int = 0
        while max_tool_name_use_iter < 5:

            tool_choose: str = f"You can only use one tool at a time. From this list of tools which one do you want to use: [{', '.join([tool.tool_name for tool in tools])}]. You must answer ONLY with the single tool name. Nothing else."
            ai_tool_choice: str = self._chat(local_history, tool_choose)
            ai_tool_choice = ai_tool_choice.strip(" \n")

            found_tools: List[Tool] = []

            for tool in tools:
                # If tool name is present somewhere in AI response
                if tool.tool_name.lower() in ai_tool_choice.lower():
                    # If tool name is not an exact match in AI response
                    if ai_tool_choice.lower() != tool.tool_name.lower():
                        logging.warning("Tool choice was not an exact match but a substring match\n")
                    found_tools.append(tool)

            # If there was more than 1 tool name in the AI answer we cannot be sure what tool it chose. So we try again.
            if len(found_tools) == 1:
                self.model_settings.reset()
                return found_tools[0]
            elif len(found_tools) >= 2:
                logging.warning("More than one tool was proposed. Trying again.\n")

            # No tool or too many tools found
            local_history.add_message(OllamaUserMessage(MessageRole.USER,
                                                        "You didn't only output a tool name. Let's try again with only outputting the tool name to use.", is_yacana_builtin=True))
            logging.info(f"[prompt]: You didn't only output a tool name. Let's try again with only outputting the tool name to use.\n")
            local_history.add_message(OllamaTextMessage(MessageRole.ASSISTANT,
                                                        "I'm sorry. I know I must ONLY output the name of the tool I wish to use. Let's try again !", is_yacana_builtin=True))
            logging.info(f"[AI_RESPONSE]: I'm sorry. I know I must ONLY output the name of the tool I wish to use. Let's try again !\n")
            max_tool_name_use_iter += 1
            # Forcing LLM to be less chatty and more focused
            if max_tool_name_use_iter >= 2:
                if self.model_settings.temperature is None:
                    self.model_settings.temperature = 2
                self.model_settings.temperature = self.model_settings.temperature / 2
            if max_tool_name_use_iter >= 3:
                self.model_settings.tfs_z = 2
            if max_tool_name_use_iter >= 4:
                # Getting the longer tool name and setting max token output to this value x 1.5. This should reduce output length of AI's response.
                self.model_settings.num_predict = len([max([tool.tool_name for tool in tools], key=len)]) * 1.5

        self.model_settings.reset()
        raise MaxToolErrorIter("[ERROR] LLM did not choose a tool from the list despite multiple attempts.")

    def _tool_call(self, tool_training_history: History, tool: Tool) -> str:
        max_call_error: int = tool.max_call_error
        max_custom_error: int = tool.max_custom_error
        tool_output: str = ""

        while True:
            additional_prompt_help: str = ""
            try:
                args: dict = json.loads(tool_training_history.get_last_message().content)
                tool_output: str = tool.function_ref(**args)
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
                    additional_prompt_help = 'Remember that you must output ONLY the tool arguments as valid JSON. For instance: ' + str(
                        {key: ("arg " + str(i)) for i, key in enumerate(tool._function_args)})
                    max_call_error -= 1

                if max_custom_error < 0:
                    raise MaxToolErrorIter(
                        f"Too many errors were raise by the tool '{tool.tool_name}'. Stopping after {tool.max_custom_error} errors. You can change the maximum errors a tool can raise in the Tool constructor with @max_custom_error.")
                if max_call_error < 0:  # @todo What happens to the history if the no raise option is true ? Maybe add a prompt + ai answer saying something went wrong.
                    raise MaxToolErrorIter(
                        f"Too many errors occurred while trying to call the python function by Yacana (tool name: {tool.tool_name}). Stopping after {tool.max_call_error} errors. You can change the maximum call error in the Tool constructor with @max_call_error.")

                fix_your_shit_prompt = f"The tool returned an error: `{tool_output}`\nUsing this error message, fix the JSON arguments you gave.\n{additional_prompt_help}"
                self._chat(tool_training_history, fix_your_shit_prompt, json_output=True)  # @todo ici je ne comprend plus comment ca boucle
        return tool_output

    def _reconcile_history_multi_tools(self, tool_training_history: History, local_history: History, tool: Tool, tool_output: str):
        # Master history + local history get fake USER prompt to ask for tool output
        self.history.add_message(OllamaUserMessage(MessageRole.USER, f"Output the tool '{tool.tool_name}' as valid JSON.", is_yacana_builtin=True))
        local_history.add_message(OllamaUserMessage(MessageRole.USER, f"Output the tool '{tool.tool_name}' as valid JSON.", is_yacana_builtin=True))

        # Master history + local history get fake ASSISTANT prompt calling the tool correctly
        self.history.add_message(OllamaTextMessage(MessageRole.ASSISTANT, tool_training_history.get_last_message().content, is_yacana_builtin=True))
        local_history.add_message(OllamaTextMessage(MessageRole.ASSISTANT, tool_training_history.get_last_message().content, is_yacana_builtin=True))

        # Master history + local history get fake USER prompt with the answer of the tool
        # @todo Finishing with a user prompt will render 2 consecutive USER prompts in the final history. This might be resolved by this kind of trick: 'USER: You will receive the tool output now' => 'ASSISTANT: Yes I got the tool output just now. It is the following: <output>'
        # Enphasing on the above @todo we now have access to a tool type so the alternation with user and assistant is probably not a problem anymore
        self.history.add_message(OllamaTextMessage(MessageRole.TOOL, tool_output, is_yacana_builtin=True))
        local_history.add_message(OllamaTextMessage(MessageRole.TOOL, tool_output, is_yacana_builtin=True))

    def _reconcile_history_solo_tool(self, last_tool_call: str, tool_output: str, task: str, tool: Tool):
        self.history.add_message(OllamaUserMessage(MessageRole.USER, task, is_yacana_builtin=True))
        self.history.add_message(
            OllamaTextMessage(MessageRole.ASSISTANT,
                              f"I can use the tool '{tool.tool_name}' related to the task to solve it correctly.", is_yacana_builtin=True))
        self.history.add_message(OllamaUserMessage(MessageRole.USER, f"Output the tool '{tool.tool_name}' as valid JSON.", is_yacana_builtin=True))
        self.history.add_message(OllamaTextMessage(MessageRole.ASSISTANT, last_tool_call, is_yacana_builtin=True))
        self.history.add_message(OllamaTextMessage(MessageRole.TOOL, tool_output, is_yacana_builtin=True))

    def _post_tool_output_reflection(self, tool: Tool, tool_output: str, history: History) -> str:
        """
        When @tool.post_tool_prompt is None we only return the tool_output and nothing else. If it's not None and a prompt is given. It must clarify what to do with this tool_output. Maybe include it in a sentence or do some kind of
        formatting with it. Whatever the prompt is, the result of this reflection WILL BE the output and not just the raw tool output. Note that this method is only called when we are wrapping things up. No more tools to call after this function !
        @param tool: Tool : The tool containing the prompt to use for post tool calling reflection
        @param tool_output: str : The final raw output of the tool that should have ended up the only answer
        @param history: History : The history to write to
        @return: str : The reflection that coming from the LLM when given the tool output and the @post_tool_prompt
        """
        raise NotImplemented
        if tool.post_tool_prompt is not None:
            return self._chat(history, f"I give you the tool output between the tags <tool_output></tool_output>: <tool_output>{tool_output}</tool_output>.\nUsing this new knowledge you have this task to solve: '{tool.post_tool_prompt}'")

    def _use_other_tool(self, local_history: History) -> bool:
        tool_continue_prompt = "Now that the tool responded do you need to make another tool call ? Explain why and what are the remaining steps are if any."
        ai_tool_continue_answer: str = self._chat(local_history, tool_continue_prompt)

        # Syncing with global history
        self.history.add_message(OllamaUserMessage(MessageRole.USER, tool_continue_prompt, is_yacana_builtin=True))
        self.history.add_message(OllamaTextMessage(MessageRole.ASSISTANT, ai_tool_continue_answer, is_yacana_builtin=True))

        tool_confirmation_prompt = "To summarize your previous answer in one word. Do you need to make another tool call ? Answer ONLY by 'yes' or 'no'."
        ai_tool_continue_answer: str = self._chat(local_history, tool_confirmation_prompt,
                                                  save_to_history=False)

        if "yes" in ai_tool_continue_answer.lower():
            logging.info("Continuing tool calls loop\n")
            return True
        else:
            logging.info("Exiting tool calls loop\n")
            return False

    def _interact(self, task: str, tools: List[Tool], json_output: bool, structured_output: Type[BaseModel] | None, medias: List[str] | None, streaming_callback: Callable | None = None) -> GenericMessage:
        tools: List[Tool] = [] if tools is None else tools

        if len(tools) == 0:
            self._chat(self.history, task, medias=medias, json_output=json_output, structured_output=structured_output, streaming_callback=streaming_callback)

        elif len(tools) == 1:
            if streaming_callback is not None: # @todo On pourrait streamer le dernier prompt mais actuellement il n'y en a pas. On fait pas comme OpenAI avec un message final qui reprend tout. Mais ca serait une idée... Et ce dernier call pourrait être streamé.
                raise (IllogicalConfiguration("Currently Yacana's custom function calling system doesn't allow streaming callbacks because there is no uniq final prompt given to the LLM that could be streamed."))
            local_history = copy.deepcopy(self.history)
            tool: Tool = tools[0]

            tmp = str(tool._function_prototype + " - " + tool.function_description)
            tool_ack_prompt = f"I give you the following tool definition that you {'must' if tool.optional is False else 'may'} use to fulfill a future task: {tmp}. Please acknowledge the given tool."
            self._chat(local_history, tool_ack_prompt)

            tool_examples_prompt = 'To use the tool you MUST extract each parameter and use it as a JSON key like this: {"arg1": "<value1>", "arg2": "<value2>"}. You must respect arguments type. For instance, the tool `getWeather(city: str, lat: int, long: int)` would be structured like this {"city": "new-york", "lat": 10, "lon": 20}. In our case, the tool call you must use must look like that: ' + str(
                {key: ("arg " + str(i)) for i, key in enumerate(tool._function_args)})
            self._chat(local_history, tool_examples_prompt)

            local_history._concat_history(tool._get_examples_as_history())

            # This section checks whether we need a tool or not. If not we call the LLM like if tools == 0 and exit the function.
            if tool.optional is True:
                task_outputting_prompt = f'You have a task to solve. In your opinion, is using the tool "{tool.tool_name}" relevant to solve the task or not ? The task is:\n{task}'
                self._chat(local_history, task_outputting_prompt, medias=medias)

                tool_use_router_prompt: str = "To summarize in one word your previous answer. Do you wish to use the tool or not ? Respond ONLY by 'yes' or 'no'."
                tool_use_ai_answer: str = self._chat(local_history, tool_use_router_prompt, save_to_history=False)
                if not ("yes" in tool_use_ai_answer.lower()):  # Better than checking for "no" as a substring could randomly match
                    self._chat(self.history, task, medias=medias, json_output=json_output, structured_output=structured_output)  # !!Actual function calling
                    return self.history.get_last_message()

            # If getting here the tool call is inevitable
            task_outputting_prompt = f'You have a task to solve. Use the tool at your disposition to solve the task by outputting as JSON the correct arguments. In return you will get an answer from the tool. The task is:\n{task}'
            self._chat(local_history, task_outputting_prompt, medias=medias, json_output=True)  # !!Actual function calling
            tool_output: str = self._tool_call(local_history, tool)  # !!Actual tool calling
            logging.debug(f"Tool output: {tool_output}\n")

            # Unused for now. Could replace the raw tool output that ends in the history with the result of this methods that reflects on a "post tool call prompt" + the tool output.
            #post_rendered_tool_output: str = self.post_tool_output_reflection(tool, tool_output, local_history)

            self._reconcile_history_solo_tool(local_history.get_last_message().content, tool_output, task, tool)

        elif len(tools) > 1:
            if streaming_callback is not None:
                raise (IllogicalConfiguration("Currently Yacana's custom function calling system doesn't allow streaming callbacks because there is no uniq final prompt given to the LLM that could be streamed."))
            local_history = copy.deepcopy(self.history)

            tools_presentation: str = "* " + "\n* ".join([
                f"Name: '{tool.tool_name}' - Usage: {tool._function_prototype} - Description: {tool.function_description}"
                for tool in tools])
            tool_ack_prompt = f"You have access to this list of tools definitions you can use to fulfill tasks :\n{tools_presentation}\nPlease acknowledge the given tools."
            self._chat(local_history, tool_ack_prompt)

            tool_use_decision: str = f"You have a task to solve. I will give it to you between these tags `<task></task>`. However, your actual job is to decide if you need to use any of the available tools to solve the task or not. If you do need tools then output their names. The task to solve is <task>{task}</task> So, would any tools be useful in relation to the given task ?"
            self._chat(local_history, tool_use_decision, medias=medias)

            tool_router: str = "In order to summarize your previous answer in one word. Did you chose to use any tools ? Respond ONLY by 'yes' or 'no'."
            ai_may_use_tools: str = self._chat(local_history, tool_router, save_to_history=False)

            if "yes" in ai_may_use_tools.lower():
                self.history.add_message(OllamaUserMessage(MessageRole.USER, task, is_yacana_builtin=True))
                self.history.add_message(
                    OllamaTextMessage(MessageRole.ASSISTANT, "I should use tools related to the task to solve it correctly.", is_yacana_builtin=True))
                while True:
                    tool: Tool = self._choose_tool_by_name(local_history, tools)
                    tool_training_history = copy.deepcopy(local_history)
                    tool_examples_prompt = 'To use the tool you MUST extract each parameter and use it as a JSON key like this: {"arg1": "<value1>", "arg2": "<value2>"}. You must respect arguments type. For instance, the tool `getWeather(city: str, lat: int, long: int)` would be structured like this {"city": "new-york", "lat": 10, "lon": 20}. In our case, the tool call you must use must look like that: ' + str(
                        {key: ("arg " + str(i)) for i, key in enumerate(tool._function_args)})
                    self._chat(tool_training_history, tool_examples_prompt)

                    tool_training_history._concat_history(tool._get_examples_as_history())

                    tool_use: str = "Now that I showed you examples on how the tool is used it's your turn. Output the tool as valid JSON."
                    self._chat(tool_training_history, tool_use, medias=medias, json_output=True)  # !!Actual function calling
                    tool_output: str = self._tool_call(tool_training_history, tool)  # !!Actual tool calling
                    self._reconcile_history_multi_tools(tool_training_history, local_history, tool, tool_output)
                    use_other_tool: bool = self._use_other_tool(local_history)
                    if use_other_tool is True:
                        continue
                    else:
                        break
            else:
                if not ("no" in ai_may_use_tools.lower()):
                    logging.warning(
                        "Yacana couldn't determine if the LLM chose to use a tool or not. As a decision must be taken "
                        "the default behavior is to not use any tools. If this warning persists you might need to rewrite your initial prompt.")
                # Getting here means that no tools were selected by the LLM and we act like tools == 0
                self._chat(self.history, task, medias=medias, json_output=json_output)

        return self.history.get_last_message()

    def _stream(self):
        pass

    def _chat(self, history: History, task: str | None, medias: List[str] | None = None, json_output=False, structured_output: Type[T] | None = None, save_to_history: bool = True, tools: List[Tool] | None = None, streaming_callback: Callable | None = None) -> str | Iterator:
        history_slot: HistorySlot = self._go(task=task,
                                             history=history if save_to_history is True else copy.deepcopy(history),
                                             json_output=(True if json_output is True else False),
                                             structured_output=structured_output,
                                             medias=medias,
                                             streaming_callback=streaming_callback
                                             )
        logging.info(f"[AI_RESPONSE][From: {self.name}]: {history_slot.get_message().get_as_pretty()}")
        if save_to_history is True:
            history.add_slot(history_slot)
        return history_slot.get_message().content

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
                    'tool_calls': getattr(message, 'tool_calls', None)
                }

            # Return the JSON string representation
            return json.dumps(result, indent=4)
        except Exception as e:
            raise TypeError(f"Failed to convert response to JSON: {e}")

    def _dispatch_chunk_if_streaming(self, completion: Mapping[str, Any] | Iterator[Mapping[str, Any]], streaming_callback: Callable | None) -> Dict | Mapping[str, Any] | Iterator[Mapping[str, Any]]:
        if streaming_callback is None:
            return completion
        all_chunks = ""
        for chunk in completion:
            if chunk['message']['content'] is not None:
                all_chunks += chunk['message']['content']
                streaming_callback(chunk['message']['content'])
            else:
                raise TaskCompletionRefusal("Streaming LLMs response returned no data (content == None).")
        return Dotdict({
                    "message": {
                        "content": all_chunks,
                    }
                }
            )

    def _go(self, task: str | None, history: History, json_output: bool, structured_output: Type[T] | None, medias: List[str] | None = None, streaming_callback: Callable | None = None) -> HistorySlot:
        if task is not None:
            logging.info(f"[PROMPT][To: {self.name}]: {task}")
            history.add_message(OllamaUserMessage(MessageRole.USER, task, is_yacana_builtin=True, medias=medias, structured_output=structured_output))
            """
            if medias is not None:
                history.add_message(OllamaMediasMessage(MessageRole.USER, task, medias, is_yacana_builtin=True))
            else:
                history.add_message(OllamaUserMessage(MessageRole.USER, task, is_yacana_builtin=True))
            """

        history_slot = HistorySlot()
        client = Client(host=self.endpoint, headers=self.headers)
        params = {
            "model": self.model_name,
            "messages": history.get_messages_as_dict(),
            "format": self._get_expected_output_format(json_output, structured_output),
            "stream": True if streaming_callback is not None else False,
            "options": self.model_settings.get_settings(),
            **self.runtime_config
        }
        response = client.chat(**params)
        if structured_output is not None:
            history_slot.add_message(OllamaStructuredOutputMessage(MessageRole.ASSISTANT, str(response['message']['content']), structured_output.model_validate_json(response['message']['content']), is_yacana_builtin=True))
        else:
            response = self._dispatch_chunk_if_streaming(response, streaming_callback)
            history_slot.add_message(OllamaTextMessage(MessageRole.ASSISTANT, response['message']['content'], is_yacana_builtin=True))

        history_slot.set_raw_llm_json(self._response_to_json(response))
        return history_slot
