import copy
import json
import logging
from json import JSONDecodeError
from typing import List, Iterator, Type, T, Dict
from ollama import Client
from pydantic import BaseModel

from .exceptions import MaxToolErrorIter, ToolError
from .history import History, Message, MessageRole
from .inference import InferenceFactory, ServerType, InferenceOutput
from .logging_config import LoggerManager
from .modelSettings import ModelSettings
from .tool import Tool

logger = logging.getLogger(__name__)


class Agent:
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

    def __init__(self, name: str, model_name: str, system_prompt: str | None = None, endpoint: str = "http://127.0.0.1:11434",
                 api_token: str = "", server_type=ServerType.OLLAMA, headers=None, model_settings: ModelSettings = None) -> None:
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
        self.server_type: ServerType = server_type
        self.headers = {} if headers is None else headers
        self.endpoint: str = endpoint

        self.history: History = History()
        if self.system_prompt is not None:
            self.history.add(Message(MessageRole.SYSTEM, system_prompt))

        if self.server_type == ServerType.OPENAI and self.api_token == "":
            logging.warning("OpenAI requires an API token to be set.")

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
                self.history.add(Message(MessageRole.ASSISTANT, "".join(complete_llm_answer)))
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
            "history": self.history.get_as_dict()
        }
        with open(file_path, 'w') as file:
            json.dump(final, file, indent=4)

    @classmethod
    def get_agent_from_state(cls, file_path: str) -> 'Agent':
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
            local_history.add(Message(MessageRole.USER,
                                      "You didn't only output a tool name. Let's try again with only outputting the tool name to use."))
            logging.info(f"[prompt]: You didn't only output a tool name. Let's try again with only outputting the tool name to use.\n")
            local_history.add(Message(MessageRole.ASSISTANT,
                                      "I'm sorry. I know I must ONLY output the name of the tool I wish to use. Let's try again !"))
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
                args: dict = json.loads(tool_training_history.get_last().content)
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
        self.history.add(Message(MessageRole.USER, f"Output the tool '{tool.tool_name}' as valid JSON."))
        local_history.add(Message(MessageRole.USER, f"Output the tool '{tool.tool_name}' as valid JSON."))

        # Master history + local history get fake ASSISTANT prompt calling the tool correctly
        self.history.add(Message(MessageRole.ASSISTANT, tool_training_history.get_last().content))
        local_history.add(Message(MessageRole.ASSISTANT, tool_training_history.get_last().content))

        # Master history + local history get fake USER prompt with the answer of the tool
        # @todo Finishing with a user prompt will render 2 consecutive USER prompts in the final history. This might be resolved by this kind of trick: 'USER: You will receive the tool output now' => 'ASSISTANT: Yes I got the tool output just now. It is the following: <output>'
        # Enphasing on the above @todo we now have access to a tool type so the alternation with user and assistant is probably not a problem anymore
        self.history.add(Message(MessageRole.TOOL, tool_output))
        local_history.add(Message(MessageRole.TOOL, tool_output))

    def _reconcile_history_solo_tool(self, last_tool_call: str, tool_output: str, task: str, tool: Tool):
        self.history.add(Message(MessageRole.USER, task))
        self.history.add(
            Message(MessageRole.ASSISTANT,
                    f"I can use the tool '{tool.tool_name}' related to the task to solve it correctly."))
        self.history.add(Message(MessageRole.USER, f"Output the tool '{tool.tool_name}' as valid JSON."))
        self.history.add(Message(MessageRole.ASSISTANT, last_tool_call))
        self.history.add(Message(MessageRole.TOOL, tool_output))

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
        self.history.add(Message(MessageRole.USER, tool_continue_prompt))
        self.history.add(Message(MessageRole.ASSISTANT, ai_tool_continue_answer))

        tool_confirmation_prompt = "To summarize your previous answer in one word. Do you need to make another tool call ? Answer ONLY by 'yes' or 'no'."
        ai_tool_continue_answer: str = self._chat(local_history, tool_confirmation_prompt,
                                                  save_to_history=False)

        if "yes" in ai_tool_continue_answer.lower():
            logging.info("Continuing tool calls loop\n")
            return True
        else:
            logging.info("Exiting tool calls loop\n")
            return False

    def _interact(self, task: str, tools: List[Tool], json_output: bool, structured_output: Type[BaseModel] | None, images: List[str] | None) -> Message:
        if self.server_type == ServerType.OLLAMA:
            return self._interact_ollama(task, tools, json_output, structured_output, images)
        elif self.server_type == ServerType.OPENAI:
            return self._interact_openai(task, tools, json_output, structured_output, images)
        elif self.server_type == ServerType.VLLM:
            return self._interact_vllm(task, tools, json_output, structured_output, images)
        else:
            raise ValueError(f"Server type {self.server_type} is not supported.")

    def _interact_vllm(self, task: str, tools: List[Tool], json_output: bool, structured_output: Type[BaseModel] | None, images: List[str] | None) -> Message:
        raise NotImplemented()

    def _openai_tool_call(self, tool: Tool, function_args: Dict) -> str:
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

    def _interact_openai(self, task: str, tools: List[Tool], json_output: bool, structured_output: Type[BaseModel] | None, images: List[str] | None) -> Message:
        tools: List[Tool] = [] if tools is None else tools
        if len(tools) == 0:
            self._chat(self.history, task, images=images, json_output=json_output, structured_output=structured_output)
        elif len(tools) > 0:
            self._chat(self.history, task, images=images, json_output=json_output, structured_output=structured_output, tools=tools)
            #print("le json de tool calling ?\n", self.history.get_last().content)
            print("pk tu pete sur ca ? = ", self.history.get_last().content)
            function_calling = json.loads(self.history.get_last().content)
            for function_data in function_calling["tool_calls"]:
                if function_data["type"] == "function":  # @todo et on fait quoi si c'est pas une fonction ?
                    tool = next((tool for tool in tools if tool.tool_name == function_data["function"]["name"]), None)
                    if tool is None:
                        raise ValueError(f"Tool {function_data['function']['name']} not found in tools list")
                    print("found ", tool.tool_name)
                    tool_output: str = self._openai_tool_call(tool, function_data["function"]["arguments"])
                    self.history.add(Message(MessageRole.TOOL, tool_output, tool_call_id=self.history.get_last().tool_call_id))
                else:
                    # @todo c'est ici le pb. Si chatGPT a choisit de ne pas utiliser de tool finalement alors c'est juste de l'inférence classique et faut choper choice[0].content
                    logging.error("Receiving a non-function type in the function calling list. This is not supported. (Continuing but you should abort...)")
                    logging.error(f"Received: {function_data}")
        return self.history.get_last()

    def _interact_ollama(self, task: str, tools: List[Tool], json_output: bool, structured_output: Type[BaseModel] | None, images: List[str] | None) -> Message:
        tools: List[Tool] = [] if tools is None else tools

        if len(tools) == 0:
            self._chat(self.history, task, images=images, json_output=json_output, structured_output=structured_output)

        elif len(tools) == 1:
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
                self._chat(local_history, task_outputting_prompt, images=images)

                tool_use_router_prompt: str = "To summarize in one word your previous answer. Do you wish to use the tool or not ? Respond ONLY by 'yes' or 'no'."
                tool_use_ai_answer: str = self._chat(local_history, tool_use_router_prompt, save_to_history=False)
                if not ("yes" in tool_use_ai_answer.lower()):  # Better than checking for "no" as a substring could randomly match
                    self._chat(self.history, task, images=images, json_output=json_output, structured_output=structured_output)  # !!Actual function calling
                    return self.history.get_last()

            # If getting here the tool call is inevitable
            task_outputting_prompt = f'You have a task to solve. Use the tool at your disposition to solve the task by outputting as JSON the correct arguments. In return you will get an answer from the tool. The task is:\n{task}'
            self._chat(local_history, task_outputting_prompt, images=images, json_output=True)  # !!Actual function calling
            tool_output: str = self._tool_call(local_history, tool)  # !!Actual tool calling
            logging.debug(f"Tool output: {tool_output}\n")

            # Unused for now. Could replace the raw tool output that ends in the history with the result of this methods that reflects on a "post tool call prompt" + the tool output.
            #post_rendered_tool_output: str = self.post_tool_output_reflection(tool, tool_output, local_history)

            self._reconcile_history_solo_tool(local_history.get_last().content, tool_output, task, tool)

        elif len(tools) > 1:
            local_history = copy.deepcopy(self.history)

            tools_presentation: str = "* " + "\n* ".join([
                f"Name: '{tool.tool_name}' - Usage: {tool._function_prototype} - Description: {tool.function_description}"
                for tool in tools])
            tool_ack_prompt = f"You have access to this list of tools definitions you can use to fulfill tasks :\n{tools_presentation}\nPlease acknowledge the given tools."
            self._chat(local_history, tool_ack_prompt)

            tool_use_decision: str = f"You have a task to solve. I will give it to you between these tags `<task></task>`. However, your actual job is to decide if you need to use any of the available tools to solve the task or not. If you do need tools then output their names. The task to solve is <task>{task}</task> So, would any tools be useful in relation to the given task ?"
            self._chat(local_history, tool_use_decision, images=images)

            tool_router: str = "In order to summarize your previous answer in one word. Did you chose to use any tools ? Respond ONLY by 'yes' or 'no'."
            ai_may_use_tools: str = self._chat(local_history, tool_router, save_to_history=False)

            if "yes" in ai_may_use_tools.lower():
                self.history.add(Message(MessageRole.USER, task))
                self.history.add(
                    Message(MessageRole.ASSISTANT, "I should use tools related to the task to solve it correctly."))
                while True:
                    tool: Tool = self._choose_tool_by_name(local_history, tools)
                    tool_training_history = copy.deepcopy(local_history)
                    tool_examples_prompt = 'To use the tool you MUST extract each parameter and use it as a JSON key like this: {"arg1": "<value1>", "arg2": "<value2>"}. You must respect arguments type. For instance, the tool `getWeather(city: str, lat: int, long: int)` would be structured like this {"city": "new-york", "lat": 10, "lon": 20}. In our case, the tool call you must use must look like that: ' + str(
                        {key: ("arg " + str(i)) for i, key in enumerate(tool._function_args)})
                    self._chat(tool_training_history, tool_examples_prompt)

                    tool_training_history._concat_history(tool._get_examples_as_history())

                    tool_use: str = "Now that I showed you examples on how the tool is used it's your turn. Output the tool as valid JSON."
                    self._chat(tool_training_history, tool_use, images=images, json_output=True)  # !!Actual function calling
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
                self._chat(self.history, task, images=images, json_output=json_output)

        return self.history.get_last()

    def _chat(self, history: History, query: str, images: List[str] | None = None, json_output=False, structured_output: Type[T] | None = None, save_to_history: bool = True, stream: bool = False, tools: List[Tool] | None = None) -> str | Iterator:
        if save_to_history is True:
            history.add(Message(MessageRole.USER, query, images=images))
        else:
            history_save = copy.deepcopy(history)
            history_save.add(Message(MessageRole.USER, query, images=images))

        logging.info(f"[PROMPT][To: {self.name}]: {query}")
        inference = InferenceFactory.get_inference(self.server_type)
        response: InferenceOutput = inference.go(model_name=self.model_name,
                                                 history=history.get_as_dict() if save_to_history is True else history_save.get_as_dict(),
                                                 endpoint=self.endpoint,
                                                 api_token=self.api_token,
                                                 model_settings=self.model_settings.get_settings(),
                                                 stream=stream,
                                                 json_output=(True if json_output is True else False),
                                                 structured_output=structured_output,
                                                 headers=self.headers,
                                                 tools=tools,
                                                 images=images
                                                 )
        if stream is True:
            return response.raw_llm_response  # Only for the simple_chat() method and is of no importance.
        logging.info(f"[AI_RESPONSE][From: {self.name}]: {response.raw_llm_response}")
        if save_to_history is True:
            print("tool call id = ", response.tool_call_id)
            history.add(Message(MessageRole.ASSISTANT, response.raw_llm_response, structured_output=response.structured_output, tool_call_id=response.tool_call_id))
        return response.raw_llm_response
