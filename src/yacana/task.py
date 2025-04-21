import copy
import uuid
from typing import List, Type, Callable, Dict
from pydantic import BaseModel

from .generic_agent import GenericAgent, GenericMessage
from .exceptions import MaxToolErrorIter, IllogicalConfiguration
from .history import History
from .logging_config import LoggerManager
from .tool import Tool

LoggerManager.set_library_log_level("httpx", "WARNING")


class Task:
    """
    A class representing a task to be solved by an LLM agent.

    The best approach to use an LLM is to define a task with a clear objective.
    Then assign an LLM to this task so that it can try to solve it. You can also add Tools
    so that when the LLM starts solving it gets some tools relevant to the current task.
    This means that tools are not specific to an agent but to the task at hand. This allows
    more flexibility by producing less confusion to the LLM as it gets access to the tools
    it needs only when faced with a task that is related.

    Parameters
    ----------
    prompt : str
        The task to solve. It is the prompt given to the assigned LLM.
    agent : GenericAgent
        The agent assigned to this task.
    json_output : bool, optional
        If True, will force the LLM to answer as JSON. Defaults to False.
    structured_output : Type[BaseModel] | None, optional
        The expected structured output type for the task. If provided, the LLM's response
        will be validated against this type. Defaults to None.
    tools : List[Tool], optional
        A list of tools that the LLM will get access to when trying to solve this task.
        Defaults to an empty list.
    medias : List[str] | None, optional
        An optional list of paths pointing to images on the filesystem. Defaults to None.
    raise_when_max_tool_error_iter : bool, optional
        If True, raises MaxToolErrorIter when tool errors exceed the limit. If False,
        returns None instead. Defaults to True.
    llm_stops_by_itself : bool, optional
        Only useful when the task is part of a GroupSolve(). Signals the assigned LLM
        that it will have to stop talking by its own means. Defaults to False.
    use_self_reflection : bool, optional
        Only useful when the task is part of a GroupSolve(). Allows keeping the self
        reflection process done by the LLM in the next GS iteration. Defaults to False.
    forget : bool, optional
        When True, the Agent won't remember this task after completion. Useful for
        routing purposes. Defaults to False.
    streaming_callback : Callable | None, optional
        Optional callback for streaming responses. Defaults to None.
    runtime_config : Dict | None, optional
        Optional runtime configuration for the task. Defaults to None.

    Raises
    ------
    IllogicalConfiguration
        If both tools and structured_output are provided, or if both streaming_callback
        and structured_output are provided.
    """

    def __init__(self, prompt: str, agent: GenericAgent, json_output=False, structured_output: Type[BaseModel] | None = None, tools: List[Tool] = None,
                 medias: List[str] | None = None, raise_when_max_tool_error_iter: bool = True,
                 llm_stops_by_itself: bool = False, use_self_reflection=False, forget=False, streaming_callback: Callable | None = None, runtime_config: Dict | None = None) -> None:
        self.prompt: str = prompt
        self.agent: GenericAgent = agent
        self.json_output: bool = json_output
        self.structured_output: Type[BaseModel] | None = structured_output
        self.tools: List[Tool] = tools if tools is not None else []
        self.raise_when_max_tool_error_iter: bool = raise_when_max_tool_error_iter  # @todo remove this feature : Not a huge fan. Maybe add a callbak that would be called when we reach max iter instead of raising
        self.llm_stops_by_itself: bool = llm_stops_by_itself
        self.use_self_reflection: bool = use_self_reflection
        self.forget: bool = forget
        self.medias: List[str] | None = medias
        self._uuid: str = str(uuid.uuid4())
        self.streaming_callback: Callable | None = streaming_callback
        self.runtime_config = runtime_config if runtime_config is not None else {}

        if len(self.tools) > 0 and self.structured_output is not None:
            raise IllogicalConfiguration("You can't have tools and structured_output at the same time. The tool output will be considered the LLM output hence not using the structured output.")

        if self.streaming_callback is not None and self.structured_output is not None:
            raise IllogicalConfiguration("You can't have streaming_callback and structured_output at the same time. Having incomplete JSON is useless.")

        # Only used when @forget is True
        self.save_history: History | None = None

        #self._update_tool_schema_if_openai()

    @property
    def uuid(self) -> str:
        """
        Get the unique identifier for this task.

        Returns
        -------
        str
            A unique task identifier.
        """
        return self._uuid

    def add_tool(self, tool: Tool) -> None:
        """
        Add a tool to the list of tools available for this task.

        Parameters
        ----------
        tool : Tool
            The tool to add to the task's tool list.
        """
        self.tools.append(tool)

    def solve(self) -> GenericMessage | None:
        """
        Execute the task using the assigned LLM agent.

        This method will call the assigned LLM to perform inference on the task's prompt.
        If tools are available, the LLM may use them, potentially making multiple calls
        to the inference server.

        Returns
        -------
        GenericMessage | None
            The last message from the LLM, or None if the task failed and
            raise_when_max_tool_error_iter is False.

        Raises
        ------
        MaxToolErrorIter
            If tool errors exceed the limit and raise_when_max_tool_error_iter is True.
        """
        if self.forget is True:
            self.save_history: History = copy.deepcopy(self.agent.history)
        try:
            answer: GenericMessage = self.agent._interact(self.prompt, self.tools, self.json_output, self.structured_output, self.medias, self.streaming_callback, self.runtime_config)
        except MaxToolErrorIter as e:
            if self.raise_when_max_tool_error_iter:
                self._reset_agent_history()
                raise MaxToolErrorIter(e.message)
            else:
                self._reset_agent_history()
                # By default, we would raise but if @raise_when_max_tool_error_iter is False we simply return None.
                # An uncaught exceptions would be fatal so returning None is a good way to continue execution after a failure.
                return None  #@todo What happens in GroupSolve if the no raise option is used ?
        self._reset_agent_history()
        return answer

    def _reset_agent_history(self) -> None:
        """
        Reset the agent's history if the task is marked to be forgotten.
        """
        if self.forget is True:
            self.agent.history = self.save_history
