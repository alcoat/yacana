import copy
import json
import uuid
from datetime import datetime
from enum import Enum
from typing import List, Dict, Type, T, Any, Sequence
import importlib
from pydantic import BaseModel
from typing_extensions import Self
from abc import ABC, abstractmethod

from .medias import Media


class MessageRole(Enum):
    """The available types of message creators.
    It can either be a message from the user or an answer from the LLM to the user's message.
    Developers can also set a system prompt that guides the LLM into a specific way of answering.

    """
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class SlotPosition(Enum):
    """
    The position of a slot in the history. This is only a syntactic sugar to make the code more readable.

    """
    BOTTOM = -1
    TOP = -2


class ToolCall:

    def __init__(self, call_id, name, arguments):
        self.call_id: str = call_id
        self.name: str = name
        self.arguments: dict = arguments

    def get_tool_call_as_dict(self):
        return {
            "id": self.call_id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments)
            }
        }

    def _export(self):
        return {
            "id": self.call_id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments
            }
        }


class GenericMessage(ABC):
    """The smallest entity representing an interaction with the LLM. Can either be a user message or an LLM message.
    Use the MessageRole() enum to specify who's message it is.

    Attributes
    ----------
    role : MessageRole
        From whom is the message from. See the MessageRole Enum
    content : str
        The actual message
    medias : List[str] | None
        An optional list of path pointing to images on the filesystem.
    Methods
    ----------
    get_as_dict(self) -> Dict

    """

    _registry = {}

    def __init__(self, role: MessageRole, content: str | None = None, tool_calls: List[ToolCall] | None = None, medias: List[str] | None = None, structured_output: Type[T] | None = None, tool_call_id: str = None, tags: List[str] | None = None, id: uuid.UUID | None = None) -> None:
        """
        Returns an instance of Message
        :param role: MessageRole: From whom is the message from. See the MessageRole Enum
        :param content: str : The actual message
        :param medias: List[str] | None : An optional list of path pointing to images or else on the filesystem.
        :param structured_output: Type[T] | None : An optional structured output that can be used to store the result of a tool call
        :param tool_calls: str | None : An optional unique identifier for the tool call (used by openAI to match the tool call with the response)
        """
        self.id = str(uuid.uuid4()) if id is None else str(id)
        self.role: MessageRole = role
        self.content: str | None = content
        self.tool_calls: List[ToolCall] | None = tool_calls
        self.medias: List[str] = medias if medias is not None else []
        self.structured_output: Type[T] | None = structured_output
        self.tool_call_id: str | None = tool_call_id
        self.tags: List[str] = tags if tags is not None else []

        # Checking that both @message and @tool_calls are neither None nor empty at the same time
        if content is None and (tool_calls is None or (tool_calls is not None and len(tool_calls) == 0)):
            raise ValueError("A Message must have a content or a tool call that is not None or [].")

        #if structured_output is not None:
        #    self.dict_to_structured_output(structured_output, globals())

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        GenericMessage._registry[cls.__name__] = cls

    def _export(self) -> Dict:
        """
        Returns a pure python dictionary mainly to save the object into a file.
        None entries are omitted.
        @return: Dict
        """
        members = self.__dict__.copy()
        members["type"] = self.__class__.__name__
        members["role"] = self.role.value
        if members["structured_output"] is not None:
            members["structured_output"] = self.structured_output_to_dict()
        return members

    @staticmethod
    def create_instance(members: Dict):
        #  Converting the role string to its matching enum
        members["role"]: MessageRole = next((role for role in MessageRole if role.value == members["role"]), None)
        #if members["structured_output"] is not None:
        #    members["structured_output"] = GenericMessage.dict_to_structured_output(members["structured_output"], globals())
        cls_name = members.pop("type")
        cls = GenericMessage._registry.get(cls_name)
        #print("normalement ca ca devrait être les id du json = ", members["id"])
        message = cls(**members)
        if message.structured_output is not None:
            message.structured_output = message.dict_to_structured_output(members["structured_output"])
        return message

    @abstractmethod
    def get_message_as_dict(self):
        raise(NotImplementedError("This method should be implemented in the child class"))

    def get_as_pretty(self) -> str:
        if self.content is not None:
            return self.content
        else:
            return json.dumps([tool_call.get_tool_call_as_dict() for tool_call in self.tool_calls])

    def add_tag(self, tag: str) -> None:
        """
        Adds a tag to the message. Tags are used to filter messages in the history.
        :param tag: str
        :return: None
        """
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """
        Removes a tag from the message. Tags are used to filter messages in the history.
        :param tag: str
        :return: None
        """
        if tag in self.tags:
            self.tags.remove(tag)

    def structured_output_to_dict(self) -> Dict:
        """
        Returns the structured output as a dictionary.
        :return: Any
        """
        raise NotImplemented(f"This method should be implemented in the child Message class when 'structured_output' is not None. Message type: {self.__class__.__name__}.")

    def dict_to_structured_output(self, data: Dict):
        raise NotImplemented(f"This method should be implemented in the child Message class when 'structured_output' is not None. Message type: {self.__class__.__name__}.")

    def __str__(self) -> str:
        """
        Override of str() to pretty print.
        :return: str
        """
        return json.dumps(self._export())


class Message(GenericMessage):
    """
    For Yacana users or simple text based interactions
    """

    def __init__(self, role: MessageRole, content: str, tags: List[str] = None, **kwargs) -> None:
        super().__init__(role, content, tags=tags, id=kwargs.get('id', None))

    def get_message_as_dict(self):  # @todo A revoir car là c'est générique donc bah le user on ne sait pas contre quel server il va l'envoyer
        return {
            "role": self.role.value,
            "content": self.content
        }

class OpenAIUserMessage(GenericMessage):

    def __init__(self, role: MessageRole, content: str, tags: List[str] = None, medias: List[str] = None, structured_output: Type[T] = None, **kwargs):
        super().__init__(role, content, medias=medias, structured_output=structured_output, tags=tags, id=kwargs.get('id', None))

    def get_message_as_dict(self):
        message_as_dict = {
            "role": self.role.value,
            "content": self.content
        }
        if self.medias is not None:
            message_as_dict["content"] = [
                {"type": "text", "text": self.content},
                *[Media.get_as_openai_dict(media_path) for media_path in self.medias]
            ]
        return message_as_dict

    def structured_output_to_dict(self) -> Dict:
        return {
            "__class__": f"{self.structured_output.__module__}.{self.structured_output.__name__}"
        }

    def dict_to_structured_output(self, data: Dict):
        full_class_path = data["__class__"]
        module_name, class_name = full_class_path.rsplit(".", 1)

        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)

        return cls

class OpenAITextMessage(GenericMessage):
    
    def __init__(self, role: MessageRole, content: str, tags: List[str] = None, **kwargs):
        tool_calls = None
        structured_output = None
        tool_call_id = None
        super().__init__(role, content, tool_calls, None, structured_output, tool_call_id, tags=tags, id=kwargs.get('id', None))

    def get_message_as_dict(self):
        return {
            "role": self.role.value,
            "content": self.content
        }


class OpenAIMediaMessage(GenericMessage):

    def __init__(self, role: MessageRole, content: str, medias: List[str], tags: List[str] = None, **kwargs):
        tool_calls = None
        structured_output = None
        tool_call_id = None
        super().__init__(role, content, tool_calls, medias, structured_output, tool_call_id, tags=tags, id=kwargs.get('id', None))

    def get_message_as_dict(self):
        message = {
            "role": self.role.value,
            "content": [
                {"type": "text", "text": self.content}
            ]
        }

        # **({'images': self.medias} if self.medias is not None else {}),
        for media in self.medias:
            message["content"].append({"type": "image_url", "image_url": {
                "url": media
            }})
        return message


class OpenAIFunctionCallingMessage(GenericMessage):

    def __init__(self, tool_calls: List[ToolCall], tags: List[str] = None, **kwargs):
        role = MessageRole.ASSISTANT
        content = None
        medias = None
        structured_output = None
        tool_call_id = None
        super().__init__(role, content, tool_calls, medias, structured_output, tool_call_id, tags=tags, id=kwargs.get('id', None))

    def get_message_as_dict(self):
        return {
            "role": self.role.value,
            "content": self.content,
            **({'tool_calls': [tool_call.get_tool_call_as_dict() for tool_call in self.tool_calls]} if self.tool_calls is not None else {})
        }


class OpenAIToolCallingMessage(GenericMessage):

    def __init__(self, content: str, tool_call_id: str, tags: List[str] = None, **kwargs):
        """
        Response from the LLM when a tool is called.
        :param content: In this context it will fe the output of the tool. We keep the variable name as is for consistency with the other messages.
        :param tool_call_id:
        :param tags:
        """
        role = MessageRole.TOOL
        tool_calls = None
        medias = None
        structured_output = None
        super().__init__(role, content, tool_calls, medias, structured_output, tool_call_id, tags=tags, id=kwargs.get('id', None))

    def get_message_as_dict(self):
        return {
            "role": self.role.value,
            "content": self.content,
            ** ({'tool_call_id': self.tool_call_id} if self.tool_call_id is not None else {}),
        }


class OpenAIStructuredOutputMessage(GenericMessage):

    def __init__(self, role: MessageRole, content: str, structured_output: Type[T], tags: List[str] = None, **kwargs):
        tool_calls = None
        medias = None
        tool_call_id = None
        super().__init__(role, content, tool_calls, medias, structured_output, tool_call_id, tags=tags, id=kwargs.get('id', None))

    def get_message_as_dict(self):
        return {
            "role": self.role.value,
            "content": self.content,
        }

    def structured_output_to_dict(self) -> Dict:
        return {
            "__class__": f"{self.structured_output.__class__.__module__}.{self.structured_output.__class__.__name__}",
            "data": self.structured_output.model_dump()
        }

    def dict_to_structured_output(self, data: Dict):
        full_class_path = data["__class__"]
        module_name, class_name = full_class_path.rsplit(".", 1)

        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)

        return cls(**data["data"])



class OllamaUserMessage(GenericMessage):

    def __init__(self, role: MessageRole, content: str, tags: List[str] = None, medias: List[str] = None, structured_output: Type[T] = None, **kwargs):
        super().__init__(role, content, medias=medias, structured_output=structured_output, tags=tags, id=kwargs.get('id', None))

    def get_message_as_dict(self):
        return {
            "role": self.role.value,
            "content": self.content,
            **({'images': self.medias} if self.medias is not None else {}),
        }

    def structured_output_to_dict(self) -> Dict:
        return {
            "__class__": f"{self.structured_output.__module__}.{self.structured_output.__name__}"
        }

    def dict_to_structured_output(self, data: Dict):
        full_class_path = data["__class__"]
        module_name, class_name = full_class_path.rsplit(".", 1)

        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)

        return cls


class OllamaTextMessage(GenericMessage):

    def __init__(self, role: MessageRole, content: str, tags: List[str] = None, **kwargs):
        super().__init__(role, content, tags=tags, id=kwargs.get('id', None))

    def get_message_as_dict(self):
        return {
            "role": self.role.value,
            "content": self.content,
            **({'images': self.medias} if self.medias is not None else {}),
        }


class OllamaStructuredOutputMessage(GenericMessage):

    def __init__(self, role: MessageRole, content: str, structured_output: Type[T], tags: List[str] = None, **kwargs):
        tool_calls = None
        medias = None
        tool_call_id = None
        super().__init__(role, content, tool_calls, medias, structured_output, tool_call_id, tags=tags, id=kwargs.get('id', None))

    def get_message_as_dict(self):
        return {
            "role": self.role.value,
            "content": self.content,
        }

    def structured_output_to_dict(self) -> Dict:
        return {
            "__class__": f"{self.structured_output.__class__.__module__}.{self.structured_output.__class__.__name__}",
            "data": self.structured_output.model_dump()
        }

    def dict_to_structured_output(self, data: Dict):
        full_class_path = data["__class__"]
        module_name, class_name = full_class_path.rsplit(".", 1)

        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)

        return cls(**data["data"])


class HistorySlot:

    def __init__(self, messages: List[GenericMessage] = None, raw_llm_json: str = None, **kwargs):
        self.id = str(kwargs.get('id', uuid.uuid4()))
        self.creation_time: int = int(kwargs.get('creation_time', datetime.now().timestamp()))
        self.messages: List[GenericMessage] = [] if messages is None else messages
        self.raw_llm_json: str = raw_llm_json
        self.currently_selected_message_index: int = 0

    def set_message_index(self, message_index: int) -> None:
        """
        Sets the index of the currently selected message in the slot.
        :param message_index: int : The index of the message to select
        :return: None
        """
        if message_index >= len(self.messages):
            raise IndexError("Index out of range: The message index is greater than the number of messages in the slot.")
        self.currently_selected_message_index = message_index

    def get_message_index(self) -> int:
        """
        Returns the index of the currently selected message in the slot.
        :return: int : The index of the currently selected message
        """
        return self.currently_selected_message_index

    def add_message(self, message: GenericMessage):
        self.messages.append(message)

    def get_message(self, message_index: int | None = None) -> GenericMessage:
        if message_index is not None and message_index >= len(self.messages):
            raise IndexError("Index out of range: The message index is greater than the number of messages in the slot.")
        if message_index is None:
            return self.messages[self.currently_selected_message_index]
        else:
            return self.messages[message_index]

    def get_all_messages(self) -> List[GenericMessage]:
        return self.messages

    def set_raw_llm_json(self, raw_llm_json: str) -> None:
        self.raw_llm_json = raw_llm_json

    def delete_message_by_index(self, message_index: int) -> None:
        self.messages.pop(message_index)

    def delete_message_by_id(self, message_id: str) -> None:
        for i, message in enumerate(self.messages):
            if message.id == message_id:
                self.messages.pop(i)

    def keep_only_selected_message(self):
        for i in range(len(self.messages)):
            if i != self.currently_selected_message_index:
                self.messages.pop(i)

    @staticmethod
    def create_instance(members: Dict):
        members["messages"] = [GenericMessage.create_instance(message) for message in members["messages"] if message is not None]
        return HistorySlot(**members)

    def _export(self) -> Dict:
        members = self.__dict__.copy()
        members["type"] = self.__class__.__name__
        members["messages"] = [message._export() for message in self.messages if message is not None]
        return members


class History:
    """
    Container for an alternation of Messages representing a conversation between the user and an LLM

    Attributes
    ----------

    Methods
    ----------
    add(message: Message) -> None
    get_as_dict() -> List[Dict]
    pretty_print() -> None
    create_check_point() -> str
    load_check_point(uid: str) -> None
    get_last() -> Message
    clean() -> None
    __str__() -> str
    """

    def __init__(self, **kwargs) -> None:
        """
        Returns a History instance
        """
        self.slots: List[HistorySlot] = kwargs.get('slots', [])
        self._checkpoints: Dict[str, list[HistorySlot]] = kwargs.get('_checkpoints', {})

    def add_slot(self, history_slot: HistorySlot, position: int | SlotPosition = SlotPosition.BOTTOM) -> None:
        """
        Adds a new slot to the history.
        The history is not a list of Message() as one would expect but is instead a list of HistorySlot(). Each slot has one or more Message()
        with one Message being the main message of the conversation. In short the History is a list of slots that each contains messages.

        @param history_slot: HistorySlot : A new slot to add to the history.
        @param position: int | SlotPosition : The position where to add the new slot.
                         You can use the SlotPosition Enum to specify the position or set the index directly.
        @return: None
        """
        if isinstance(position, SlotPosition):
            if position == SlotPosition.BOTTOM:
                self.slots.append(history_slot)
            elif position == SlotPosition.TOP:
                self.slots.insert(0, history_slot)
        else:
            self.slots.insert(position, history_slot)

    def delete_slot(self, index: int) -> None:
        """
        Deletes a slot from the history. This will remove all the messages within the slot.
        @param index: int : The index of the slot to delete
        @return: None
        """
        self.slots.pop(index)

    def get_last_slot(self) -> HistorySlot:
        """
        Returns the last slot of the history. Not very useful but a good syntactic sugar to get the last item from
        the conversation
        @return: HistorySlot
        """
        if len(self.slots) <= 0:
            raise IndexError("History is empty (no slots, so no messages)")
        return self.slots[-1]

    def add_message(self, message: GenericMessage) -> None:
        """
        Adds a new slot (HistorySlot) to the history containing the Message. each slot has a main message that compose the complete conversation.
        @param message: Message : A message with information about the sender type and the content of the message.
        @return: None
        """
        self.slots.append(HistorySlot([message]))

    def _export(self) -> Dict:
        """
        Returns the RAW history (aka list of slots and messages) as a pure python dictionary
        @return: The list of slots and messages as a python dictionary
        """
        members_as_dict = self.__dict__.copy()

        # Exporting checkpoints
        for uid, slots in self._checkpoints.items():
            exported_slots = [slot._export() for slot in slots]
            members_as_dict["_checkpoints"][uid] = exported_slots

        # Exporting slots
        slots_list: List[Dict] = []
        for slot in self.slots:
            slots_list.append(slot._export())
        members_as_dict["slots"] = slots_list
        return members_as_dict

    @staticmethod
    def create_instance(members: Dict):
        """
        !!Warning!! This is a concatenation of the given dict to the existing one.
        Loads a list of messages as raw List. ie: [
        {
            "role": "system",
            "content": "You are an AI assistant"
        },
        ...
        ]
        @param messages_dict: A python dictionary
        @return: None
        """

        # Loading slots
        members["slots"] = [HistorySlot.create_instance(slot) for slot in members["slots"]]

        # Loading checkpoints
        for uid, slot in members["_checkpoints"]:
            members["_checkpoints"][uid] = [HistorySlot.create_instance(slot) for slot in slot]
        return History(**members)

    def get_messages_as_dict(self) -> List[Dict]:
        formated_messages = []
        for slot in self.slots:
            formated_messages.append(slot.get_message().get_message_as_dict())
        return formated_messages

    def pretty_print(self) -> None:
        """
        Prints the history on the std with shinny colors
        @return: None
        """
        for slot in self.slots:
            message = slot.get_message()
            if message.role == MessageRole.USER:
                print('\033[92m[' + message.role.value + "]:\n" + message.get_as_pretty() + '\033[0m')
            elif message.role == MessageRole.ASSISTANT:
                print('\033[95m[' + message.role.value + "]:\n" + message.get_as_pretty() + '\033[0m')
            elif message.role == MessageRole.SYSTEM:
                print('\033[93m[' + message.role.value + "]:\n" + message.get_as_pretty() + '\033[0m')
            elif message.role == MessageRole.TOOL:
                print('\033[96m[' + message.role.value + "]:\n" + message.get_as_pretty() + '\033[0m')
            print("")

    def create_check_point(self) -> str:
        """
        Saves the current history so that you can load it back later. Useful when you want to keep a clean history in
        a flow that didn't worked out as expected and want to rollback in time.
        @return: str : A unique identifier that can be used to load the checkpoint at any time.
        """
        uid: str = str(uuid.uuid4())
        self._checkpoints[uid] = copy.deepcopy(self.slots)
        return uid

    def load_check_point(self, uid: str) -> None:
        """
        Loads the history saved from a particular checkpoint in time.
        It replaces the current history with the loaded one. Perfect for a timey wimey rollback in time.
        @param uid:
        @return:
        """
        self.slots = self._checkpoints[uid]

    def get_message(self, index) -> GenericMessage:
        """
        Returns the message at the given index
        @param index: int : The index of the message to return
        @return: Message
        """
        if len(self.slots) <= 0:
            raise IndexError("History is empty (no slots, so no messages)")
        return self.slots[index].get_message()

    def get_messages_by_tags(self, tags: List[str], strict=True) -> Sequence[GenericMessage]:
        """
        Returns all the messages that have the given tags.
        @param tags: List[str] : The tags to filter the messages
        @param strict: bool : If True, the message must have all the tags. If False, the message can have any of the tags.
        @return: List[Message]
        """
        if len(self.slots) <= 0:
            raise IndexError("History is empty (no slots, so no messages)")
        messages = []
        for slot in self.slots:
            if strict is True:
                if set(tags).issubset(set(slot.get_message().tags)):
                    messages.append(slot.get_message())
            else:
                if set(tags).intersection(set(slot.get_message().tags)):
                    messages.append(slot.get_message())
        return messages

    def get_last_message(self) -> GenericMessage:
        """
        Returns the last message of the history. Not very useful but a good syntactic sugar to get the last item from
        the conversation
        @return: Message
        """
        if len(self.slots) <= 0:
            raise IndexError("History is empty (no slots, so no messages)")
        return self.slots[-1].get_message()

    def get_all_messages(self) -> List[Message]:
        """
        Returns a list of Messages. Each message is the main message of each slots.
        @return: Message
        """
        if len(self.slots) <= 0:
            raise IndexError("History is empty (no slots, so no messages)")
        return [slot.get_message() for slot in self.slots]

    def clean(self) -> None:
        """
        Resets the history, preserving only the initial system prompt
        @return: None
        """
        if len(self.slots) > 0 and self.slots[0].get_message().role == MessageRole.SYSTEM:
            self.slots = [self.slots[0]]
        else:
            self.slots = []

    def _concat_history(self, history: Self) -> None:
        """
        Concat a History instance to the current history.
        @param history:
        @return:
        """
        self.slots = self.slots + history.slots

    def __str__(self) -> str:
        """
        Override str() for pretty print
        :return: str
        """
        result = []
        for slot in self.slots:
            result.append(slot.get_message()._export())
        return json.dumps(result)
