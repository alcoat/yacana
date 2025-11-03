import copy
import json
import uuid
from datetime import datetime
from enum import Enum, unique
from typing import List, Dict, T, Sequence
from typing_extensions import Self
import logging

from .TokenCount import count_tokens_using_huggingface, HFMessage
from .exceptions import SpecializedTokenCountingError, IllogicalConfiguration
from .messages import GenericMessage, Message, MessageRole


@unique
class SlotPosition(Enum):
    """
    <b>ENUM:</b> The position of a slot in the history. This is only a syntactic sugar to make the code more readable.

    Attributes
    ----------
    BOTTOM : int
        The slot is at the bottom of the history.
    TOP : int
        The slot is at the top of the history.
    """
    BOTTOM = -1
    TOP = -2


class HistorySlot:
    """
    A slot is a container for messages. It can contain one or more messages.
    Most of the time it will only contain one message but when using `n=2` or`n=x` in the OpenAI API, it will contain multiple variations hence multiple messages.

    Parameters
    ----------
    messages : List[GenericMessage], optional
        A list of messages. Each message is a variation of the main message (defined by the
        @main_message_index parameter).
    raw_llm_json : str, optional
        The raw LLM JSON response for the slot. This is the raw JSON from the inference server.
        When using OpenAI this may contain more than one message hence the slot system acts as a
        container for the messages.
    **kwargs
        Additional keyword arguments including:
        id : str, optional
            The unique identifier for the slot.
        creation_time : int, optional
            The timestamp when the slot was created.

    Attributes
    ----------
    id : str
        The unique identifier for the slot.
    creation_time : int
        The timestamp when the slot was created.
    messages : List[GenericMessage]
        List of messages in the slot.
    raw_llm_json : str | None
        The raw LLM JSON response for the slot.
    main_message_index : int
        The index of the main message in the slot.
    """

    def __init__(self, messages: List[GenericMessage] = None, raw_llm_json: str = None, **kwargs):
        self.id = str(kwargs.get('id', uuid.uuid4()))
        self.creation_time: int = int(kwargs.get('creation_time', datetime.now().timestamp()))
        self.messages: List[GenericMessage] = [] if messages is None else messages
        self.raw_llm_json: str = raw_llm_json
        self.main_message_index: int = 0

    def set_main_message_index(self, message_index: int) -> None:
        """
        A slot can contain any number of concurrent message. But only one can be the main slot message and actually be part of the History.
        This method sets the index of the main message within the list of available messages in the slot.

        Parameters
        ----------
        message_index : int
            The index of the message to select as the main message.

        Raises
        ------
        IndexError
            If the message index is greater than the number of messages in the slot.
        """
        if message_index >= len(self.messages):
            raise IndexError("Index out of range: The message index is greater than the number of messages in the slot.")
        self.main_message_index = message_index

    def get_main_message_index(self) -> int:
        """
        Returns the index of the main message in the slot.

        Returns
        -------
        int
            The index of the currently selected message.
        """
        return self.main_message_index

    def add_message(self, message: GenericMessage):
        """
        Adds a new message to the slot.

        Parameters
        ----------
        message : GenericMessage
            The message to add to the slot.
        """
        self.messages.append(message)

    def get_message(self, message_index: int | None = None) -> GenericMessage:
        """
        Returns the main message of the slot or the one at the given index if index is provided.

        Parameters
        ----------
        message_index : int | None, optional
            The index of the message to return. If None, returns the currently selected message.

        Returns
        -------
        GenericMessage
            The requested message.

        Raises
        ------
        IndexError
            If the message index is greater than the number of messages in the slot.
        IllogicalConfiguration
            An HistorySlot should not be empty (no messages).
        """
        if message_index is None:
            return self.messages[self.main_message_index]
        if message_index >= len(self.messages):
            raise IndexError("Index out of range: The message index is greater than the number of messages in the slot.")
        else:
            return self.messages[message_index]

    def get_all_messages(self) -> List[GenericMessage]:
        """
        Returns all the messages in the slot.

        Returns
        -------
        List[GenericMessage]
            All messages in the slot.
        """
        return self.messages

    def set_raw_llm_json(self, raw_llm_json: str) -> None:
        """
        Sets the raw LLM JSON response for the slot.
        This is the raw JSON from the inference server. When using OpenAI this may contain more than one message hence the slot system acts as a container for the messages.

        Parameters
        ----------
        raw_llm_json : str
            The raw JSON response from the LLM.
        """
        self.raw_llm_json = raw_llm_json

    def _delete_message_by_index(self, message_index: int) -> None:
        """
        Deletes a message from the slot by index. An HistorySlot should NOT be empty (no messages). Use this method at your own risk.
        To delete a message from a slot, delete it using the `delete_message` method from the History class instead.
        This way, if the slot ends being empty it will be cleaned by the History class.

        Parameters
        ----------
        message_index : int
            The index of the message to delete.

        Raises
        ------
        IndexError
            If the message index is greater than the number of messages in the slot,
            or if trying to delete the last message in the slot.
        """
        if message_index >= len(self.messages):
            raise IndexError("Index out of range: The message index is greater than the number of messages in the slot.")
        
        if len(self.messages) <= 1:
            raise IndexError("Cannot delete the last message in a slot. Delete the slot from the history instead.")
        
        self.messages.pop(message_index)
        
        # Always reset to the first message for simplicity
        self.main_message_index = 0
        logging.debug("Main message index reset to 0 after message deletion")

    def _delete_message_by_id(self, message_id: str) -> None:
        """
        Deletes a message from the slot by id. An HistorySlot should NOT be empty (no messages). Use this method at your own risk.
        To delete a message from a slot, delete it using the `delete_message` method from the History class instead.
        This way, if the slot ends being empty it will be cleaned by the History class.

        Parameters
        ----------
        message_id : str
            The ID of the message to delete.

        Raises
        ------
        IndexError
            If trying to delete the last message in the slot.
        """
        for i, message in enumerate(self.messages):
            if message.id == message_id:
                self._delete_message_by_index(i)
                break

    def keep_only_selected_message(self):
        """
        Keeps only the currently selected message in the slot and deletes all the others.
        If there's only one message, this method does nothing.

        Raises
        ------
        IndexError
            If there are no messages in the slot.
        """
        if len(self.messages) == 0:
            raise IndexError("Cannot operate on an empty slot.")
            
        if len(self.messages) == 1:
            logging.debug("Keeping only selected message: Only one message in slot. Nothing to do.")
            return
            
        # Store the main message
        main_message = self.messages[self.main_message_index]
        
        # Clear all messages
        self.messages.clear()
        
        # Add back only the main message
        self.messages.append(main_message)
        self.main_message_index = 0
        logging.debug("Main message index reset to 0 after keeping only the selected message.")

    @staticmethod
    def create_instance(members: Dict):
        """
        Creates an instance of the HistorySlot class from a dictionary.
        Mainly used to import the object from a file.

        Parameters
        ----------
        members : Dict
            Dictionary containing the slot data.

        Returns
        -------
        HistorySlot
            A new instance of HistorySlot.
        """
        members["messages"] = [GenericMessage.create_instance(message) for message in members["messages"] if message is not None]
        return HistorySlot(**members)

    def _export(self) -> Dict:
        """
        Returns the slot as a dictionary.
        Mainly used to export the object to a file.

        Returns
        -------
        Dict
            A dictionary representation of the slot.
        """
        members = self.__dict__.copy()
        members["type"] = self.__class__.__name__
        members["messages"] = [message._export() for message in self.messages if message is not None]
        return members


class History:
    """
    Container for an alternation of Messages representing a conversation between the user and an LLM.
    To be precise, the history is a list of slots and not actual messages. Each slot contains at least one or more messages.
    This class does its best to hide the HistorySlot implementation. Meaning that many methods allows you to deal with the messages directly, but under the hood it always manages the slot wrapper.

    Parameters
    ----------
    llm_model_name : str | None, optional
        The name of the LLM model. Used to count tokens more accurately when using an OpenAi model. Will use Tiktoken under the hood.
    hugging_face_details: HuggingFaceDetails | None, optional
        Details for Hugging Face models, including repo name and access token. Used to count tokens more accurately when using an HuggingFace model. Will use the transformers library under the hood.
    **kwargs: Any
        Additional keyword arguments including:
        slots : List[HistorySlot], optional
            List of history slots.
        _checkpoints : Dict[str, list[HistorySlot]], optional
            Dictionary of checkpoints for the history.

    Attributes
    ----------
    slots : List[HistorySlot]
        List of history slots.
    _checkpoints : Dict[str, list[HistorySlot]]
        Dictionary of checkpoints for the history.
    llm_model_name : str | None
        The name of the LLM model used to count tokens when using an OpenAi model.
    """

    def __init__(self, llm_model_name: str | None = None, **kwargs) -> None:
        self.slots: List[HistorySlot] = kwargs.get('slots', [])
        self._checkpoints: Dict[str, list[HistorySlot]] = kwargs.get('_checkpoints', {})
        self.llm_model_name = llm_model_name

    def add_slot(self, history_slot: HistorySlot, position: int | SlotPosition = SlotPosition.BOTTOM) -> None:
        """
        Adds a new slot to the history at the specified position.

        Parameters
        ----------
        history_slot : HistorySlot
            The slot to add to the history.
        position : int | SlotPosition, optional
            The position where to add the slot. Can be an integer or a SlotPosition enum value.
            Defaults to SlotPosition.BOTTOM.
        """
        if isinstance(position, SlotPosition):
            if position == SlotPosition.BOTTOM:
                self.slots.append(history_slot)
            elif position == SlotPosition.TOP:
                self.slots.insert(0, history_slot)
        else:
            self.slots.insert(position, history_slot)

    def get_last_slot(self) -> HistorySlot:
        """
        Returns the last slot of the history. A good syntactic sugar to get the last item from the conversation.

        Returns
        -------
        HistorySlot
            The last slot in the history.

        Raises
        ------
        IndexError
            If the history is empty.
        """
        if len(self.slots) <= 0:
            raise IndexError("History is empty (no slots, so no messages)")
        return self.slots[-1]

    def get_slot_by_index(self, index: int) -> HistorySlot:
        """
        Returns the slot at the given index.

        Parameters
        ----------
        index : int
            The index of the slot to return.

        Returns
        -------
        HistorySlot
            The slot at the given index.

        Raises
        ------
        IndexError
            If the history is empty or the index is out of range.
        """
        if len(self.slots) <= 0:
            raise IndexError("History is empty (no slots, so no messages)")
        return self.slots[index]

    def get_slot_by_id(self, id: str) -> HistorySlot:
        """
        Returns the slot with the given ID.

        Parameters
        ----------
        id : str
            The ID of the slot to return.

        Returns
        -------
        HistorySlot
            The slot with the given ID.

        Raises
        ------
        IndexError
            If the history is empty.
        ValueError
            If no slot with the given ID is found.
        """
        if len(self.slots) <= 0:
            raise IndexError("History is empty (no slots, so no messages)")
        for slot in self.slots:
            if slot.id == id:
                return slot
        raise ValueError(f"Slot with id {id} not found in history.")

    def get_slot_by_message(self, message: GenericMessage) -> HistorySlot:
        """
        Returns the slot containing the given message.

        Parameters
        ----------
        message : GenericMessage
            The message to search for.

        Returns
        -------
        HistorySlot
            The slot containing the message.

        Raises
        ------
        IndexError
            If the history is empty.
        ValueError
            If no slot contains the given message.
        """
        if len(self.slots) <= 0:
            raise IndexError("History is empty (no slots, so no messages)")
        for slot in self.slots:
            if any(message.id == msg.id for msg in slot.get_all_messages()):
                return slot
        raise ValueError(f"Message with id {message.id} not found in history.")

    def add_message(self, message: GenericMessage) -> HistorySlot:
        """
        Adds a new message to the history by creating a new slot.

        Parameters
        ----------
        message : GenericMessage
            The message to add to the history.

        Returns
        -------
        HistorySlot
            The new slot containing the message added to the history. Useful for chaining.
        """
        slot = HistorySlot([message])
        self.slots.append(slot)
        return slot

    def delete_slot(self, slot: HistorySlot) -> None:
        """
        Deletes a slot from the history.

        Parameters
        ----------
        slot : HistorySlot
            The slot to delete.

        Raises
        ------
        ValueError
            If the slot is not found in the history.
        """
        if slot not in self.slots:
            raise ValueError("Slot not found in history.")

        self.slots.remove(slot)
        logging.debug(f"Slot {slot} deleted from history.")

    def delete_slot_by_id(self, slot_id: str) -> None:
        """
        Deletes a slot from the history by its ID. If the ID does not exist, it logs a warning.

        Parameters
        ----------
        slot_id : str
            The ID of the slot to delete.
        """
        slot_to_delete = next((slot for slot in self.slots if slot.id == slot_id), None)
        if not slot_to_delete:
            logging.warning(f"No slot found with ID {slot_id}.")
        else:
            self.slots.remove(slot_to_delete)
            logging.debug(f"Slot with ID {slot_id} deleted from history.")

    def delete_message(self, message: Message) -> None:
        """
        Deletes a message from all slots in the history.

        Parameters
        ----------
        message : Message
            The message to delete.

        Raises
        ------
        ValueError
            If the message is not found in any slot.
        """
        for slot in self.slots:
            if message in slot.messages:
                if len(slot.messages) == 1:
                    self.delete_slot(slot)
                    logging.debug(f"Slot {slot} deleted from history because it contained only the message to delete.")
                else:
                    slot._delete_message_by_id(message.id)
                    logging.debug(f"Message {message} deleted from slot {slot}.")
                return
        raise ValueError("Message not found in any slot.")

    def delete_message_by_id(self, message_id: str) -> None:
        """
        Deletes a message from all slots in the history by its ID.

        Parameters
        ----------
        message_id : str
            The ID of the message to delete. If the ID does not exist, it logs a warning.
        """
        for slot in self.slots:
            message_to_delete = next((message for message in slot.messages if message.id == message_id), None)
            if message_to_delete:
                if len(slot.messages) == 1:
                    self.delete_slot(slot)
                    logging.debug(f"Slot {slot} deleted from history because it contained only the message to delete.")
                else:
                    slot._delete_message_by_id(message_to_delete.id)
                    logging.debug(f"Message with ID {message_id} deleted from slot {slot}.")
                return
        logging.warning(f"No message found with ID {message_id}.")

    def _export(self) -> Dict:
        """
        Returns the history as a dictionary.
        Mainly used to export the object to a file.

        Returns
        -------
        Dict
            A dictionary representation of the history.
        """
        members_as_dict = self.__dict__.copy()
        members_as_dict["_checkpoints"] = {}

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
        Creates a new instance of History from a dictionary.

        Parameters
        ----------
        members : Dict
            Dictionary containing the history data.

        Returns
        -------
        History
            A new instance of History.
        """
        # Loading slots
        members["slots"] = [HistorySlot.create_instance(slot) for slot in members["slots"]]

        # Loading checkpoints
        for uid, slots in members["_checkpoints"].items():
            members["_checkpoints"][uid] = [HistorySlot.create_instance(slot) for slot in slots]
        return History(**members)

    def get_messages_as_dict(self) -> List[Dict]:
        """
        Returns all messages in the history as a list of dictionaries.

        Returns
        -------
        List[Dict]
            List of message dictionaries.
        """
        formated_messages = []
        for slot in self.slots:
            formated_messages.append(slot.get_message().get_message_as_dict())
        return formated_messages

    def pretty_print(self) -> None:
        """
        Prints the history to stdout with colored output.
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
        Creates a checkpoint of the current history state.

        Returns
        -------
        str
            A unique identifier for the checkpoint.
        """
        uid: str = str(uuid.uuid4())
        self._checkpoints[uid] = copy.deepcopy(self.slots)
        return uid

    def load_check_point(self, uid: str) -> None:
        """
        Loads a checkpoint of the history. Perfect for a timey wimey rollback in time.

        Parameters
        ----------
        uid : str
            The unique identifier of the checkpoint to load.
        """
        self.slots = self._checkpoints[uid]

    def get_message(self, index) -> GenericMessage:
        """
        Returns the message at the given index.

        Parameters
        ----------
        index : int
            The index of the message to return.

        Returns
        -------
        GenericMessage
            The message at the given index.

        Raises
        ------
        IndexError
            If the history is empty or the index is out of range.
        """
        if len(self.slots) <= 0:
            raise IndexError("History is empty (no slots, so no messages)")
        return self.slots[index].get_message()

    def get_messages_by_tags(self, tags: List[str], strict=False) -> Sequence[GenericMessage]:
        """
        Returns messages that match the given tags based on the matching mode.

        Parameters
        ----------
        tags : List[str]
            The tags to filter messages by.
        strict : bool, optional
            Controls the matching mode:
            - If False (default), returns messages that have ANY of the specified tags.
              For example, searching for ["tag1"] will match messages with ["tag1", "tag2"].
              This is useful for broad filtering.
            - If True, returns messages that have EXACTLY the specified tags (and possibly more).
              For example, searching for ["tag1", "tag2"] will match messages with ["tag1", "tag2", "tag3"]
              but not messages with just ["tag1"] or ["tag2"].
              This is useful for precise filtering.

        Returns
        -------
        Sequence[GenericMessage]
            List of messages matching the tag criteria.

        Raises
        ------
        IndexError
            If the history is empty.

        Examples
        --------
        # Find all messages with tag1 (broad matching)
        `history.get_messages_by_tags(["tag1"])`
        # Returns messages with ["tag1"], ["tag1", "tag2"], etc.

        # Find messages with exactly tag1 and tag2 (strict matching)
        `history.get_messages_by_tags(["tag1", "tag2"], strict=True)`
        # Returns messages with ["tag1", "tag2"], ["tag1", "tag2", "tag3"], etc.
        # But not messages with just ["tag1"] or ["tag2"]
        """
        if len(self.slots) <= 0:
            raise IndexError("History is empty (no slots, so no messages)")
        messages = []
        for slot in self.slots:
            message_tags = set(slot.get_message().tags)
            if strict is False:
                # Non-strict mode: message must have ANY of the specified tags
                if set(tags).intersection(message_tags):
                    messages.append(slot.get_message())
            else:
                # Strict mode: message must have ALL specified tags
                if set(tags).issubset(message_tags):
                    messages.append(slot.get_message())


        return messages

    def get_last_message(self) -> GenericMessage:
        """
        Returns the last message in the history.

        Returns
        -------
        GenericMessage
            The last message in the history.

        Raises
        ------
        IndexError
            If the history is empty.
        """
        if len(self.slots) <= 0:
            raise IndexError("History is empty (no slots, so no messages)")
        return self.slots[-1].get_message()

    def get_all_messages(self) -> List[Message]:
        """
        Returns all messages in the history.

        Returns
        -------
        List[Message]
            List of all messages in the history.

        Raises
        ------
        IndexError
            If the history is empty.
        """
        if len(self.slots) <= 0:
            raise IndexError("History is empty (no slots, so no messages)")
        return [slot.get_message() for slot in self.slots]

    def clean(self) -> None:
        """
        Resets the history, preserving only the initial system prompt if present.
        """
        if len(self.slots) > 0 and self.slots[0].get_message().role == MessageRole.SYSTEM:
            self.slots = [self.slots[0]]
        else:
            self.slots = []

    def _concat_history(self, history: Self) -> None:
        """
        Concatenates another history to this one.

        Parameters
        ----------
        history : Self
            The history to concatenate.
        """
        self.slots = self.slots + history.slots

    def get_token_count(self, hugging_face_repo_name: str | None = None, hugging_face_token: str | None = None, padding_per_message: int = 4, evaluate_all_history_as_one: bool = False) -> int:
        """
        Get the total token count of messages in the history.

        * If the hugging_face_repo_name is provided, the token count will be calculated using the transformers library.
        (If the llm is gated (private), the hugging_face_token must be provided to access the repo.)
        * If no Hugging Face details are given and the model is recognized as an OpenAi model then the token count will be calculated using the tiktoken library.
        * If none of the above conditions are met, an approximative token count will be returned. This is only a rough estimate and should not be used for precise calculations.

        Note that using Tiktoken and transformers are precise but quite slow (loging to HF using the token is the worst). The approximative token count is very fast but not precise.
        Important point: If you provide Hugging Face details, Yacana will try to evaluate the token count using the transformers library first even if the model is from OpenAi. If the Hugging Face evaluation fails then it won't fallback to Tiktoken but to the approximative method instead.

        Parameters
        ----------
        hugging_face_repo_name : str | None, optional
            The name of the Hugging Face repository for the model. If provided, the token count will be calculated using the transformers library.
        hugging_face_token : str | None, optional
            The Hugging Face access token for private models. Required if the model is gated (private).
        padding_per_message : int, optional
            The number of extra tokens to add per message for padding. Default is 4.
        evaluate_all_history_as_one : bool, optional
            If True, evaluates the entire history as a single input using the full chat template for token counting.
            This is more accurate but only available for Hugging Face models. Default is False.

        Returns
        -------
        int
            The total token count of all messages in history.
        """
        if evaluate_all_history_as_one is True and hugging_face_repo_name is None:
            raise IllogicalConfiguration("Parameter `evaluate_all_history_as_one` can only be used when hugging_face_repo_name is provided. It allows to count the tokens of the entire history at once using the full chat template instead of applying the template one message at a time which is less accurate. But it's only available for hugging face models that's why you need to set `hugging_face_repo_name`.")
        if hugging_face_token is not None and hugging_face_repo_name is None:
            raise IllogicalConfiguration("Parameter `hugging_face_token` can only be used when `hugging_face_repo_name` is provided. The token is used to access private models on Hugging Face Hub.")

        try:
            if hugging_face_repo_name and evaluate_all_history_as_one is True:
                logging.debug("Will try to count tokens using the whole history instead of message per message.")
                hf_messages_list: List[HFMessage] = [message.get_message_as_hugging_face_dict() for message in self.get_all_messages()]
                return count_tokens_using_huggingface(hf_messages_list, hugging_face_repo_name, hugging_face_token=hugging_face_token) + padding_per_message * len(hf_messages_list)
        except SpecializedTokenCountingError:
            logging.warning("Falling back to per-message token counting even though `evaluate_all_history_as_one` was True due to error in hugging_face token counting.")

        tokens = 0
        for message in self.get_all_messages():
            tokens += message.get_token_count(llm_model_name=self.llm_model_name, hugging_face_repo_name=hugging_face_repo_name, hugging_face_token=hugging_face_token, padding_per_message=padding_per_message)
        return tokens

    def __str__(self) -> str:
        """
        Returns a string representation of the history.

        Returns
        -------
        str
            A JSON string representation of the history.
        """
        result = []
        for slot in self.slots:
            result.append(slot.get_message()._export())
        return json.dumps(result)
