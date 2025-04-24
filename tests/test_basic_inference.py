import unittest
import os
from tests.test_base import BaseAgentTest
from yacana import Task, Message, MessageRole, History, HistorySlot, OpenAiModelSettings

class TestBasicInference(BaseAgentTest):
    """Test basic inference capabilities of all agent types."""

    def test_simple_completion(self):
        """Test basic text completion with all agent types."""
        prompt = "Count from 1 to 5 (no additionnal text, numbers only):"
        expected = "1, 2, 3, 4, 5"
        
        # Test Ollama agent
        if self.run_ollama:
            message = Task(prompt, self.ollama_agent).solve()
            self.assertIn("1", message.content)
            self.assertIn("5", message.content)
        
        # Test OpenAI agent
        if self.run_openai:
            message = Task(prompt, self.openai_agent).solve()
            self.assertIn("1", message.content)
            self.assertIn("5", message.content)
        
        # Test VLLM agent
        if self.run_vllm:
            message = Task(prompt, self.vllm_agent).solve()
            self.assertIn("1", message.content)
            self.assertIn("5", message.content)

    def test_history_management(self):
        """Test basic history management operations."""
        history = History()
        
        # Test adding messages using Message class
        user_message = Message(MessageRole.USER, "Hello, how are you?")
        history.add_message(user_message)
        
        # Test adding messages using Task with all agents
        if self.run_ollama:
            assistant_message = Task("Respond to the greeting", self.ollama_agent).solve()
            history.add_message(assistant_message)
        
        if self.run_openai:
            assistant_message = Task("Respond to the greeting", self.openai_agent).solve()
            history.add_message(assistant_message)
        
        if self.run_vllm:
            assistant_message = Task("Respond to the greeting", self.vllm_agent).solve()
            history.add_message(assistant_message)
        
        # Verify history structure
        expected_slots = sum([self.run_ollama, self.run_openai, self.run_vllm]) + 1  # +1 for user message
        self.assertEqual(len(history.slots), expected_slots)
        
        # Test getting messages as dictionary
        messages_dict = history.get_messages_as_dict()
        self.assertEqual(len(messages_dict), expected_slots)
        self.assertEqual(messages_dict[0]["role"], "user")
        for i in range(1, expected_slots):
            self.assertEqual(messages_dict[i]["role"], "assistant")

    def test_slot_management(self):
        """Test slot management operations."""
        history = History()
        
        # Create a slot with multiple messages
        slot = HistorySlot()
        user_message = Message(MessageRole.USER, "What's the weather like?")
        slot.add_message(user_message)
        
        # Add slot to history
        history.add_slot(slot)
        
        # Test changing default message with all agents
        if self.run_ollama:
            new_message = Task("Describe the weather", self.ollama_agent).solve()
            slot.add_message(new_message)
        
        if self.run_openai:
            new_message = Task("Describe the weather", self.openai_agent).solve()
            slot.add_message(new_message)
        
        if self.run_vllm:
            new_message = Task("Describe the weather", self.vllm_agent).solve()
            slot.add_message(new_message)
        
        # Change the selected message and verify
        expected_messages = sum([self.run_ollama, self.run_openai, self.run_vllm]) + 1  # +1 for user message
        self.assertEqual(len(slot.messages), expected_messages)
        
        # Test keeping only selected message
        slot.set_main_message_index(1)  # Select the first assistant message
        slot.keep_only_selected_message()
        self.assertEqual(len(slot.messages), 1)
        self.assertEqual(slot.get_message().role, MessageRole.ASSISTANT)

    def test_message_labelling(self):
        """Test message labelling system."""
        history = History()
        
        # Create messages with tags for all agents
        user_message = Message(MessageRole.USER, "What's the weather like?", tags=["weather", "query"])
        history.add_message(user_message)
        
        if self.run_ollama:
            assistant_message = Task("Describe the weather", self.ollama_agent).solve()
            assistant_message.add_tag("weather")
            history.add_message(assistant_message)
        
        if self.run_openai:
            assistant_message = Task("Describe the weather", self.openai_agent).solve()
            assistant_message.add_tag("weather")
            history.add_message(assistant_message)
        
        if self.run_vllm:
            assistant_message = Task("Describe the weather", self.vllm_agent).solve()
            assistant_message.add_tag("weather")
            history.add_message(assistant_message)
        
        # Test getting messages by tags
        weather_messages = history.get_messages_by_tags(["weather"])
        expected_messages = sum([self.run_ollama, self.run_openai, self.run_vllm]) + 1  # +1 for user message
        self.assertEqual(len(weather_messages), expected_messages)
        
        # Test strict tag matching
        weather_messages = history.get_messages_by_tags(["weather", "nonexistent"], strict=True)
        self.assertEqual(len(weather_messages), 0)
        
        # Test non-strict tag matching
        weather_messages = history.get_messages_by_tags(["weather", "nonexistent"], strict=False)
        self.assertEqual(len(weather_messages), expected_messages)

    def test_multiple_responses(self):
        """Test getting multiple responses from the model."""
        # Test OpenAI agent
        if self.run_openai:
            settings = OpenAiModelSettings(temperature=0.1, n=2)
            self.openai_agent.model_settings = settings
            task = Task("Count from 1 to 3", self.openai_agent)
            slot = task.solve()
            self.assertGreater(len(slot.messages), 1)
            for message in slot.messages:
                self.assertIsInstance(message.content, str)
                self.assertGreater(len(message.content), 0)
        
        # Test VLLM agent (which also supports multiple responses)
        if self.run_vllm:
            settings = OpenAiModelSettings(temperature=0.1, n=2)
            self.vllm_agent.model_settings = settings
            task = Task("Count from 1 to 3", self.vllm_agent)
            slot = task.solve()
            self.assertGreater(len(slot.messages), 1)
            for message in slot.messages:
                self.assertIsInstance(message.content, str)
                self.assertGreater(len(message.content), 0)
        
        # Note: Ollama doesn't support multiple responses (n parameter), so we skip it

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        super().setUpClass()
        
        # Get which agents to run from environment variables
        cls.run_ollama = os.getenv('TEST_OLLAMA', 'true').lower() == 'true'
        cls.run_openai = os.getenv('TEST_OPENAI', 'true').lower() == 'true'
        cls.run_vllm = os.getenv('TEST_VLLM', 'true').lower() == 'true'

if __name__ == '__main__':
    unittest.main() 