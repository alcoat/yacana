import unittest
import os
from tests.test_base import BaseAgentTest
from yacana import Task, Message, MessageRole, History, HistorySlot, OpenAiModelSettings
from yacana.generic_agent import GenericAgent


class TestMcp(BaseAgentTest):
    """Test basic inference capabilities of all agent types."""

    def setUp(self):
        """Clean up agent histories before each test."""
        super().setUp()
        if self.run_ollama:
            self.ollama_agent.history.clean()
        if self.run_openai:
            self.openai_agent.history.clean()
        if self.run_vllm:
            self.vllm_agent.history.clean()

    def test_connection(self):
        """Test connection to MCP servers and checks that the number of tools matches the expected number for that server."""
        pass

    def test_error_handling(self):
        """
        Test for mixing tool execution types YACANA and OPENAI, which should raise an error.
        """

    def test_local_tool_override_mcp_tool(self):
        """
        Test that local tools override MCP tools with the same name.
        """
        pass

    def test_forget_about_tool(self):
        """
        Test that the forget_tool method works as expected.
        """
        pass

    def test_tool_types(self):
        """
        Test that the tool types are correctly set when getting tools using .get_tools_as() method.
        """
        pass

    def test_tools(self):
        """
        Calls the tools and make sure the output is as expected.
        Use tool execution types YACANA and OPENAI to test the tools.
        Use agent type Ollama and OpenAi to test the tools.
        """
        pass


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