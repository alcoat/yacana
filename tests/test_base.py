import unittest
import os
import tempfile
from yacana import OllamaAgent
from yacana import OpenAiAgent
from yacana import OllamaModelSettings, OpenAiModelSettings

class BaseAgentTest(unittest.TestCase):
    """Base class for agent tests with common setup and utilities."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        # Create temporary directory for test files
        cls.temp_dir = tempfile.mkdtemp()
        
        # Initialize agents with minimal settings
        cls.ollama_settings = OllamaModelSettings(
            temperature=0.1
        )
        
        cls.openai_settings = OpenAiModelSettings(
            temperature=0.1
        )
        
        # Get API tokens from environment variables
        openai_api_token = os.getenv('OPENAI_API_TOKEN')
        if not openai_api_token:
            raise unittest.SkipTest("OPENAI_API_TOKEN environment variable not set")
        
        # Initialize agents
        cls.ollama_agent = OllamaAgent(
            name="AI assistant",
            model_name="llama3.2:latest",
            model_settings=cls.ollama_settings,
            system_prompt="You are a helpful AI assistant",
            endpoint="http://127.0.0.1:11434"
        )
        
        cls.ollama_vision_agent = OllamaAgent(
            name="AI assistant",
            model_name="llama3.2-vision:latest",
            model_settings=cls.ollama_settings,
            system_prompt="You are a helpful AI assistant",
            endpoint="http://127.0.0.1:11434"
        )
        
        cls.openai_agent = OpenAiAgent(
            name="AI assistant",
            model_name="gpt-4o-mini",
            model_settings=cls.openai_settings,
            system_prompt="You are a helpful AI assistant",
            api_token=openai_api_token
        )
        
        # VLLM agent uses OpenAiAgent with different endpoint
        cls.vllm_agent = OpenAiAgent(
            name="AI assistant",
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            model_settings=cls.openai_settings,
            system_prompt="You are a helpful AI assistant",
            endpoint="http://localhost:8000/v1",
            api_token=openai_api_token
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures after running tests."""
        # Clean up temporary files
        for file in os.listdir(cls.temp_dir):
            os.remove(os.path.join(cls.temp_dir, file))
        os.rmdir(cls.temp_dir)

    def get_test_image_path(self, filename):
        """Get the path to a test image file."""
        return os.path.join("tests", "assets", filename) 