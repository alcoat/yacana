import unittest
import os
from tests.test_base import BaseAgentTest
from yacana import Task

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

    def test_image_description(self):
        """Test image description capabilities."""
        burger_path = self.get_test_image_path("burger.jpg")
        prompt = "Describe this image in one sentence:"
        
        # Test Ollama agent
        if self.run_ollama:
            message = Task(prompt, self.ollama_agent, medias=[burger_path]).solve()
            self.assertIsInstance(message.content, str)
            self.assertGreater(len(message.content), 0)
        
        # Test OpenAI agent
        if self.run_openai:
            message = Task(prompt, self.openai_agent, medias=[burger_path]).solve()
            self.assertIsInstance(message.content, str)
            self.assertGreater(len(message.content), 0)
        
        # Test VLLM agent
        if self.run_vllm:
            message = Task(prompt, self.vllm_agent, medias=[burger_path]).solve()
            self.assertIsInstance(message.content, str)
            self.assertGreater(len(message.content), 0)

    def test_multi_image_comparison(self):
        """Test comparing multiple images."""
        burger_path = self.get_test_image_path("burger.jpg")
        flower_path = self.get_test_image_path("flower.png")
        prompt = "Compare these two images in one sentence:"
        
        # Test Ollama agent
        if self.run_ollama:
            message = Task(prompt, self.ollama_agent, medias=[burger_path, flower_path]).solve()
            self.assertIsInstance(message.content, str)
            self.assertGreater(len(message.content), 0)
        
        # Test OpenAI agent
        if self.run_openai:
            message = Task(prompt, self.openai_agent, medias=[burger_path, flower_path]).solve()
            self.assertIsInstance(message.content, str)
            self.assertGreater(len(message.content), 0)
        
        # Test VLLM agent
        if self.run_vllm:
            message = Task(prompt, self.vllm_agent, medias=[burger_path, flower_path]).solve()
            self.assertIsInstance(message.content, str)
            self.assertGreater(len(message.content), 0)

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