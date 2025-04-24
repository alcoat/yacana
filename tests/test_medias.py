import unittest
import os
from tests.test_base import BaseAgentTest
from yacana import Task

class TestMediasInference(BaseAgentTest):
    """Test media inference capabilities of all agent types."""

    def test_image_description(self):
        """Test image description capabilities."""
        burger_path = self.get_test_image_path("burger.jpg")
        prompt = "Describe this image in one sentence:"
        
        # Test Ollama agent
        if self.run_ollama:
            message = Task(prompt, self.ollama_vision_agent, medias=[burger_path]).solve()
            self.assertIsInstance(message.content, str)
            self.assertGreater(len(message.content), 0)
            # Check for burger-related terms in the description
            self.assertTrue(
                any(term in message.content.lower() for term in ["burger", "hamburger", "sandwich", "bun", "patty"]),
                f"Expected burger-related terms in description, got: {message.content}"
            )
        
        # Test OpenAI agent
        if self.run_openai:
            message = Task(prompt, self.openai_agent, medias=[burger_path]).solve()
            self.assertIsInstance(message.content, str)
            self.assertGreater(len(message.content), 0)
            # Check for burger-related terms in the description
            self.assertTrue(
                any(term in message.content.lower() for term in ["burger", "hamburger", "sandwich", "bun", "patty"]),
                f"Expected burger-related terms in description, got: {message.content}"
            )
        
        # Test VLLM agent
        if self.run_vllm:
            message = Task(prompt, self.vllm_agent, medias=[burger_path]).solve()
            self.assertIsInstance(message.content, str)
            self.assertGreater(len(message.content), 0)
            # Check for burger-related terms in the description
            self.assertTrue(
                any(term in message.content.lower() for term in ["burger", "hamburger", "sandwich", "bun", "patty"]),
                f"Expected burger-related terms in description, got: {message.content}"
            )

    def test_multi_image_comparison(self):
        """Test comparing multiple images."""
        burger_path = self.get_test_image_path("burger.jpg")
        flower_path = self.get_test_image_path("flower.png")
        prompt = "Compare these two images in one sentence:"
        
        # Note: Ollama doesn't support multi-image comparison, so we skip it
        
        # Test OpenAI agent
        if self.run_openai:
            message = Task(prompt, self.openai_agent, medias=[burger_path, flower_path]).solve()
            self.assertIsInstance(message.content, str)
            self.assertGreater(len(message.content), 0)
            # Check for both burger and flower-related terms in the comparison
            self.assertTrue(
                any(term in message.content.lower() for term in ["burger", "hamburger", "sandwich", "bun", "patty"]),
                f"Expected burger-related terms in comparison, got: {message.content}"
            )
            self.assertTrue(
                any(term in message.content.lower() for term in ["flower", "bloom", "petal", "plant", "garden"]),
                f"Expected flower-related terms in comparison, got: {message.content}"
            )
        
        # Test VLLM agent
        if self.run_vllm:
            message = Task(prompt, self.vllm_agent, medias=[burger_path, flower_path]).solve()
            self.assertIsInstance(message.content, str)
            self.assertGreater(len(message.content), 0)
            # Check for both burger and flower-related terms in the comparison
            self.assertTrue(
                any(term in message.content.lower() for term in ["burger", "hamburger", "sandwich", "bun", "patty"]),
                f"Expected burger-related terms in comparison, got: {message.content}"
            )
            self.assertTrue(
                any(term in message.content.lower() for term in ["flower", "bloom", "petal", "plant", "garden"]),
                f"Expected flower-related terms in comparison, got: {message.content}"
            )

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