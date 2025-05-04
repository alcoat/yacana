import unittest
import os
from tests.test_base import BaseAgentTest
from yacana import Task
from yacana.exceptions import IllogicalConfiguration

class TestStreaming(BaseAgentTest):
    """Test streaming capabilities of all agent types."""

    def test_basic_streaming(self):
        """Test basic streaming functionality."""
        prompt = "Count from 1 to 5 (no additionnal text, numbers only):"
        expected_numbers = {"1", "2", "3", "4", "5"}
        received_numbers = set()
        
        def streaming_callback(chunk: str):
            self.assertIsInstance(chunk, str)
            for num in expected_numbers:
                if num in chunk:
                    received_numbers.add(num)
        
        # Test Ollama agent
        if self.run_ollama:
            received_numbers.clear()
            message = Task(prompt, self.ollama_agent, streaming_callback=streaming_callback).solve()
            self.assertEqual(received_numbers, expected_numbers)
        
        # Test OpenAI agent
        if self.run_openai:
            received_numbers.clear()
            message = Task(prompt, self.openai_agent, streaming_callback=streaming_callback).solve()
            self.assertEqual(received_numbers, expected_numbers)
        
        # Test VLLM agent
        if self.run_vllm:
            received_numbers.clear()
            message = Task(prompt, self.vllm_agent, streaming_callback=streaming_callback).solve()
            self.assertEqual(received_numbers, expected_numbers)

    def test_streaming_with_media(self):
        """Test streaming with image input."""
        image_path = self.get_test_image_path("burger.jpg")
        prompt = "Describe this image in one sentence:"
        received_chunks = []
        
        def streaming_callback(chunk: str):
            self.assertIsInstance(chunk, str)
            received_chunks.append(chunk)
        
        # Test Ollama agent
        if self.run_ollama:
            received_chunks.clear()
            message = Task(prompt, self.ollama_vision_agent, medias=[image_path], streaming_callback=streaming_callback).solve()
            full_response = "".join(received_chunks)
            self.assertGreater(len(full_response), 0)
            self.assertTrue(
                any(keyword in full_response.lower() for keyword in ['food', 'burger', 'meal', 'dish', 'eating']),
                "Response should mention food-related content"
            )
        
        # Test OpenAI agent
        if self.run_openai:
            received_chunks.clear()
            message = Task(prompt, self.openai_agent, medias=[image_path], streaming_callback=streaming_callback).solve()
            full_response = "".join(received_chunks)
            self.assertGreater(len(full_response), 0)
            self.assertTrue(
                any(keyword in full_response.lower() for keyword in ['food', 'burger', 'meal', 'dish', 'eating']),
                "Response should mention food-related content"
            )
        
        # Test VLLM agent
        #if self.run_vllm:
        #    received_chunks.clear()
        #    message = Task(prompt, self.vllm_agent, medias=[image_path], streaming_callback=streaming_callback).solve()
        #    full_response = "".join(received_chunks)
        #    self.assertGreater(len(full_response), 0)
        #    self.assertTrue(
        #        any(keyword in full_response.lower() for keyword in ['food', 'burger', 'meal', 'dish', 'eating']),
        #        "Response should mention food-related content"
        #    )

    def test_streaming_with_structured_output(self):
        """Test that using both streaming and structured output raises IllogicalConfiguration."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        prompt = "Generate a person's information with name and age:"
        
        def streaming_callback(chunk: str):
            pass  # This should never be called
        
        # Test Ollama agent
        if self.run_ollama:
            with self.assertRaises(IllogicalConfiguration):
                Task(prompt, self.ollama_agent, structured_output=Person, streaming_callback=streaming_callback).solve()
        
        # Test OpenAI agent
        if self.run_openai:
            with self.assertRaises(IllogicalConfiguration):
                Task(prompt, self.openai_agent, structured_output=Person, streaming_callback=streaming_callback).solve()
        
        # Test VLLM agent
        if self.run_vllm:
            with self.assertRaises(IllogicalConfiguration):
                Task(prompt, self.vllm_agent, structured_output=Person, streaming_callback=streaming_callback).solve()

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