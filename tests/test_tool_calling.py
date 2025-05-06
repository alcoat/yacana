import unittest
import os
from tests.test_base import BaseAgentTest
from yacana import Task, Tool, ToolError

def add(number_one: int, number_two: int) -> int:
    """Add two numbers together."""
    if type(number_one) is not int or type(number_two) is not int:
        raise ToolError("Both arguments must be integers.")
    return number_one + number_two

def subtract(number_one: int, number_two: int) -> int:
    """Subtract the second number from the first."""
    if type(number_one) is not int or type(number_two) is not int:
        raise ToolError("Both arguments must be integers.")
    return number_one - number_two

def multiply(number_one: int, number_two: int) -> int:
    """Multiply two numbers together."""
    if type(number_one) is not int or type(number_two) is not int:
        raise ToolError("Both arguments must be integers.")
    return number_one * number_two

def divide(number_one: int, number_two: int) -> int:
    """Divide the first number by the second."""
    if type(number_one) is not int or type(number_two) is not int:
        raise ToolError("Both arguments must be integers.")
    if number_two == 0:
        raise ToolError("Division by zero is not allowed.")
    return number_one // number_two

def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The current real time weather from Weather.com in {city} is sunny with a high of 25°C."

class TestToolCalling(BaseAgentTest):
    """Test tool calling capabilities of all agent types."""

    def setUp(self):
        """Set up test fixtures before each test."""
        super().setUp()
        # Create tools
        self.addition = Tool("Addition", "Add two integer numbers and returns the result.", add)
        self.subtraction = Tool("Subtraction", "Subtracts two integer numbers and returns the result.", subtract)
        self.multiplication = Tool("Multiplication", "Multiplies two integer numbers and returns the result.", multiply)
        self.division = Tool("Division", "Divides two integer numbers and returns the result.", divide)
        self.get_weather = Tool("Get_weather", "Calls a weather API and returns the current weather in the given city.", get_weather)

    def test_basic_arithmetic(self):
        """Test basic arithmetic operations using tools."""
        prompt = "Calculate 2+4-6*7 by decomposing the operations step by step and according to order of operations (PEMDAS/BODMAS). Use the provided tools. Do not make the math yourself. Only use the tools."
        
        # Test Ollama agent
        if self.run_ollama:
            message = Task(prompt, self.ollama_agent, tools=[self.addition, self.subtraction, self.multiplication, self.division]).solve()
            self.assertTrue(
                any(result in message.content for result in ["-36", "40"]),
                f"Expected either -36 or 40 in the result, got: {message.content}"
            )

        
        # Test OpenAI agent
        if self.run_openai:
            message = Task(prompt, self.openai_agent, tools=[self.addition, self.subtraction, self.multiplication, self.division]).solve()
            self.assertTrue(
                any(result in message.content for result in ["-36", "40"]),
                f"Expected either -36 or 40 in the result, got: {message.content}"
            )
        
        # Test VLLM agent
        #if self.run_vllm:
        #    message = Task(prompt, self.vllm_agent, tools=[self.addition, self.subtraction, self.multiplication, self.division]).solve()
        #    self.assertTrue(
        #        any(result in message.content for result in ["-36", "40"]),
        #        f"Expected either -36 or 40 in the result, got: {message.content}"
        #    )

    def test_weather_tool(self):
        """Test weather information tool."""
        prompt = "What's the weather in Paris?"
        
        # Test Ollama agent
        if self.run_ollama:
            message = Task(prompt, self.ollama_agent, tools=[self.get_weather]).solve()
            self.assertIn("Paris", message.content)
            self.assertIn("sunny", message.content.lower())
            self.assertIn("25", message.content)
        
        # Test OpenAI agent
        if self.run_openai:
            message = Task(prompt, self.openai_agent, tools=[self.get_weather]).solve()
            self.assertIn("Paris", message.content)
            self.assertIn("sunny", message.content.lower())
            self.assertIn("25", message.content)
        
        # Test VLLM agent
        if self.run_vllm:
            message = Task(prompt, self.vllm_agent, tools=[self.get_weather]).solve()
            self.assertIn("Paris", message.content)
            self.assertIn("sunny", message.content.lower())
            self.assertIn("25", message.content)


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