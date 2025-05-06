import unittest
import os
from tests.test_base import BaseAgentTest
from yacana import Task
from pydantic import BaseModel

class CountryFact(BaseModel):
    """A fact about a country."""
    name: str
    fact: str

class Facts(BaseModel):
    """Collection of country facts."""
    countryFacts: list[CountryFact]

class TestStructuredOutput(BaseAgentTest):
    """Test structured output capabilities of all agent types."""

    def test_basic_structured_output(self):
        """Test basic structured output with a simple prompt."""
        prompt = "Tell me 3 facts about Canada."
        
        # Test Ollama agent
        if self.run_ollama:
            message = Task(prompt, self.ollama_agent, structured_output=Facts).solve()
            self._validate_facts_response(message.structured_output)
        
        # Test OpenAI agent
        if self.run_openai:
            message = Task(prompt, self.openai_agent, structured_output=Facts).solve()
            self._validate_facts_response(message.structured_output)
        
        # Test VLLM agent
        if self.run_vllm:
            message = Task(prompt, self.vllm_agent, structured_output=Facts).solve()
            self._validate_facts_response(message.structured_output)

    def test_structured_output_with_context(self):
        """Test structured output with additional context in the prompt."""
        prompt = """I'm writing a travel guide. Please provide 3 interesting facts about France.
        For each fact, include both the name of the fact and a detailed description.
        Make sure the facts are unique and interesting for tourists."""
        
        # Test Ollama agent
        if self.run_ollama:
            message = Task(prompt, self.ollama_agent, structured_output=Facts).solve()
            self._validate_facts_response(message.structured_output)
            # Additional validation for context-specific requirements
            for fact in message.structured_output.countryFacts:
                self.assertGreater(len(fact.fact), 50, "Facts should be detailed for a travel guide")
        
        # Test OpenAI agent
        if self.run_openai:
            message = Task(prompt, self.openai_agent, structured_output=Facts).solve()
            self._validate_facts_response(message.structured_output)
            for fact in message.structured_output.countryFacts:
                self.assertGreater(len(fact.fact), 50, "Facts should be detailed for a travel guide")
        
        # Test VLLM agent
        if self.run_vllm:
            message = Task(prompt, self.vllm_agent, structured_output=Facts).solve()
            self._validate_facts_response(message.structured_output)
            for fact in message.structured_output.countryFacts:
                self.assertGreater(len(fact.fact), 50, "Facts should be detailed for a travel guide")

    def test_structured_output_with_media(self):
        """Test structured output with media input."""
        image_path = self.get_test_image_path("burger.jpg")
        prompt = "Tell me 3 facts about what is depicted in this image."
        
        # Test Ollama agent
        if self.run_ollama:
            message = Task(prompt, self.ollama_vision_agent, medias=[image_path], structured_output=Facts).solve()
            self._validate_facts_response(message.structured_output)
            # Additional validation for media-specific requirements
            for fact in message.structured_output.countryFacts:
                # Check if the fact mentions food items from the image
                fact_text = f"{fact.name} {fact.fact}".lower()
                self.assertTrue(
                    any(keyword in fact_text for keyword in ['burger', 'hamburger', 'fries', 'french fries', 'food', 'meal', 'fast food']),
                    f"Fact should mention food items from the image, got: {fact_text}"
                )
        
        # Test OpenAI agent
        if self.run_openai:
            message = Task(prompt, self.openai_agent, medias=[image_path], structured_output=Facts).solve()
            self._validate_facts_response(message.structured_output)
            for fact in message.structured_output.countryFacts:
                fact_text = f"{fact.name} {fact.fact}".lower()
                self.assertTrue(
                    any(keyword in fact_text for keyword in ['burger', 'hamburger', 'fries', 'french fries', 'food', 'meal', 'fast food']),
                    f"Fact should mention food items from the image, got: {fact_text}"
                )
        
        # Test VLLM agent
        #if self.run_vllm:
        #    message = Task(prompt, self.vllm_agent, medias=[image_path], structured_output=Facts).solve()
        #    self._validate_facts_response(message.structured_output)
        #    for fact in message.structured_output.countryFacts:
        #        fact_text = f"{fact.name} {fact.fact}".lower()
        #        self.assertTrue(
        #            any(keyword in fact_text for keyword in ['burger', 'hamburger', 'fries', 'french fries', 'food', 'meal', 'fast food']),
        #            f"Fact should mention food items from the image, got: {fact_text}"
        #        )

    def _validate_facts_response(self, facts):
        """Validate the facts response structure."""
        self.assertIsInstance(facts, Facts)
        self.assertIsInstance(facts.countryFacts, list)
        self.assertEqual(len(facts.countryFacts), 3)
        for fact in facts.countryFacts:
            self.assertIsInstance(fact, CountryFact)
            self.assertIsInstance(fact.name, str)
            self.assertIsInstance(fact.fact, str)
            self.assertGreater(len(fact.name), 0)
            self.assertGreater(len(fact.fact), 0)

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