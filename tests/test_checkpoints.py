import unittest
import os
import json
from tests.test_base import BaseAgentTest
from yacana import Task, Message, MessageRole, Agent
from pydantic import BaseModel
from yacana.generic_agent import GenericAgent

class CountryFact(BaseModel):
    """A fact about a country."""
    name: str
    fact: str

class Facts(BaseModel):
    """Collection of country facts."""
    countryFacts: list[CountryFact]

class TestCheckpoints(BaseAgentTest):
    """Test checkpoint functionality of the History class."""

    def setUp(self):
        """Clean up agent histories before each test."""
        super().setUp()
        if self.run_ollama:
            self.ollama_agent.history.clean()
        if self.run_openai:
            self.openai_agent.history.clean()
        if self.run_vllm:
            self.vllm_agent.history.clean()

    def test_simple_checkpoint(self):
        """Test basic checkpoint creation and loading."""
        def test_agent_checkpoint(agent):
            # Create initial state
            message1 = Task("Tell me 1 fact about Canada.", agent).solve()
            initial_content = message1.content
            
            # Create checkpoint
            checkpoint_id = agent.history.create_check_point()
            
            # Add more messages
            message2 = Task("Tell me 1 fact about France.", agent).solve()
            
            # Verify history has all messages (system prompt + 2 interactions = 5 messages)
            self.assertEqual(len(agent.history.slots), 5)
            
            # Load checkpoint
            agent.history.load_check_point(checkpoint_id)
            
            # Verify history is restored to initial state (system prompt + 1 interaction = 3 messages)
            self.assertEqual(len(agent.history.slots), 3)
            self.assertEqual(agent.history.get_last_message().content, initial_content)
            
            # Verify we can continue from checkpoint (system prompt + 2 interactions = 5 messages)
            message3 = Task("Tell me 1 fact about England.", agent).solve()
            self.assertEqual(len(agent.history.slots), 5)
            self.assertNotEqual(message3.content, initial_content)
        
        # Test with all available agents
        if self.run_ollama:
            test_agent_checkpoint(self.ollama_agent)
        if self.run_openai:
            test_agent_checkpoint(self.openai_agent)
        if self.run_vllm:
            test_agent_checkpoint(self.vllm_agent)

    def test_checkpoint_with_structured_output(self):
        """Test checkpoints with structured output."""
        def test_agent_structured_checkpoint(agent):
            # Create initial state with structured output
            message1 = Task("Tell me 1 fact about Canada.", agent, structured_output=Facts).solve()
            initial_fact = message1.structured_output.countryFacts[0].fact
            
            # Create checkpoint
            checkpoint_id = agent.history.create_check_point()
            
            # Add more messages with structured output
            message2 = Task("Tell me 1 fact about France.", agent, structured_output=Facts).solve()
            
            # Verify history has all messages (system prompt + 2 interactions = 5 messages)
            self.assertEqual(len(agent.history.slots), 5)
            
            # Load checkpoint
            agent.history.load_check_point(checkpoint_id)
            
            # Verify history is restored to initial state (system prompt + 1 interaction = 3 messages)
            self.assertEqual(len(agent.history.slots), 3)
            self.assertEqual(agent.history.get_last_message().structured_output.countryFacts[0].fact, initial_fact)
            
            # Verify we can continue from checkpoint with structured output (system prompt + 2 interactions = 5 messages)
            message3 = Task("Tell me 1 fact about England.", agent, structured_output=Facts).solve()
            self.assertEqual(len(agent.history.slots), 5)
            self.assertNotEqual(message3.structured_output.countryFacts[0].fact, initial_fact)
        
        # Test with all available agents
        if self.run_ollama:
            test_agent_structured_checkpoint(self.ollama_agent)
        if self.run_openai:
            test_agent_structured_checkpoint(self.openai_agent)
        if self.run_vllm:
            test_agent_structured_checkpoint(self.vllm_agent)

    def test_checkpoint_with_state_persistence(self):
        """Test checkpoints with agent state persistence."""
        def test_agent_state_checkpoint(agent):
            # Create initial state
            message1 = Task("Tell me 1 fact about Canada.", agent).solve()
            initial_content = message1.content
            
            # Create checkpoint
            checkpoint_id = agent.history.create_check_point()
            
            # Add more messages
            message2 = Task("Tell me 1 fact about France.", agent).solve()
            
            # Save agent state
            state_file = os.path.join(self.temp_dir, f"{agent.name}_state.json")
            agent.export_to_file(state_file)
            
            # Create new agent from state
            new_agent = Agent.import_from_file(state_file)
            # Verify new agent has all messages (system prompt + 2 interactions = 5 messages)
            self.assertEqual(len(new_agent.history.slots), 5)
            
            # Load checkpoint in new agent
            new_agent.history.load_check_point(checkpoint_id)
            
            # Verify history is restored to initial state (system prompt + 1 interaction = 3 messages)
            self.assertEqual(len(new_agent.history.slots), 3)
            self.assertEqual(new_agent.history.get_last_message().content, initial_content)
        
        # Test with all available agents
        if self.run_ollama:
            test_agent_state_checkpoint(self.ollama_agent)
        if self.run_openai:
            test_agent_state_checkpoint(self.openai_agent)
        if self.run_vllm:
            test_agent_state_checkpoint(self.vllm_agent)

    def test_checkpoint_with_structured_output_and_state_persistence(self):
        """Test checkpoints with both structured output and state persistence."""
        def test_agent_full_checkpoint(agent: GenericAgent):
            # Create initial state with structured output
            message1 = Task("Tell me 1 fact about Canada.", agent, structured_output=Facts).solve()
            initial_fact = message1.structured_output.countryFacts[0].fact
            
            # Create checkpoint
            checkpoint_id = agent.history.create_check_point()
            
            # Add more messages with structured output
            message2 = Task("Tell me 1 fact about France.", agent, structured_output=Facts).solve()
            
            # Save agent state
            state_file = os.path.join(self.temp_dir, f"{agent.name}_full_state.json")

            agent.export_to_file(state_file)
            
            # Create new agent from state
            new_agent = Agent.import_from_file(state_file)
            
            # Verify new agent has all messages (system prompt + 2 interactions = 5 messages)
            self.assertEqual(len(new_agent.history.slots), 5)
            
            # Load checkpoint in new agent
            new_agent.history.load_check_point(checkpoint_id)
            
            # Verify history is restored to initial state (system prompt + 1 interaction = 3 messages)
            self.assertEqual(len(new_agent.history.slots), 3)
            self.assertEqual(new_agent.history.get_last_message().structured_output.countryFacts[0].fact, initial_fact)
            
            # Verify we can continue from checkpoint with structured output (system prompt + 2 interactions = 5 messages)
            message3 = Task("Tell me 1 fact about England.", new_agent, structured_output=Facts).solve()
            self.assertEqual(len(new_agent.history.slots), 5)
            self.assertNotEqual(message3.structured_output.countryFacts[0].fact, initial_fact)
        
        # Test with all available agents
        if self.run_ollama:
            test_agent_full_checkpoint(self.ollama_agent)
        if self.run_openai:
            test_agent_full_checkpoint(self.openai_agent)
        if self.run_vllm:
            test_agent_full_checkpoint(self.vllm_agent)

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