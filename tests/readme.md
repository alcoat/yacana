From repo root use the following cmd lines to start tests.

Execute whole file:
```
TEST_OLLAMA=true TEST_OPENAI=false TEST_VLLM=false TEST_LMSTUDIO=false python3 -m unittest tests/test_state_persistence.py -v
```

Execute specific test inside file:
```
TEST_OLLAMA=true TEST_OPENAI=false TEST_VLLM=false TEST_LMSTUDIO=false python3 -m unittest tests.test_state_persistence.TestStatePersistence.test_state_with_structured_output -v
```

To test with OpenAI models set the appropriate OpenAI key:
```
export OPENAI_API_TOKEN=sk-proj-XXXXXXXXXXXX
```

To test token counting set the appropriate Hugging face token:
```
export HF_TOKEN=HF_XXXX
```