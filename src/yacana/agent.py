import logging

from .generic_agent import GenericAgent


class Agent(GenericAgent):
    """
    Deprecated ! Do not use !
    Use OllamaAgent() or OpenAiAgent() instead depending on the backend you want to use.
    """

    def __init__(self, *args, **kwargs) -> None:
        deprecated_message = """
                Deprecation error: Use specialized Agents class instead. Sorry this had to be removed... But now that Yacana manages more than Ollama as backend, this class didn't make much sense.
                * If you are using Ollama as backend then the new `Agent` class name you should use is `OllamaAgent()`.
                * For OpenAi compatible backends (chatGPT, VLLM, etc), use `OpenAiAgent()`.
                """
        logging.fatal(deprecated_message)
        raise DeprecationWarning(deprecated_message)

    def _interact(self, task, tools, json_output, structured_output, medias, streaming_callback, task_runtime_config, tags):
        pass
