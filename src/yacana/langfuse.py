import os
import uuid
from langfuse.openai import OpenAI
from langfuse import get_client, propagate_attributes


class LangfuseConnector:
    """
    Initializes connection to a remote Langfuse instance.
    When given to an agent every LLM call will be logged to Langfuse.

    Parameters
    ----------
    endpoint: str
        The Langfuse endpoint URL. Ie http://127.0.0.1:3000
    public_key: str
        The Langfuse public_key
    secret_key: str
        The Langfuse secret_key
    metadata: dict, optional
        Your custom metadata to associate with the trace.
    user_id: str, optional
        ID identifying a user in Langfuse to associate the trace with.
    observation_suffix: str
        Suffix to append to observations' names logged to Langfuse. Names start with the agent name + "-{suffix}" if provided.

    Attributes
    ----------
    endpoint: str
        The Langfuse endpoint URL.
    public_key: str
        The Langfuse public_key
    secret_key: str
        The Langfuse secret_key
    metadata: dict
        Your custom metadata to associate with the trace. Defaults to {}.
    user_id: str
        ID identifying a user in Langfuse to associate the trace with.
    observation_suffix: str
        Suffix to append to observations' names logged to Langfuse. Names start with the agent name + "-{suffix}" if provided.
    session_id: str
        Unique identifier for the session. Generated using UUID4. Used to link traces with each others in Langfuse.
    client: Langfuse Client
        The Langfuse client instance.
    """
    def __init__(self, endpoint: str, public_key: str, secret_key: str, metadata: dict | None = None, user_id: str = None, observation_suffix: str = ""):
        self._endpoint: str = endpoint
        self._public_key: str = public_key
        self._secret_key: str = secret_key
        self.metadata: dict | None = metadata if metadata else {}
        self.user_id: str = user_id
        self.observation_suffix: str = observation_suffix if observation_suffix == "" else "-" + observation_suffix
        self.session_id = str(uuid.uuid4())
        self._openai_client = None
        os.environ["LANGFUSE_PUBLIC_KEY"] = self._public_key
        os.environ["LANGFUSE_SECRET_KEY"] = self._secret_key
        os.environ["LANGFUSE_BASE_URL"] = self._endpoint
        self.client = get_client()

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def public_key(self) -> str:
        return self._public_key

    @property
    def secret_key(self) -> str:
        return self._secret_key

    def get_openai_client(self, openai_endpoint, openai_api_token) -> OpenAI:
        """
        Returns the Langfuse OpenAI client.

        Parameters
        ----------
        openai_api_token: str
            The OpenAI API token.
        openai_endpoint: str
            The OpenAI endpoint URL.

        Returns
        -------
        OpenAI
            The Langfuse OpenAI client.
        """
        if not self._openai_client:
            self._openai_client = OpenAI(
                api_key=openai_api_token,
                base_url=openai_endpoint
            )
        return self._openai_client
