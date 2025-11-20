import os
import uuid
from langfuse.openai import openai


class LangfuseConnector:
    def __init__(self, api_key: str):
        self.langfuse_session_id = str(uuid.uuid4())
        self.client = OpenAI(
            api_key=self.api_token,
            base_url=self.endpoint
        )
        os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-ee664b4c-f18f-4611-98d5-694744b45ca3"
        os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-0fea30ad-1c52-40c2-9829-c096c8103567"
        os.environ["LANGFUSE_BASE_URL"] = "http://localhost:3000"
        self.langfuse_client = get_client()
