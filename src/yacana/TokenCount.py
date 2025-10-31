import logging
from typing import List, TypedDict
import regex

from .exceptions import SpecializedTokenCountingError


class HuggingFaceDetails:
    def __init__(self, repo_name: str, token: str | None = None):
        self.repo_name = repo_name
        self.token = token


class HFMessage(TypedDict):
    role: str
    content: str


def count_tokens_using_huggingface(hugging_faces_messages: List[HFMessage], hugging_face_repo_name: str, hugging_face_token: str | None = None) -> int:
    try:
        from transformers import AutoTokenizer
        if hugging_face_token:
            from huggingface_hub import login
            logging.debug("Logging into Hugging Face Hub to access private model for token counting. This may take some time...")
            login(hugging_face_token)
        logging.debug("Loading tokenizer from Hugging Face Hub for model: " + hugging_face_repo_name)
        tokenizer = AutoTokenizer.from_pretrained(hugging_face_repo_name)
        tokens = tokenizer.apply_chat_template(hugging_faces_messages, tokenize=True)
        nb_tokens = len(tokens)
        return nb_tokens
    except Exception as e:
        logging.warning(f"Could not load tokenizer for model {hugging_face_repo_name}. Falling back to approximative token count. Error: {e}")
        raise SpecializedTokenCountingError(str(e))


def count_tokens_using_tiktoken(llm_model_name: str, message: HFMessage) -> int:
    import tiktoken
    try:
        logging.debug("Loading tiktoken encoding for model: " + llm_model_name)
        enc = tiktoken.encoding_for_model(llm_model_name)  # Getting correct encoding if it's an OpenAI model ONLY else ValueError
        return len(enc.encode(message["role"])) + len(enc.encode(message["content"]))
    except KeyError:
        logging.debug(f"Could not find encoding for model {llm_model_name}. This is normal if this model is not from OpenAI. You should set the @hugging_face_repo_nama and token in the Agent class so Yacana may use the correct tokeniser for this LLM. Falling back to approximative token count instead.")
        raise SpecializedTokenCountingError(f"Could not find encoding for model {llm_model_name}.")


def count_tokens_using_regex(message: HFMessage) -> int:
    token_match_regex = regex.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+", regex.MULTILINE)
    logging.debug("Using approximative token count method.")
    return len(token_match_regex.findall(message["role"]) + token_match_regex.findall(message["content"]))
