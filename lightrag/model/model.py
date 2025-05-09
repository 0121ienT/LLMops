from base import BaseModel
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.llm.ollama import ollama_model_complete, ollama_embedding
from lightrag.utils import setup_logger

setup_logger("lightrag", level="INFO")


class OpenAIModel(BaseModel):
    gpt_4o_mini_complete
    openai_embed
    ollama_embedding
    ollama_model_complete
