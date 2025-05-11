import os
from dotenv import load_dotenv
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.llm.ollama import ollama_model_complete, ollama_embedding
from lightrag.utils import setup_logger, EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

load_dotenv()

setup_logger("lightrag", level="INFO")
if not os.path.exists(os.getenv("WORKING_DIR")):
    os.mkdir(os.getenv("WORKING_DIR"))


async def openai_initialize_rag():
    rag = LightRAG(
        working_dir=os.getenv("WORKING_DIR"),
        llm_model_func=gpt_4o_mini_complete,
        graph_storage="Neo4JStorage",
        embedding_func=EmbeddingFunc(
            embedding_dim=1536, max_token_size=8192, func=openai_embed
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def ollama_initialize_rag():
    rag = LightRAG(
        working_dir=os.getenv("WORKING_DIR"),
        llm_model_func=ollama_model_complete,
        llm_model_name="llama3.2",
        llm_model_kwargs={"options": {"num_ctx": 32768}},
        graph_storage="Neo4JStorage",
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embedding(texts, embed_model="bge-m3"),
        ),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag
