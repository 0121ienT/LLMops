import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.llm.ollama import ollama_model_complete, ollama_embedding
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger, EmbeddingFunc

setup_logger("lightrag", level="INFO")

WORKING_DIR = "./lightrag/rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag():
    gpt_4o_mini_complete, openai_embed
    rag = LightRAG(
        working_dir=WORKING_DIR,
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


async def main():
    try:
        rag = await initialize_rag()
        rag.insert("I want to be an AI engineer")

        mode = "hybrid"
        print(
            await rag.query(
                "What are the top themes in this story?", param=QueryParam(mode=mode)
            )
        )

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
