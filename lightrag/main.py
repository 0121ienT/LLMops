import os
import logging
import textract
import asyncio
from dotenv import load_dotenv
from model import openai_initialize_rag
from lightrag.utils import setup_logger, TokenTracker
from lightrag.base import QueryParam


load_dotenv()

setup_logger("lightrag", level="INFO", log_file_path="./logs/lightrag.log")

logging.basicConfig(level=logging.INFO)


async def main():
    rag = await openai_initialize_rag()
    token_tracker = TokenTracker()
    init_number_files = 0
    try:
        data_folder_path = os.getenv("DATA_DIR")
        logging.info("Extracting Information in Data Folder...")
        current_number_files = len(os.listdir(data_folder_path))
        logging.info(f"Number of files: {init_number_files} {current_number_files}")
        if init_number_files != current_number_files:
            for filename in os.listdir(data_folder_path):
                try:
                    file_path = os.path.join(data_folder_path, filename)
                    text_content = textract.process(file_path)
                    await rag.ainsert(text_content.decode("utf-8"))
                except Exception as e:
                    logging.error(
                        f"Error processing file {filename}: {e}", exc_info=True
                    )
                    raise
            init_number_files = current_number_files
            logging.info("Processed Data Folder Done !!!")
        else:
            logging.info("Database is already up to date !!!")

        conversation_history = [
            {
                "role": "user",
                "content": "What is the main character's attitude towards Christmas?",
            },
            {
                "role": "assistant",
                "content": "At the beginning of the story, Ebenezer Scrooge has a very negative attitude towards Christmas...",
            },
            {"role": "user", "content": "How does his attitude change?"},
        ]

        config_query = QueryParam(
            mode="hybrid", conversation_history=conversation_history, stream=True
        )
        prompt = input("User: ")
        with token_tracker:
            response = await rag.aquery(prompt, param=config_query)
        result = ""
        print("AI: ", end="")
        async for chunk in response:
            result += chunk
            print(chunk, end="", flush=True)
    except Exception as e:
        print(f"An error occurred: {e}")
        logging.error(f"Error occurred: {e}", exc_info=True)
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
