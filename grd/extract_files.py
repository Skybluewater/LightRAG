import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.siliconcloud import siliconcloud_embedding
from lightrag.utils import EmbeddingFunc
import numpy as np
import glob

WORKING_DIR = "./20250317"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

SILICONFLOW_API_KEY = os.getenv("SILICON_API_KEY")

from openai import (
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "Qwen/Qwen2.5-7B-Instruct",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=SILICONFLOW_API_KEY,
        base_url="https://api.siliconflow.cn/v1/",
        **kwargs,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await siliconcloud_embedding(
        texts,
        model="BAAI/bge-m3",
        api_key=SILICONFLOW_API_KEY,
        max_token_size=512,
    )


rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=1024, max_token_size=8192, func=embedding_func
    ),
    graph_storage="Neo4JStorage",
    # log_level="DEBUG",
)

import fitz  # PyMuPDF

def extract_text_from_pdf(file_path):
    # Open the PDF file
    pdf_document = fitz.open(file_path)
    
    # Initialize an empty string to store the extracted text
    text = ""
    
    # Iterate over each page in the PDF
    for page_num in range(len(pdf_document)):
        # Get the page
        page = pdf_document.load_page(page_num)
        
        # Extract text from the page
        text += page.get_text()
    
    return text

# Example usage
# file_path = 'TEXT.pdf'
# text_content = extract_text_from_pdf(file_path)

# rag.insert(text_content)

# Function to read all PDF files in a folder and extract text
def extract_texts_from_pdfs_in_folder(folder_path):
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        print(f"Extracting text from {pdf_file}")
        rag.insert(text)

# Example usage
folder_path = './pdfs'
extract_texts_from_pdfs_in_folder(folder_path)

# with open("./book.txt", encoding="utf-8") as f:
#     rag.insert(f.read())

# Perform naive search
print(
    rag.query("数字表演有什么关键方法?", param=QueryParam(mode="naive"))
)

# Perform local search
print(
    rag.query("数字表演有什么关键方法?", param=QueryParam(mode="local"))
)

# Perform global search
print(
    rag.query("数字表演有什么关键方法?", param=QueryParam(mode="global"))
)

# Perform hybrid search
print(
    rag.query("数字表演有什么关键方法?", param=QueryParam(mode="hybrid"))
)
