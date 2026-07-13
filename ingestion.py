import asyncio
import os
import re
import ssl
from typing import Any, Dict, List

import certifi
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_tavily import TavilyExtract, TavilyMap

from logger import Colors, log_error, log_header, log_info, log_success, log_warning

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=".hf_cache",
)

vectorstore = Chroma(
    collection_name="langchain-doc-index-v2",
    persist_directory="chroma_db",
    embedding_function=embeddings,
)

# Keep the original crawl-based flow, but with better ranking later.
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)


def chunk_urls(urls: List[str], chunk_size: int = 20) -> List[List[str]]:
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunks.append(urls[i : i + chunk_size])
    return chunks


async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    try:
        log_info(f"🔄 TavilyExtract batch {batch_num} with {len(urls)} URLs", Colors.BLUE)
        docs = await tavily_extract.ainvoke(input={"urls": urls, "extract_depth": "advanced"})
        return docs
    except Exception as exc:
        log_error(f"Tavily extract failed for batch {batch_num}: {exc}")
        return {"results": []}


async def async_extract(url_batches: List[List[str]]) -> List[Document]:
    tasks = [extract_batch(batch, i + 1) for i, batch in enumerate(url_batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_pages: List[Document] = []
    for result in results:
        if isinstance(result, Exception):
            log_error(f"Batch failed: {result}")
            continue
        for extracted_page in result.get("results", []):
            url = extracted_page.get("url", "")
            content = extracted_page.get("raw_content", "")
            if content and len(content) > 300:
                all_pages.append(
                    Document(
                        page_content=content,
                        metadata={"source": url, "title": url.split("/")[-1]},
                    )
                )
    return all_pages


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_chunks(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    return splitter.split_documents(documents)


async def main() -> None:
    log_header("DOCUMENTATION INGESTION PIPELINE")
    log_info("Crawling LangChain docs with Tavily", Colors.PURPLE)

    site_map = tavily_map.invoke("https://python.langchain.com/")
    urls = list(site_map.get("results", []))
    log_success(f"Tavily mapped {len(urls)} URLs")

    url_batches = chunk_urls(urls, chunk_size=20)
    all_docs = await async_extract(url_batches)
    log_success(f"Extracted {len(all_docs)} documents")

    # Keep the agents and tools pages prominent by boosting their content in the index.
    preferred_sources = [
        "https://docs.langchain.com/oss/python/langchain/agents",
        "https://docs.langchain.com/oss/python/langchain/tools",
        "https://docs.langchain.com/oss/python/langchain/messages",
        "https://docs.langchain.com/oss/python/langchain/models",
        "https://docs.langchain.com/oss/python/langchain/retrieval",
    ]
    preferred_docs = []
    for doc in all_docs:
        source = doc.metadata.get("source", "")
        if any(source.startswith(pref) for pref in preferred_sources):
            preferred_docs.append(doc)

    # Include the preferred docs first, then the rest.
    ordered_docs = preferred_docs + [doc for doc in all_docs if doc not in preferred_docs]

    chunks = build_chunks(ordered_docs)
    log_success(f"Created {len(chunks)} chunks for indexing")

    vectorstore.add_documents(chunks)
    log_success("Indexed documents into Chroma")


if __name__ == "__main__":
    asyncio.run(main())