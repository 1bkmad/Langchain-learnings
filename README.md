# LangChain Streamlit RAG Demo

This repository contains a Streamlit demo application built with LangChain and Retrieval-Augmented Generation (RAG). The app answers user questions using content indexed from the official LangChain documentation.

## Project structure

- `ingestion.py` - loads source documents, processes them into chunks, generates embeddings, and stores vectors for retrieval.
- `core.py` - defines the retrieval and answer generation workflow, including vector search and prompt composition.
- `main.py` - provides the Streamlit application interface and ties the core RAG pipeline to the user experience.

## Key concepts covered

- Document ingestion and preprocessing
- Vector embeddings and similarity search
- Retrieval-Augmented Generation (RAG)
- Streamlit app deployment
- LangChain pipeline design

## Prerequisites

- Python 3.10+ (or compatible version)
- `pip` installed
- Suitable OpenAI or another LLM API key configured in environment variables if the implementation uses an LLM provider

## Installation

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate    # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set the required environment variables, for example:

```bash
export OPENAI_API_KEY="your_api_key"
```

## Running the app

```bash
streamlit run main.py
```

Then open the local Streamlit URL shown in the terminal.

## How it works

1. `ingestion.py` reads the source documentation or content files.
2. Text is split into smaller passages or document chunks.
3. Each chunk is converted into a vector embedding.
4. Embeddings are stored in a vector store for fast lookup.
5. `core.py` accepts a user query and performs a similarity search against the vector store.
6. The most relevant chunks are retrieved and combined into a context prompt.
7. An LLM generates an answer based on the retrieved context plus the original question.
8. `main.py` displays the question input, query results, and generated response in Streamlit.

## Component details

- `ingestion.py`
  - Responsible for data collection and preprocessing.
  - Creates document chunks that are easier for embeddings to represent.
  - Persists the vector index for reuse.

- `core.py`
  - Manages queries and retrieval.
  - Builds prompt context from relevant documents.
  - Calls the LLM with the retrieved context to generate final answers.

- `main.py`
  - Streamlit front-end.
  - Allows users to enter questions, view results, and explore the RAG flow.

## Tips for interview questions

- Explain the RAG architecture: retrieval step + generation step.
- Mention why vector embeddings are used instead of keyword search.
- Describe how document chunks improve retrieval relevance.
- Show that the app separates ingestion, retrieval, and UI concerns.
- Note that Streamlit is used for rapid prototyping and demonstrating the user workflow.

## Possible improvements

- Add support for more data sources and file types.
- Implement caching for repeated queries.
- Add user session history and analytics.
- Use a production-grade vector database like Pinecone, Milvus, or FAISS.
- Add tests for ingestion, retrieval, and response generation.

## License

This is an example demo repository. Adjust the license and attribution as needed for your project.

