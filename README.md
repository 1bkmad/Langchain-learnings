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
- **GROQ_API_KEY** - Get your free API key at https://console.groq.com/keys
  - Free tier: Limited requests per day
  - Paid: ~$0.05–$0.20 per 1M input tokens
  - **Important**: Only your API key costs money; HuggingFace embeddings and ChromaDB are completely free

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

3. Set the required environment variables:

```bash
export GROQ_API_KEY="your_groq_api_key"
```

To get a free Groq API key:
1. Go to https://console.groq.com/keys
2. Create an account or sign in
3. Generate an API key
4. Set it in your environment

**Cost breakdown:**
- ✅ HuggingFace embeddings: FREE (runs locally)
- ✅ ChromaDB vector store: FREE (runs locally in `chroma_db/`)
- ⚠️ Groq API: FREE tier available, then pay-as-you-go

All embeddings and vector storage run locally on your machine. Only the LLM API calls (to Groq) incur costs beyond the free tier.

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
  - Uses Tavily API for web crawling (optional; requires TAVILY_API_KEY)

- `core.py`
  - Manages queries and retrieval using MMR (Maximum Marginal Relevance) for better diversity.
  - Builds prompt context from relevant documents.
  - Calls the LLM with the retrieved context to generate final answers.
  - Validates GROQ_API_KEY before starting.

- `main.py`
  - Streamlit front-end.
  - Allows users to enter questions, view results, and explore the RAG flow.
  - Displays retrieved sources with expandable source details.

## Costs & Deployment

**Local Components (FREE):**
- HuggingFace embeddings: Runs on your machine, no API calls
- ChromaDB: Runs locally, stored in `chroma_db/` directory

**Cloud Components (Groq API):**
- Only the LLM completions cost money
- User provides their own GROQ_API_KEY
- Groq free tier: ~5,000 free requests/month
- Paid tier: ~$0.05–$0.20 per 1M input tokens




## License

This is an example demo repository. Adjust the license and attribution as needed for your project.

