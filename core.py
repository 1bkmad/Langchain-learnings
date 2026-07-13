import os
import sys
from typing import Any, Dict
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import ToolMessage
# from langchain_pinecone import PineconeVectorStore
# from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Validate GROQ_API_KEY before initializing anything
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("❌ ERROR: GROQ_API_KEY environment variable is not set.")
    print("Please set it before running the app:")
    print("  export GROQ_API_KEY='your-api-key'")
    print("Get your free API key at: https://console.groq.com/keys")
    sys.exit(1)

#Initialize embeddings (free, runs locally)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#initialize vector store (free, runs locally)
vectorstore = Chroma(
    collection_name="langchain-doc-index-v2",
    persist_directory="chroma_db",
    embedding_function=embeddings,
)

#initialize chat model (uses Groq API with provided key)
model = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=groq_api_key,
    temperature=0.2,
)
"""
if the tool is content, it will return one value and if it's content and artifact, it will return 2 values
1. content, included in LLM prompt but artifact is excluded
2. Typically strings are in content but for artifcat, we can evne have python objects like Dict etc
3. Content is used for reasoning context, minimal info whereas artifact is used for metadata. Raw Data, UI instructions etc
Both are part of same tool message just different purpose.
That's why usually content is used for model reasoning and providede to model/llm whereas artifact is mostly used for application logic only like rendering, DB storage, API calls etc
"""

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """ Retrieve relevent documentation to help answer the user query"""
    # Use MMR (Maximum Marginal Relevance) for better diversity in results
    # This avoids returning duplicate/redundant pages like multiple overview pages
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance instead of similarity
        search_kwargs={
            "k": 6,  # Get more candidates
            "fetch_k": 12,  # Fetch more to filter for diversity
            "lambda_mult": 0.5  # Balance between relevance and diversity
        }
    )
    retrieved_docs = retriever.invoke(query)
    
    # Filter out purely overview/generic pages
    filtered_docs = []
    for doc in retrieved_docs:
        source = doc.metadata.get('source', 'Unknown')
        # Skip pure overview pages to get more specific content
        if not any(x in source for x in ['/overview', '/use-these-docs', 'python.langchain.com/']):
            filtered_docs.append(doc)
        elif len(filtered_docs) < 3:  # But keep some if we need results
            filtered_docs.append(doc)
    
    # Use filtered docs, or fall back to all if we don't have enough
    docs_to_use = filtered_docs if len(filtered_docs) >= 3 else retrieved_docs[:4]
    
    #serialize documents for the model
    serialized = "\n\n".join(
            (f"Source: {doc.metadata.get('source','Unknown')}\n\nContent: {doc.page_content}") 
            for doc in docs_to_use
    )
    return serialized, docs_to_use #content, artifcat

def run_llm(query:str)->Dict[str,Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.

    Args:
        query: The user's question
    Returns:
        Dictionary containing:
            -answer: The generated answer
            -context: List of retrieved  documents
    """

    system_prompt=(
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "IMPORTANT: You MUST use the retrieve_context tool to find relevant documentation for EVERY question. "
        "Do not answer from your training data. "
        "Always call the retrieve_context tool first before generating your answer. "
        "Base your answer ONLY on the retrieved documentation. "
        "If the user asks for a definition or concise explanation, give a short direct answer first and then a one-sentence explanation. "
        "For agent questions, prefer the specific agents documentation page over generic overview pages. "
        "Always cite the sources by including 'Source: [URL]' in your response. "
        "If the retrieved documentation does not contain the answer, explicitly state that it is not found in the documentation."
    )

    agent = create_agent(model,tools = [retrieve_context], system_prompt=system_prompt)
    messages = [{"role": "user", "content":query}]
    response = agent.invoke({"messages":messages})
    answer = response["messages"][-1].content  # as there will be a lot of tool call, and messages so we take the latest
    context_docs = []
    for message in response["messages"]:
        #Check if this is a toolmessage with artifcat
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            #Artifcat should contain list of docs objects
            if isinstance(message.artifact,list):
                context_docs.extend(message.artifact)
    return {
        "answer": answer,
        "context": context_docs
    }


if __name__ == '__main__':
    result = run_llm(query="what are deep agents?")
    print(result)