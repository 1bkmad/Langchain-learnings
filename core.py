import os
from typing import Any, Dict
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import ToolMessage
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

#Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#initialize vector store
vectorstore = PineconeVectorStore(
    index_name="langchain-doc-index",
    embedding=embeddings,
)

#initialize chat model
model = init_chat_model("gpt-4o-mini", model_provider="openai")

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
    #retrieving top4 most similar documents
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)  #for the vectorstore as retriever, the invoke is gonna perfomr a similarity search
    
    #serialize documents for the model
    serialized = "\n\n".join(
            (f"Source: {doc.metadata.get('source','Unknown')}\n\nContent: {doc.page_content}") 
            for doc in retrieved_docs
    )
    return serialized, retrieved_docs #content, artifcat

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
        "You are a helpful AI assistant that answers questions about Langchain documentation."
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent = create_agent(model,tools = [retrieve_context], system_prompt=system_prompt)
    messages = [{"role": "user", "content":query}]
    response = agent.invoke({"messages":messages})
    answer = response["messages"][-1].content  # as there will be a lot of tool call, and messages so we take the latest
    context_docs = []
    print("context aagya")
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