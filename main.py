import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
print("Initializing components...")
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-3.5-turbo")
vectorstore = PineconeVectorStore(
    index_name = os.environ["INDEX_NAME"], embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k":3})
prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context, WITHOUT thinking deep:
    {context}

    Question: {question}

    Provide an answer in less than 400 tokens:

    """
)
def format_docs(docs):
    """Format retrieved documents into single string. """
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query:str):
    """
    Simple retrieval chain without lcel.
    Manually retrieves documents, formats them, and generates a response.
    """
    # step 1: Retrieve relevent document
    retrieved_list = retriever.invoke(query)
    context = format_docs(retrieved_list)
    messages = prompt_template.format_messages(context = context,question = query)
    response = llm.invoke(messages)
    print(response)
    return response.content






if __name__ == "__main__":
    query = "what is pinecone in machine learning?"
    result = retrieval_chain_without_lcel(query)
    print(f"answer = {result}")
