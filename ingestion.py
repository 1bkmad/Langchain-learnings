import os
import dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


dotenv.load_dotenv()

if __name__ == "__main__":
    print("Ingesting data...")
    print(os.environ.get("PINECONE_API_KEY"))
    loader = TextLoader("mediumblog1.txt", autodetect_encoding = True)    
    # we can use loader to load any type of file, like email, whatsapp, micrsoft onedrive, youtube transcripts etc. 
    #Just change the TextLoader
    document = loader.load()
    print("splitting...")
    print(document[0].page_content)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")
    embeddings = OpenAIEmbeddings(openai_api_key = os.environ.get("OPENAI_API_KEY"),model = "text-embedding-3-small")

    print("ingesting...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name = os.environ['INDEX_NAME'])
    print("finish")
