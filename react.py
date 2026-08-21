from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv


load_dotenv()

@tool
def triple(num:float) -> float:
    """
    Take a number and return it's triple
    parameter num: a number to triple 
    returns : triple of the input number
    """
    return 3*float(num)

tools= [TavilySearch(max_results = 1), triple]

llm = ChatOllama(model = "qwen3:1.7b", temperature = 0).bind_tools(tools)

