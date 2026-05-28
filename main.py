# import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
# from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
# from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
load_dotenv()

# tavily = TavilyClient()

# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over internet
#     Args:
#         query (str): The search query
#     Returns:
#         str: The search results 
#     """

#     print(f"searching for {query}")
#     response = tavily.search(query = query)
#     return response

llm = ChatOpenAI()
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="Seardch job postings for an AI engineer using langchain in Noida, India or Remote on Linkedin and list their titles and links")})
    print(result)
if __name__ == "__main__":
    main()