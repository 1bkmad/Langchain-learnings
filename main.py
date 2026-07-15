import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
load_dotenv()


def main():
    print("Hi!")
    information = """
    Chess is a board game for two players, played on a square board consisting of 64 squares arranged in an 8×8 grid. The players, referred to as "White" and "Black", each control sixteen pieces: one king, one queen, two rooks, two bishops, two knights, and eight pawns, with each piece type having a different pattern of movement. An enemy piece may be captured (removed from the board) by moving one's own piece onto the square it occupies. The object of the game is to "checkmate" (threaten with inescapable capture) the enemy king. There are also several ways a game can end in a draw.

    The recorded history of chess dates back to the emergence of chaturanga in 7th-century India. Chaturanga is also thought to be an ancestor of similar games like janggi, xiangqi, and shogi. After its introduction to Persia, it spread to the Arab world and then to Europe. The modern rules of chess emerged in Europe at the end of the 15th century, becoming standardized and gaining universal acceptance by the end of the 19th century. Today, chess is one of the world's most popular games, with millions of players worldwide.
    """

    summary_template = """
    GIven the following information {information} about a game, I want you to create:
    1. A concise summary of the game in one sentence.
    2. A list 2 intresting things about it.
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
        )
    #llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    llm = ChatOllama(model="gemma3:270m", temperature=0)
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response)
    print(response.content)
if __name__ == "__main__":
    main()
