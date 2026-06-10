from dotenv import load_dotenv

load_dotenv()
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
MAX_ITERATIONS= 10
MODEL = "qwen3:1.7b"

#-----------------------------
#Tool definition

@tool
def get_product_price(product:str) -> float:
    """ Look up the price of a product in the catalog"""
    print(f"__>> executing get_product_price(product='{product}')")
    prices ={"laptop":1999.99, "headphones":150, "keyboard":90 }
    return prices.get(product,0)  # if there is any missmatch, we are returning 0 simply

@tool
def apply_discount(price: float, discount_tier : str) -> float:
    """
    Apply a discount tier to a price and return the final price.
    Available tiers : bronze, silver and gold.
    """
    print(f"__>> Ececuting apply_discount(price = {price},discount_tier = {discount_tier})")
    discount_percentages = {"bronze": 5, "silver": 10, "gold" : 15}
    discount = (discount_percentages.get(discount_tier, "bronze")*price)/100
    return round(price - discount, 2)

# -- AGENT LOOP
@traceable(name ="LangChain Agent Loop")
def run_agent(question:str) :
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name : t for t in tools}
    llm = init_chat_model(f"ollama:{MODEL}", temperature = 0)
    llm_with_tools = llm.bind_tools(tools)
    # print(tools_dict)

    print(f"question : {question}")
    print("="*60)
    messages = [
        SystemMessage(
            content = (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool. \n\n"
                "STRICT RULES - you must follow these exactly \n"
                "1. Never guess or assume any product price. "
                "YOU MUST call the get_product_price first to get the real price "
                "2. Only call the apply_discount AFTER you have received "
                " a price from get_product_price. Pass the exact price "
                "returned by get_product_price - do NOT pass a made-up number.\n"
                "3. NEVER calculate the discounts yourself using maths."
                "Always use the apply_discount tool. \n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use - do NOT assume one."
                )
        ),
        HumanMessage(content = question),
    ]
    for iterations in range(1,MAX_ITERATIONS+1):
        print(f"\n---ITERATION {iterations}---")
        ai_message = llm_with_tools.invoke(messages)
        # print(f"AI Message : {ai_message}")
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print(f"final answer : {ai_message.content}")
            return ai_message.content
        # Process only the first tool call - forcing one tool per iterations
        tool_call = tool_calls[0]
        print(tool_calls)
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args",{})
        tool_call_id = tool_call.get("id")
        print(f" [Tool Selected ] {tool_name} with args : {tool_args}")
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        observation = tool_to_use.invoke(tool_args)
        print(f"[Tool Result] = {observation}")
        messages.append( ai_message)
        messages.append(
                ToolMessage(content=str(observation), tool_call_id=tool_call_id)
                )
        # print(f"message = {messages}")
    print("ERROR: Max iterations reached without a final answer")
    return None

if __name__ == "__main__":
    print("hello langchain agent (.bind_tools)!")
    print()
    result = run_agent("what is the price of a laptop after applying silver discount")
    
