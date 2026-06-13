from dotenv import load_dotenv

load_dotenv()


from langsmith import traceable

MAX_ITERATIONS= 10
MODEL = "qwen3:1.7b"
import ollama
#-----------------------------
#Tool definition

@traceable(run_type = "tool")
def get_product_price(product:str) -> float:
    """ Look up the price of a product in the catalog"""
    print(f"__>> executing get_product_price(product='{product}')")
    prices ={"laptop":1999.99, "headphones":150, "keyboard":90 }
    return prices.get(product,0)  # if there is any missmatch, we are returning 0 simply

@traceable(run_type = "tool")
def apply_discount(price: float, discount_tier : str) -> float:
    """
    Apply a discount tier to a price and return the final price.
    Available tiers : bronze, silver and gold.
    """
    print(f"__>> Ececuting apply_discount(price = {price},discount_tier = {discount_tier})")
    discount_percentages = {"bronze": 5, "silver": 10, "gold" : 15}
    discount = (discount_percentages.get(discount_tier, "bronze")*price)/100
    return round(price - discount, 2)

#in case we are not using the @tool decorator, we need to define the JSON schema for each function, which is exactly what the langchain does for us automatically.
tools_for_llm=[
        {
            "type": "function",
            "function":{
                "name" : "get_product_price",
                "description": "Look up the price of a product in the catalog.",
                "parameters":{
                    "type": "object",
                    "properties":{
                        "product": {
                            "type": "string",
                            "description": " The Product name, e.g. 'laptop', 'headphones', 'keyboard'", 
                        },                       
                    },
                },
                "required": ["product"],
            },
        },
        {
            "type": "function",
            "function": {
                "name": "apply_discount",
                "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "price": {"type": "number", "description": "The original price"},
                        "discount_tier": {
                            "type": "string",
                            "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                        },
                    },
                    "required": ["price", "discount_tier"],
                },
            },
        },

        ]

# -- AGENT LOOP

# Ollama can also auto-generate these schemas if you pass the python functions directly to the llm using the tools argument in the invoke method
# tools_for_llm = [get_product_price, apply_discount]  --- IGNORE --- 
# however this requires the docstrings to follow the google docstring format
#def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """

# helper : traced ollama call
@traceable(run_type = "ollama Chat")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools = tools_for_llm, messages = messages)


@traceable(name ="LangChain Agent Loop")
def run_agent(question:str) :
    tools = [get_product_price, apply_discount]
    tools_dict = {"get_product_price": get_product_price, "apply_discount": apply_discount,}

    print(f"question : {question}")
    print("="*60)
    messages = [
        {
            "role":"system",
            "content" : (
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


        },
        {"role": "user", "content": question},
     ]
    for iterations in range(1,MAX_ITERATIONS+1):
        print(f"\n---ITERATION {iterations}---")
        response = ollama_chat_traced(messages = messages)
        ai_message = response.message
        
        # Process only the first tool call - forcing one tool per iterations
        tool_calls = ai_message.tool_calls
        
        if not tool_calls:
            print(f"final answer : {ai_message.content}")
            return ai_message.content
        tool_call = tool_calls[0]
        # print(tool_call)
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        # tool_call_id = tool_call.get("id")
        print(f" [Tool Selected ] {tool_name} with args : {tool_args}")
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        observation = tool_to_use(**tool_args)
        print(f"[Tool Result] = {observation}")
        messages.append( ai_message)
        messages.append(
                {"role": "tool", "content": str(observation)}
                )
        # print(f"message = {messages}")
    print("ERROR: Max iterations reached without a final answer")
    return None

if __name__ == "__main__":
    print("hello langchain agent (.bind_tools)!")
    print()
    result = run_agent("what is the price of a laptop after applying silver discount")
    
