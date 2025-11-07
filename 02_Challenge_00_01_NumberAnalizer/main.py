from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.config import get_stream_writer

load_dotenv()

@tool
def analyze_number(element):
    '''This tool analizes the number and tells you the type of number to give back'''
    writer = get_stream_writer()
    if isinstance(element, int):
        writer("Confirmed this is a number, analizyng:")
        return "Yep, this is a number for sure\n"
    else:
        writer("this is not even a number fool, try again")
    return "This is not a number fool\n"

agent = create_agent(
    model="anthropic:claude-haiku-4-5",
    tools=[analyze_number],
    system_prompt="You are a mathematical assistant that helps to analize number using the tool 'analyze_number'. You only use the tool to analize. If you dont use the tool, you need to say 'I did not use the Analizys tool'"
)

number_chosen = input("Choose the element to analize:\n")

message = HumanMessage(content=f"Hi, analize: {number_chosen}")

for chunk in agent.stream({"messages": [message]}, stream_mode=["messages", "custom"]):
    mode, data = chunk
    
    if mode == "messages":
        msg, metadata = data
        
        # Check if content is a list (streaming chunks)
        if isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    print(item['text'], end="", flush=True)
        # Check if content is a string (final message)
        elif isinstance(msg.content, str) and msg.content:
            print(msg.content, end="", flush=True)
        
    elif mode == "custom":
        print(f"\n[TOOL]: {data}")
