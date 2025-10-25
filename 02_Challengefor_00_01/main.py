from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

@tool
def analyze_number(element):
    '''This tool analizes the number and tells you the type of number to give back'''
    if isinstance(element, int):
        print("Yep, this is a number")
        return "Yep, this is a number for sure"
    else:
        print("this is not even a number fool!")
    return "This is not a number fool"

agent = create_agent(
    model="anthropic:claude-haiku-4-5",
    tools=[analyze_number],
    system_prompt="You are a mathematical assistant that helps to analize number using the tool 'analyze_number'. You only use the tool to analize. If you dont use the tool, you need to say 'I did not use the Analizys tool'"
)

number_chosen = input("Choose the element to analize:\n")

message = HumanMessage(content=f"Hi, analize {number_chosen}")

for chunk in agent.stream({"messages": [message]}, stream_mode=["messages"]):
    print("successfull message")
