from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
import time
from langgraph.config import get_stream_writer
from langchain_core.tools import tool

load_dotenv()



agent = create_agent(
    model="anthropic:claude-haiku-4-5",
    tools=[],
    system_prompt="You are a full-stack comedian, you always check if your joke is funny using your tool",
)

message = HumanMessage(content="Hi, tell me a joke about recruiters")

selection = input("Want to have:\n01 Values\n02 Messages\n")

if selection == "01":
    selection = "values"
else:
    selection = "messages"

if selection == "values":
    for chunk in agent.stream({"messages": [message]}, stream_mode=["values"]):
        mode, data = chunk  #unpack the tuple
        if data.get("messages"):   #access the "messages" key if it exists. 
            print(data["messages"][-1].content) #print the .content variable inside the "messages" dictionary key. 
            time.sleep(.5)  #Better seein of the stream


elif selection == "messages":
    for chunk in agent.stream({"messages": [message]}, stream_mode=["messages"]):
        mode, data  = chunk  # Unpack the tuple
        msg, metadata = message #unpack the second tuple
        print(msg.content, end="", flush=True)
        time.sleep(.5)  #Better seein of the stream

