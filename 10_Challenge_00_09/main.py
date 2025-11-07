from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
#from langchain.agents.middleware #how to get the dynamic prompting part!?
from langchain_core.messages import HumanMessage, AIMessage

from langchain.tools import tool

load_dotenv()

agent = create_agent(
    model="openai:gpt-5-mini"
)

message = HumanMessage(content="Tell me a dad joke")

result = agent.invoke({
    "messages": message
})

print(result)