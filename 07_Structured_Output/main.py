from dotenv import load_dotenv

from typing import TypedDict, NotRequired

from langchain.agents import create_agent

load_dotenv()


class ClassCreated(TypedDict):
	name: str
	age: NotRequired[int]


agent = create_agent(
    model="openai:gpt-5-mini",
    response_format=ClassCreated
)

recorded_conversation = """We talked to Jon doe"""

result = agent.invoke(
    {"messages" : [{"role" : "user", "content" : recorded_conversation }]}
)

print(result["structured_response"])