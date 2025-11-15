from errno import ENETUNREACH
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

SYSTEM_PROMPT = """
You are a full stack comedian. There is no audience. Keep the Joke short.
You do not explain the joke
"""

model = ChatOllama(
    model="gemma3:27b"
)

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT
)

message = "Tell me a joke about France"

result = agent.invoke({
    "role" : "user",
    "messages" : message
}
)

for i,msg in enumerate(result["messages"]):
    msg.pretty_print()