from langchain_ollama import ChatOllama
from langchain.agents import create_agent


SYSTEM_PROMPT="Youa re a full stack comedian"

model = ChatOllama(
    model="gemma3:27b"
)

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT
)

query = input("Your message:\n")

result = agent.invoke({
    "role" : "user",
    "messages" : query
    }
)

for i,msg in enumerate(result["messages"]):
    msg.pretty_print()