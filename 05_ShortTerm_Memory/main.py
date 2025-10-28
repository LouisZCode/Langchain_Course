from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
#New for the Memory Short term:
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

agent = create_agent(
    model= "anthropic:claude-haiku-4-5",
    system_prompt= "You are a english to German translator",
    checkpointer= InMemorySaver()
)

while True:

    x = input("\nwrite a message:\n")

    message = HumanMessage(content= x )

    result = agent.invoke(
        {"messages" : [message]},
        {"configurable": {"thread_id": "1"}}
        )
        
    for i, msg in enumerate(result["messages"]):
        msg.pretty_print()