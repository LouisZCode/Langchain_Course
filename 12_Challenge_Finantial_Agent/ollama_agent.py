from langchain.agents import create_agent
from langchain_ollama import ChatOllama



def call_ollama(system_prompt : str, query : str) -> str:


    model = ChatOllama(
        model="gemma3:27b"
    )

    agent = create_agent(
        model=model,
        system_prompt=system_prompt
    )

    result = agent.invoke({
        "messages" : query
        }
    )

    return result["messages"][1].content

"""see = call_ollama("tell me ajoke about France")
print(see)"""