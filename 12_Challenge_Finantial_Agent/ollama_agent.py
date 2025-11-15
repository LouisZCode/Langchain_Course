from langchain.agents import create_agent
from langchain_ollama import ChatOllama


def call_ollama(query : str) -> str:
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

    message = query

    result = agent.invoke({
        "role" : "user",
        "messages" : message
    }
    )

    return result["messages"][1].content

"""see = call_ollama("tell me ajoke about France")
print(see)"""