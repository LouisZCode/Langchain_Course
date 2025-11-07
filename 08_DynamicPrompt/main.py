from dotenv import load_dotenv

from langchain.agents import create_agent

from langchain.agents.middleware.types import ModelRequest, dynamic_prompt

load_dotenv()

@dynamic_prompt
def anti_color(request: ModelRequest) -> str:
    #grab the message of the user:
    user_message = request.messages[-1].content.lower()
    if user_message == "red":
        return """Say: The opposite of this olor is WELL DONE"""
    else:
        return """Say: Correct color not Chosen!"""

agent = create_agent(
    model="openai:gpt-5-mini",
    middleware=[anti_color]
)


message = input("What color do you love?")

result = agent.invoke(
    {"messages" : [{"role" : "user", "content" : message}]}
)

print(result)
