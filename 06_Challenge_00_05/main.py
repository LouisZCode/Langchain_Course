from dotenv import load_dotenv
from langchain.agents import create_agent


load_dotenv()

agent = create_agent(
    model="anthropic:claude-haiku-4-5",
    system_prompt="You are Ryan"
)



result = agent.invoke({"messages" : "Hello! who are you?"})
print (result)