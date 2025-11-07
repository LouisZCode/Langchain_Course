from email import message
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware

load_dotenv()

agent = create_agent(
    system_prompt = "you are a full stack comedian",
    model="openai:gtp-5-mini",
    middleware = [HumanInTheLoopMiddleware()]
    )

agent.invoke()