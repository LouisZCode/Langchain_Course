from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

load_dotenv()


decision = input("choose your carachter:\n01: OpenAI\n02: Claude\n")

if decision == "01":
    todays_model = "openai:gpt-4"
elif decision == "02":
    todays_model = "anthropic:claude-sonnet-4-5"


agent = create_agent(
    model=todays_model,
    system_prompt="You are super honest and candid always. No preambles"
)

human_msg = HumanMessage(content="what model are you and who created you?")

result = agent.invoke({"messages": [human_msg]})

print(result['messages'][-1].content)
