from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

load_dotenv()


@tool
def check_haiku_lines(text: str):
	"""check if the given haiku has exactly 3 lines.
	Returns NONE if it is correct, otherwise an error message"""
	#split the text into lines, ignoring leadint/trailing spaces
	lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
	print(f"TOO USED\nchecking haiku, it has {len(lines)} lines:\n{text}")
	
	if len(lines) != 3:
		return f"incorrect, this haiku has {len(lines)} lines. A haiku must have 3 lines"
	return "Correct, this haiku has 3 lines"


decision = input("choose your carachter:\n01: OpenAI\n02: Claude\n")

if decision == "01":
    todays_model = "openai:gpt-5-mini-2025-08-07"
elif decision == "02":
    todays_model = "anthropic:claude-haiku-4-5"


agent = create_agent(
    model=todays_model,
    tools=[check_haiku_lines],
    system_prompt="You are a sport poem creator. You always check your work using the tools you have access to"
)

human_msg = HumanMessage(content="make a Haiku about basketball")

result = agent.invoke({"messages": [human_msg]})

#print(result['messages'][-1].content)
for i, msg in enumerate(result["messages"]):
    print(msg)
    #msg.pretty_print()
