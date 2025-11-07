from turtle import pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
#from langchain.agents.middleware #how to get the dynamic prompting part!?
from langchain_core.messages import HumanMessage, AIMessage

import os

import pandas as pd
import gradio as gr

from langchain.tools import tool

APPLICATIONS = "applications.csv"

if os.path.exists(APPLICATIONS):
    new_datafreame = pd.DataFrame(APPLICATIONS, columns=["company", "position", "status", "deadline", "applied_date", "salary_range", "notes"])
    new_datafreame.to_csv(APPLICATIONS, index=False)
    #Save the database?

load_dotenv()

agent = create_agent(
    model="openai:gpt-5-mini"
)

message = HumanMessage(content="Tell me a dad joke")

result = agent.invoke({
    "messages": message
})

#Make the answers pretty for now
for i,msg in enumerate(result["messages"]):
    msg.pretty_print()