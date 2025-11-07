from turtle import pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
#from langchain.agents.middleware #how to get the dynamic prompting part!?
from langchain_core.messages import HumanMessage

import os

import pandas as pd
import gradio as gr

from langchain.tools import tool

APPLICATIONS = "applications.csv"

if not os.path.exists(APPLICATIONS):
    new_datafreame = pd.DataFrame(columns=["company", "position", "status", "deadline", "applied_date", "salary_range", "notes"])
    new_datafreame.to_csv(APPLICATIONS, index=False)
    #Save the database?

load_dotenv()


SYSTEM_PROMPT = """
You are a professiona job application tracker agent.
Your job is to help the user follow on his or her job applications, and
to create cover letters for new applications if requested.
For that, you have access to different tools, and you always use
them to get the job done. You tell the user what you did, and only that.
You do not propose next steps.
If you dont seem to have access to the tools, you say:
'Sorry, I dont seem to have access to my tools right now' and no more*
instead of inventing something.
You dont propose never next steps, you follow the lead of the user.
"""

@tool(
    "read_current_applications",
    parse_docstring=True,
    description=(
        "Read the database with all the job application information"
    )
)
def read_job_application_database():
    """
    This tool helps you read the database with the current job application data

    Returns: the dataframe
    """
    datafreame = pd.read_csv(APPLICATIONS)

    return datafreame



agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[read_job_application_database]
)

message = HumanMessage(content=SYSTEM_PROMPT)


result = agent.invoke({
    "messages": message
})

#Make the answers pretty for now
for i,msg in enumerate(result["messages"]):
    msg.pretty_print()