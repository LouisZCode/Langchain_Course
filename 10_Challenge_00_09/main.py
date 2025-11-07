from turtle import pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, dynamic_prompt, ModelRequest
#from langchain.agents.middleware #how to get the dynamic prompting part!?
from langchain_core.messages import HumanMessage

import os

from datetime import datetime, timedelta

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
to do now = {current_instructions}
"""

human_message = input("Tell the Agent the first action:\n")

@dynamic_prompt
def my_prompt(request:ModelRequest) -> str:
    if human_message:
        current_instructions = human_message
    return (SYSTEM_PROMPT.format(current_instructions = current_instructions))


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


@tool(
    "add_job_application_row",
    parse_docstring=True,
    description=(
        "You can add a new row to the database using this tool"
    )
)
def add_job_application(company : str, position:str, status:str, deadline:str, salary:float, notes:int) -> float:
    """
    creates a new row in the database with the information given by the user. Needs all parameters to work

    Args:
        company (str): The company name.
        position (str): the position the application is for
        status (str): The current situation of that application. starts at "Just Applied"
        deadline (str): a time limit
        salary (float): the salary budget offered for this job
        notes: (str): additional option information provided in case there is need for them

    Returns:
        Confirmation that a new row has been added to the database

    Raises:
        an error in creating the new row, and details about what is missing of why it failed
    """
    applied_date = datetime.now().date()
    deadline_date = applied_date + timedelta(days=15)

    dataframe = pd.read_csv(APPLICATIONS)
    new_row = pd.DataFrame([{"company": company,  "position": position, "status": "Just Applied", "deadline": deadline_date, "applied_date":applied_date, "salary_range": salary, "notes":notes}])
    df = pd.concat([dataframe , new_row], ignore_index=True)
    df.to_csv()
    return f"f'New row createw with the following values:{new_row}"


agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[read_job_application_database, add_job_application],
    middleware=[my_prompt]
)

message = HumanMessage(content=SYSTEM_PROMPT)


result = agent.invoke({
    "messages": message
})

#Make the answers pretty for now
for i,msg in enumerate(result["messages"]):
    msg.pretty_print()