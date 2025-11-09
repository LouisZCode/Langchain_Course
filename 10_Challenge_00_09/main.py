from turtle import pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, dynamic_prompt, ModelRequest
from langchain_core.messages import HumanMessage

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


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


# TODO: Include type hints with Literal for constrained parameters

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
def add_job_application(company : str, position:str, status:str, deadline:str, salary:float, notes:str) -> float:
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
    new_row = pd.DataFrame([{"company": company.casefold(),  "position": position, "status": "Just Applied", "deadline": deadline_date, "applied_date":applied_date, "salary_range": salary, "notes": notes}])
    df = pd.concat([dataframe , new_row], ignore_index=True)
    df.to_csv(APPLICATIONS, index=False)
    return f"f'New row createw with the following values:{new_row}"

@tool(
    "Edit_job_application_status",
    parse_docstring=True,
    description=(
        "edits the status of the Job application on a specific company according to the users request"
    )
)
def edit_job_status(new_status : str, company_name : str) -> str:
    """
    Description:
        Edits the status of a specified job in the database, to the status requested by the user.

    Args:
        new_status (str): the desired new status of that job application.
        company_name (str): the name of the company where to update the job application status.

    Returns:
        The confirmation of the company status that ahs been updated, with the details.

    Raises:
        If there is no company named like that, lets the user know
    """

    df = pd.read_csv(APPLICATIONS)
    #align wthe value to the one the databas would have:
    company_name = company_name.casefold()
    #We need to first filter the row, if it exist
    if company_name in df["company"].values:
        df.loc[df["company"] == company_name, "status"] = new_status
        df.to_csv(APPLICATIONS, index=False)
        return f"The application to the {company_name} has been updated to {new_status}"

    else:
        return "This company does not exist in the database"

@tool(
    "delete_job_application",
    parse_docstring=True,
    description=(
        "used to delete a complete row in the database, erasing the job application"
    )
)
def delete_job_application(company_name : str) -> str:
    """
    Description:
        erases a job application of the database simply by removing the row where the application was saved

    Args:
        company_name (str): The name of the company where the job application was done.

    Returns:
        Confirmation or Rejection by Admin of this job application

    Raises:
        Lets the user know if no application in the requested company exist in the first place
    """

    company_name = company_name.casefold()
    df = pd.read_csv(APPLICATIONS)
    if company_name not in df["company"].values:
        return f"There was no application to the {company_name} made before"
    else:
        df = df[df["company"] != company_name]
        df.to_csv(APPLICATIONS, index=False)
        return f"the application data for the {company_name} has been permanently deleted" 


# TODO: Add a delete tool, that will have a HumanInTheLoop to confirm or reject the deletion of the application
# TODO Add Memory, so we can have various tool calls in the same session.


# TODO Cover Letter generation tool, with a TypedDict

agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[read_job_application_database, add_job_application, edit_job_status, delete_job_application],
    middleware=[my_prompt,
    HumanInTheLoopMiddleware(interrupt_on={"delete_job_application" : {
            "allowed_decisions": ["approve", "reject"],
            "description": "confirm or reject the permanent deletion of this job application"
            }
            }
        )
    ],
    checkpointer=InMemorySaver()
)

message = HumanMessage(content=SYSTEM_PROMPT)


result = agent.invoke(
    {"messages": message},
    {"configurable" : {"thread_id" : "01"}}
    )

if "__interrupt__" in result:
    interrupt_info = result["__interrupt__"]
    print("⚠️ Approval needed!")
    print()

    decision = input("Deleting a job application, continue? (yes/no): \n")
    decision = decision.casefold()

    if decision in ["yes" , "y"]:
        decision = "approve"
    elif decision in ["no", "n"]:
        decision = "reject"
    else:
        print("Invalid input, treating as reject")
        decision = "reject"


    result = agent.invoke(
        Command(resume={"decisions": [{"type": decision}]}),
        {"configurable" : {"thread_id" : "01"}}
        )

    print("✅ Action completed!")

else:
#Make the answers pretty for now
    for i,msg in enumerate(result["messages"]):
        msg.pretty_print()