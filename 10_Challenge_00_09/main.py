from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, dynamic_prompt, ModelRequest
from langchain_core.messages import HumanMessage

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from typing import TypedDict, NotRequired

import os

from datetime import datetime, timedelta

import pandas as pd
import gradio as gr
from gradio import ChatMessage

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
For that, you have access to different tools, and you always first check if you need to use
them to get the job done. If the tool was actually used, you tell the user what you did, and only that.
If no tool was used, let the user know, and answer the question.
You do not propose next steps.
If you dont seem to have access to the tools, you say:
'Sorry, I dont seem to have access to my tools right now' and no more*
instead of inventing something.

IMPORTANT: If the user just casually chat, you dont need to use the tools, you can normally 
answer as the extremely helpful job application tracker you are.

You dont propose never next steps, you follow the lead of the user.
to do now = {current_instructions}
"""


@dynamic_prompt
def my_prompt(request: ModelRequest) -> str:
    current_instructions = request.messages[-1].content
    return SYSTEM_PROMPT.format(current_instructions=current_instructions)


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
    dataframe = pd.read_csv(APPLICATIONS)

    return dataframe


@tool(
    "add_job_application_row",
    parse_docstring=True,
    description=(
        "You can add a new row to the database using this tool"
    )
)
def add_job_application(company : str, position:str, status:str, deadline:str, salary:float, notes:str) -> str:
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


# TODO Fix the READ DATABASE so the LLM can see it all (remove string only)

@tool(
    "cover_letter_writing",
    parse_docstring=True,
    description=
    "Allow you to write a cover letter for the user"
)
def cover_letter_writing(company_name : str, job_applied : str, header : str, body : str, goodbye : str) -> str:
    """
    Description:
        Creates a cover letter for the user specifically for the company requested and for the job applied to

    Args:
        company_name (str): The company name.
        job_applied (str): The job position.
        header (str): The greeting/header section.
        body (str): The main body of the letter.
        goodbye (str): The closing section.

    Returns:
        A saved cover letter in the system

    Raises:
        Error if not able to save the text.
    """
    # Combine the parts
    full_letter = f"{header}\n\n{body}\n\n{goodbye}"
    
    # Write to file
    filename = f"Cover_letter_to_{company_name}.txt"
    with open(filename, mode="w") as cl:
        cl.write(full_letter)

    return f"Cover Letter to {company_name} for the {job_applied} role has been created"



agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[read_job_application_database, add_job_application, edit_job_status, delete_job_application, cover_letter_writing],
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

thread_id_state = gr.State("001") 

def respond(message, history, thread_id):
    # 1. Build the full message list (history + current message)
    messages = history + [{"role": "user", "content": message}]
    
    # 2. Invoke the agent
    response = agent.invoke(
        {"messages": messages}, 
        {"configurable": {"thread_id": thread_id}}
    )
    
    return response["messages"][-1].content

demo = gr.ChatInterface(
    fn=respond,
    type="messages",
    additional_inputs=[thread_id_state]
)

demo.launch()