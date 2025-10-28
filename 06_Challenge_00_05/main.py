from dotenv import load_dotenv
import pandas as pd
import os
from langchain.agents import create_agent
from langchain_core.tools import tool
import time

load_dotenv()

ITEMS_DATABASE = "itemsDB.csv"


#We check if the database does not exists:
if not os.path.exists(ITEMS_DATABASE):
    print("Database does not exists, creating one:")
    time.sleep(1)
    #if not, create an empty one
    df = pd.DataFrame(columns=["item", "quantity", "category", "added_date"])
    df.to_csv(ITEMS_DATABASE, index=False)
    print("Database created succesfully!")
    time.sleep(1)

print("Databse loaded successfully...")
time.sleep(1)

#Function-Tool to Add tot he shopping list

#Function to show the shopping list
@tool(
    "see shopping list",
    parse_docstring=True,
    description="shows the shopping list databse"
)
def show_shopping_list():
    """This tool shows the shopping list database
        
    Returns:
        The contents of the current shopping list database
        
    Raises:
        if there is no database, lets the user know"""

    df = pd.read_csv(ITEMS_DATABASE)
    return df

#whats the name of the tool(parameter)
#whats the format so the google_way is correct and not error


#Function to calculate budget? Question: how do we get the unit price of the items?


#Search and connect to 1 MCP server... random but lets do it.
#Idea: Mcp that browses online for prices of X Item


agent = create_agent(
    model="anthropic:claude-haiku-4-5",
    tools=[],
    system_prompt="You are a shopping assistant that will help the user with their shopping needs. Your task is to use the tools you have access to, to help the user. If you dont have a tool for that exact task, you let the user know instead of trying to do the task.You do not help with anything else, your complete focus is on the shopping assisting and the use of the tools."
)

#The format of the messages so they are "pretty_printed"

result = agent.invoke({"messages" : "Hello! what can you do?"})
for i, msg in enumerate(result["messages"]):
    msg.pretty_print()
    #print(msg)


#Once it iw working with tools and memory, we implement Gradio, which sounds like it is going to change the procedural formatting ofthe code