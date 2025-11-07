
from dotenv import load_dotenv
import pandas as pd
import os
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
import time
from datetime import datetime
import gradio as gr


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
    "see_shopping_list",
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


@tool(
    "add_item_to_database",
    parse_docstring=True,
    description="Add a new item to the database as a new row"
)
def add_item(item_name: str, quantity: int, category:str):
    """this tool adds an item to the shopping list database
    
    Args:
        item_name (str): The name of the item.
        quantity (int): a float or integer as the quantity of the item.
        category (str): the category this item belongs to.

    Return:
        The new row information wwith the item, quantity, category and date created

    Raises:
        let the user know if one of the arguments is not there
    
    """
    date = datetime.now()
    df = pd.read_csv(ITEMS_DATABASE)
    new_row = pd.DataFrame([{"item" : item_name,  "quantity": quantity, "category": category, "added_date": date}])
    df = pd.concat([new_row, df], ignore_index=True)
    df.to_csv(ITEMS_DATABASE, index=False)
    return(f"successfully added this new rox: {new_row}")

@tool(
    "delete_item_from_list",
    parse_docstring=True,
    description="Deletes an item form the list"
)
def delete_item(item_name : str):
    """Deletes a row with the name of the item
    
    Args:
        item_name (str): the name of the item to delete with the whole row it is in

    returns:
        a confirmation that the item was deleter
    
    Raises:
        Lets the user know if the item requested to be deleted, doe snot exist.
    """
    df = pd.read_csv(ITEMS_DATABASE)
    print(item_name)
    filtered_df = df[df["item"] == item_name]
    if filtered_df.empty:
        return f"there is not an item with the name: {item_name}. Suggest the user to request to see the database"
    else:
        df = df[df["item"] != item_name]
        df.to_csv(ITEMS_DATABASE, index=False)
        return(f"the {item_name} has been deleted succesfully form the database")



agent = create_agent(
    model="anthropic:claude-haiku-4-5",
    tools=[show_shopping_list, add_item, delete_item],
    system_prompt="You are a shopping assistant that will help the user with their shopping needs. Your task is to use the tools you have access to, to help the user. If you dont have a tool for that exact task, you let the user know instead of trying to do the task.You do not help with anything else, your complete focus is on the shopping assisting and the use of the tools.",
    checkpointer=InMemorySaver()
)


def respond(message, history):
    """
    message: current user input (str)
    history: Gradio's display history (not used for agent memory)
    """
    #history starts as   []
    messages_to_send = history + [{"role": "user", "content": message}]
    
    response = agent.invoke({"messages" : messages_to_send}, {"configurable" : {"thread_id": "20"}})  #

    ai_message = response['messages'][-1].content

    return ai_message

demo = gr.ChatInterface(fn=respond, type="messages")
#The text box inside this ChatInterface, will send a  "message"  as a str variable to the function

demo.launch()