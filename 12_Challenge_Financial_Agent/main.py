from langchain.agents import create_agent
#Ollama agents not able to use tools, to figure out later.
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import create_retriever_tool, tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import os
import pandas as pd

from tabulate import tabulate
from datetime import datetime

import gradio as gr
import yaml

from dotenv import load_dotenv

from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.types import Command

MY_PORTOLIO = "my_portfolio.csv"


## Create or check if the databases exist, if not, create an empty one

if not os.path.exists(MY_PORTOLIO):
    new_dataframe = pd.DataFrame(columns=["ticket_symbol", "number_of_stocks", "individual_price_bought", "total_cost_trade", "date_bought"])
    new_dataframe.to_csv(MY_PORTOLIO, index=False)

    #and adds a "cash" row with 0 dollars.


##  Load of API Keys and Prompts
load_dotenv()

with open("prompts.yaml", "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)
    quarter_results_prompt = prompts["QUATERLY_RESULTS_EXPERT"]
    my_portfolio_prompt = prompts["MY_PORTFOLIO_EXPERT"]

#print(prompt)


##Create Retriever RAG Tool
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.load_local("Quaterly_Reports", embedding, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
retriever_tool = create_retriever_tool(
    retriever,
    name="retriever_tool",
    description="Search through the document knowledge base to find relevant information."
)

## Tools for My_Portfolio Agent
@tool(
    "read_my_portfolio",
    parse_docstring=True,
    description="reads the current potfolio information of the user"
)
def read_my_portfolio():
    """
    Description:
        Reads the current information saved in the users portfolio

    Args:
        None

    Returns:
        The dataframe information inside the portfolio

    Raises:
        Lets the user know if there is no information inside the portfolio
    """
    dataframe = pd.read_csv(MY_PORTOLIO)
    portfolio_inforamtion = dataframe.to_markdown(index=False)
    return portfolio_inforamtion


@tool(
    "add_to_portfolio",
    parse_docstring=True,
    description="adds a new buy row to the portfolio of the user"
)
def add_to_portfolio(ticket_symbol : str, number_of_stocks : float, individual_price_bought : float) -> str:
    """
    Description:
        adds a new buy row to the portfolio of the user

    Args:
        ticket_symbol (str) : the initials of the stock bought, number_of_stocks (float) : the quantity of stocks bought in this transaction, individual_price_bought (float): the proce of each individual stock, total_cost_trade (float): the result of the multiplication of number of stocks bough by the cost per individual stock, date_bought: todays date.

    Returns:
        Lets the user know a new buy has been done and gives the details of that new transaction

    Raises:
        Lets the user know if there is a lack of information to create this transaction
    """
    date_bought = datetime.now()

    total_cost_trade = number_of_stocks * individual_price_bought
    
    df = pd.read_csv(MY_PORTOLIO)
    new_row = pd.DataFrame([{
        "ticket_symbol" : ticket_symbol,
        "number_of_stocks" : number_of_stocks,
        "individual_price_bought" : individual_price_bought,
        "total_cost_trade" : total_cost_trade,
        "date_bought" : date_bought
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(MY_PORTOLIO, index=False)
    
    return f"New trade saved with these details:\n 'ticket_symbol': {ticket_symbol}\n'number_of_stocks': {number_of_stocks}\n'individual_price_bought': {individual_price_bought}\n'total_cost_trade': {total_cost_trade}\n'date_bought': {date_bought}"
    


# TODO add MiddleWare to have HumanInTheLoop to confirma  buy of a stock.
# TODO it is cheap or expensive? Tool to ca



##Add allinfo to agent
quarter_result_agent = create_agent(
    model="openai:gpt-5-mini",
    system_prompt=quarter_results_prompt,
    checkpointer=InMemorySaver(),
    tools=[retriever_tool]
)


my_portfolio_agent = create_agent(
    model="openai:gpt-5-mini",
    system_prompt=my_portfolio_prompt,
    checkpointer=InMemorySaver(),
    tools=[read_my_portfolio, add_to_portfolio],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "add_to_portfolio": {
                    "allowed_decisions": ["approve", "reject"],
                    "description": "Confirm addition of new stock to portfolio"
                }
            }
        )
    ])

# TODO Agent that recommends or not different stocks (Multi Agent panel) and saves in a csv: Buy-Hold-Sell based on finantials
#and based on the current price action of the company:  A.k.a. access to finantial data.
# TODO This agent saves the information in a csv and shows it. gets the info form 2 years and shows a 10-25-50 % disscount, and
#if it is a good, bad and X price for the ticket.



##Gradio-ing
def response_quaterly(message, history):

    response = quarter_result_agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        {"configurable": {"thread_id": "thread_001"}}
    )

    for i, msg in enumerate(response["messages"]):
        msg.pretty_print()

    return response["messages"][-1].content



def response_my_portfolio(message, history, waiting_for_approval):

    if waiting_for_approval:

        decision = message.lower().strip()

        if decision in ["yes" , "y", "approve"]:
            decision = "approve"
        elif decision in ["no", "n", "reject"]:
            decision = "reject"
        else:
            decision = "reject"

        response = my_portfolio_agent.invoke(
            Command(resume={"decisions": [{"type": decision}]}),
            {"configurable" : {"thread_id" : "thread_002"}}
            )

        
        for i, msg in enumerate(response["messages"]):
            msg.pretty_print()

        return response["messages"][-1].content, False

    else:

        messages = history + [{"role": "user", "content": message}]

        response = my_portfolio_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            {"configurable": {"thread_id": "thread_002"}}
        )

        for i, msg in enumerate(response["messages"]):
            msg.pretty_print()

        if "__interrupt__" in response:

            approval_message = (
                f"⚠️ **Approval Required**\n\n"
                f"The agent wants BUY stock"
                f"Do you approve? (yes/no)"
                )

            return approval_message, True

        return response["messages"][-1].content, False


waiting_for_approval_state = gr.State(False)

with gr.Blocks() as demo:
    with gr.Tabs():

        with gr.Tab("Quaterly Reports Expert"):
            gr.ChatInterface(
                fn=response_quaterly
            )

        with gr.Tab("My Portfolio Management"):
            gr.ChatInterface(
                fn=response_my_portfolio,
                additional_inputs=[waiting_for_approval_state],
                additional_outputs=[waiting_for_approval_state]
            )

demo.launch()