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

TRADE_LOG = "trades_log.csv"
PORTFOLIO = "my_portfolio.csv"
CASH_LOG = "my_cash.csv"
AVAILABLE_CASH = 0


## Create or check if the databases exist, if not, create an empty one

if not os.path.exists(TRADE_LOG):
    new_dataframe = pd.DataFrame(columns=["buy_or_sell", "ticket_symbol", "number_of_stocks", "individual_price_bought", "total_cost_trade", "date_bought"])
    new_dataframe.to_csv(TRADE_LOG, index=False)

if not os.path.exists(PORTFOLIO):
    new_dataframe = pd.DataFrame(columns=["ticket_symbol", "porcentage_weight", "number_of_stocks", "average_price", "total_cost_stock", "total_PL"])
    new_dataframe.to_csv(PORTFOLIO, index=False)

if not os.path.exists(CASH_LOG):
    new_dataframe = pd.DataFrame(columns=["add_or_withdraw","cash_ammount", "date_of_transaction"])
    new_dataframe.to_csv(CASH_LOG, index=False)
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

## global add-cash tool
@tool(
    "add_cash",
    parse_docstring=True,
    description="adds cash to the portfolio cash position"
)
def add_cash(cash_ammount : float) -> str:
    """
    Description:
        adds cash to the account

    Args:
        cash_ammount (str) : the cash ammount to be added to the account

    Returns:
        Lets the user know that cash has been added

    Raises:
        Lets the user know if there is a lack of information to create this transaction
    """
    date_transaction = datetime.now()
    
    df = pd.read_csv(CASH_LOG)
    new_row = pd.DataFrame([{
    "add_or_withdraw" : "add",
    "cash_ammount" : cash_ammount, 
    "date_of_transaction" : date_transaction
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(CASH_LOG, index=False)
    
    return f"Added {cash_ammount} usd to the cah position"

@tool(
    "withdraw_cash",
    parse_docstring=True,
    description="adds cash to the portfolio cash position"
)
def withdraw_cash(cash_ammount : float) -> str:
    """
    Description:
        withraws cash to the account

    Args:
        cash_ammount (str) :to be removed from the acocunt

    Returns:
        Lets the user know that cash has been withrawn

    Raises:
        Lets the user know if there is a lack of information to create this transaction
    """
    date_transaction = datetime.now()
    
    df = pd.read_csv(CASH_LOG)
    new_row = pd.DataFrame([{
    "add_or_withdraw" : "withdraw",
    "cash_ammount" : -cash_ammount, 
    "date_of_transaction" : date_transaction
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(CASH_LOG, index=False)
    
    return f"Withdrew {cash_ammount} usd from the cash position"

@tool(
    "count_cash",
    parse_docstring=True,
    description="tell the total of available cash in cash position"
)
def cash_position_count() -> str:
    #Sum and Rest all the cash logs to give a final cash position
    """
    Description:
        gives you the total ammount of available cash in the users account

    Returns:
        Lets the user know how much cash is available

    Raises:
        Lets the user know if there is a lack of information to create this transaction
    """
    df = pd.read_csv(CASH_LOG)
    cash_column_total = df["cash_ammount"].sum()

    return f"the user has {cash_column_total} available usd"


## Tools for Trades Agent
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
    dataframe = pd.read_csv(TRADE_LOG)
    trading_log = dataframe.to_markdown(index=False)
    return trading_log 


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
    
    df = pd.read_csv(TRADE_LOG)
    new_row = pd.DataFrame([{
        "buy_or_sell" : "buy",
        "ticket_symbol" : ticket_symbol,
        "number_of_stocks" : number_of_stocks,
        "individual_price_bought" : individual_price_bought,
        "total_cost_trade" : total_cost_trade,
        "date_bought" : date_bought
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(TRADE_LOG, index=False)
    
    return f"New trade saved with these details:\n 'ticket_symbol': {ticket_symbol}\n'number_of_stocks': {number_of_stocks}\n'individual_price_bought': {individual_price_bought}\n'total_cost_trade': {total_cost_trade}\n'date_bought': {date_bought}"
    

# TODO Tool to sell or delete a row form the stock list
# TODO add MiddleWare to have HumanInTheLoop to confirma  SELL of a stock.



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
    tools=[read_my_portfolio, add_to_portfolio, add_cash, withdraw_cash, cash_position_count],
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

        with gr.Tab("Trade Assistant"):
            gr.ChatInterface(
                fn=response_my_portfolio,
                additional_inputs=[waiting_for_approval_state],
                additional_outputs=[waiting_for_approval_state]
            )

demo.launch()