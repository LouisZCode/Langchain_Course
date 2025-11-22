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

from typing import TypedDict, NotRequired

from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

import re
import random
import json
import ast

TRADE_LOG = "trades_log.csv"
PORTFOLIO = "my_portfolio.csv"
CASH_LOG = "my_cash.csv"
STOCK_EVALS = "stock_evaluations.csv"


## region Create Databases CSVs  exist, if not create an empty one
"""
Here the system checks if any of the 3 databases exists, and if not, creates an empty one.
"""

if not os.path.exists(TRADE_LOG):
    new_dataframe = pd.DataFrame(columns=["buy_or_sell", "ticket_symbol", "number_of_stocks", "individual_price", "total_cost_trade", "date_transaction"])
    new_dataframe.to_csv(TRADE_LOG, index=False)

if not os.path.exists(PORTFOLIO):
    new_dataframe = pd.DataFrame(columns=["ticket_symbol", "porcentage_weight", "number_of_stocks", "average_price", "total_cost_stock", "total_PL"])
    new_dataframe.to_csv(PORTFOLIO, index=False)

if not os.path.exists(CASH_LOG):
    new_dataframe = pd.DataFrame(columns=["add_or_withdraw","cash_ammount", "date_of_transaction"])
    new_dataframe.to_csv(CASH_LOG, index=False)

if not os.path.exists(STOCK_EVALS):
    new_dataframe = pd.DataFrame(columns=["stock", "LLM_1", "LLM_2", "LLM_3", "price", "price_description", "p/e", "one_sentence_reasoning"])
    new_dataframe.to_csv(STOCK_EVALS, index=False)
   

# endregion

##  Load of API Keys and Prompts
load_dotenv()

#Load of the prompts yaml to be loaded by the different agents
with open("prompts.yaml", "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)
    quarter_results_prompt = prompts["QUATERLY_RESULTS_EXPERT"]
    my_portfolio_prompt = prompts["MY_PORTFOLIO_EXPERT"]
    checker_prompt = prompts["CHECKER"]
#check the prompt loaded in terminal by unlocking and changing the name of this part:
#print(prompt)

# endregion

##  Create Retriever RAG Tool
"""
This created the retriever for the RAG and gives a retriever_tool to be used bxy an agent
"""

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

# endregion

##  Portfolio Information Update Function
"""
This is a helper function that will update the information in the portfolio database, based ont the information of the 
trades logs database, and will organize it in a way that make sit understandable at a glamce.
Called everytime there is a movement or change in the portfolio like cash movements, or trades.
"""

def _update_portfolio_info(trade_log_path=TRADE_LOG, portfolio_path=PORTFOLIO, cash_log_path=CASH_LOG):
    """
    Rebuilds the portfolio.csv from scratch based on trade_log.csv,
    including cash position and percentage weights.
    """
    trade_log_df = pd.read_csv(trade_log_path)
    cash_log_df = pd.read_csv(cash_log_path)

    # Calculate total cash (already has +/- signs in the data)
    total_cash = cash_log_df['cash_ammount'].sum()

    # Start with empty portfolio dictionary
    portfolio = {}

    # Process all trades to rebuild portfolio from scratch
    for index, row in trade_log_df.iterrows():
        symbol = row['ticket_symbol']
        quantity = row['number_of_stocks']
        price = row['individual_price']
        buy_or_sell = row['buy_or_sell']

        if buy_or_sell == 'buy':
            if symbol not in portfolio:
                # New stock
                portfolio[symbol] = {
                    'number_of_stocks': quantity,
                    'total_cost_stock': quantity * price,
                    'average_price': price
                }
            else:
                # Add to existing position
                current_qty = portfolio[symbol]['number_of_stocks']
                current_cost = portfolio[symbol]['total_cost_stock']
                
                new_qty = current_qty + quantity
                new_cost = current_cost + (quantity * price)
                new_avg_price = new_cost / new_qty
                
                portfolio[symbol]['number_of_stocks'] = new_qty
                portfolio[symbol]['total_cost_stock'] = new_cost
                portfolio[symbol]['average_price'] = new_avg_price

        elif buy_or_sell == 'sell':
            if symbol in portfolio:
                # Reduce position based on average cost
                current_qty = portfolio[symbol]['number_of_stocks']
                current_avg_price = portfolio[symbol]['average_price']
                
                new_qty = current_qty - quantity
                cost_of_sold = quantity * current_avg_price
                new_cost = portfolio[symbol]['total_cost_stock'] - cost_of_sold
                
                if new_qty <= 0:
                    # Fully sold - remove from portfolio
                    del portfolio[symbol]
                else:
                    # Partial sale - keep average price the same
                    portfolio[symbol]['number_of_stocks'] = new_qty
                    portfolio[symbol]['total_cost_stock'] = new_cost
                    portfolio[symbol]['average_price'] = current_avg_price

    # Convert dictionary to DataFrame
    portfolio_data = []
    for symbol, data in portfolio.items():
        portfolio_data.append({
            'ticket_symbol': symbol,
            'number_of_stocks': data['number_of_stocks'],
            'average_price': data['average_price'],
            'total_cost_stock': data['total_cost_stock'],
            'porcentage_weight': 0.0,  # Calculate below
            'total_PL': 0.0
        })
    
    portfolio_df = pd.DataFrame(portfolio_data)

    # Check if portfolio has any stocks
    if len(portfolio_df) > 0:
        # Has stocks - calculate percentages
        total_portfolio_value = portfolio_df['total_cost_stock'].sum()
        total_account_value = total_portfolio_value + total_cash
        
        if total_account_value > 0:
            portfolio_df['porcentage_weight'] = round((portfolio_df['total_cost_stock'] / total_account_value) * 100, 2)
            cash_percentage = round((total_cash / total_account_value) * 100, 2)
        else:
            cash_percentage = 100.0
    else:
        # No stocks yet - 100% cash
        total_account_value = total_cash
        cash_percentage = 100.0

    # Add CASH row at the top
    cash_row = pd.DataFrame([{
        'ticket_symbol': 'CASH',
        'porcentage_weight': cash_percentage,
        'number_of_stocks': 0,
        'average_price': 0,
        'total_cost_stock': total_cash,
        'total_PL': 0.0
    }])

    # Combine cash row with portfolio
    portfolio_df = pd.concat([cash_row, portfolio_df], ignore_index=True)

    # Save to CSV
    portfolio_df.to_csv(portfolio_path, index=False)

"""
This is a helper function that will clean the answers form the open_ai, Claude and different LLMS into the same format,
so it is better digested by the "councel"
"""

def _extract_structured_data(response_content):
    """
    Extract structured data from LLM response, handling different formats.
    
    Works for:
    - OpenAI: '{"financials":"Strong",...}'
    - Claude: "Returning structured response: {'financials': 'Strong',...}"
    """
    content = response_content.replace("Returning structured response:", "").strip()
    # Try to find JSON object
    json_match = re.search(r'\{[^}]+\}', content)
    
    if json_match:
        json_str = json_match.group(0)
        try:
            # First try standard JSON (double quotes)
            return json.loads(json_str)
        except json.JSONDecodeError:
            # If that fails, try Python literal syntax (single quotes)
            try:
                return ast.literal_eval(json_str)
            except (ValueError, SyntaxError):
                pass # Fallback
    
    # Fallback
    try:
        return json.loads(content)
    except:
        return {} # Return empty dict or handle error gracefully

# endregion

##  cash management functions and tools
def _withdraw_cash(cash_ammount : float) -> str:
    df = pd.read_csv(CASH_LOG)
    cash_column_total = df["cash_ammount"].sum()

    if cash_column_total < cash_ammount:
        return "You dont have enough funds to withdraw that ammount"

    else:
        date_transaction = datetime.now()
        
        new_row = pd.DataFrame([{
        "add_or_withdraw" : "withdraw",
        "cash_ammount" : -cash_ammount, 
        "date_of_transaction" : date_transaction
        }])

        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(CASH_LOG, index=False)

        _update_portfolio_info()

    return f"Added {cash_ammount} usd to the cah position"


def _add_cash(cash_ammount: float) -> str:
    date_transaction = datetime.now()
    
    df = pd.read_csv(CASH_LOG)
    new_row = pd.DataFrame([{
    "add_or_withdraw" : "add",
    "cash_ammount" : cash_ammount, 
    "date_of_transaction" : date_transaction
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(CASH_LOG, index=False)

    _update_portfolio_info()

    return f"Added {cash_ammount} usd to the cah position"


@tool(
    "add_cash",
    parse_docstring=True,
    description="adds cash to the portfolio cash position"
)
def add_cash_tool(cash_ammount : float) -> str:
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
    success = _add_cash(cash_ammount)

    return success

@tool(
    "withdraw_cash",
    parse_docstring=True,
    description="Checks if the user has enough cash, and if so, removes cash from the available cash of the users account"
)
def withdraw_cash_tool(cash_ammount : float) -> str:
    """
    Description:
        Checks if the user has enough cash, and if so, removes cash from the available cash of the users account

    Args:
        cash_ammount (str) :to be removed from the acocunt

    Returns:
        Lets the user know that cash has been withrawn

    Raises:
        Lets the user know if there is a lack of information to create this transaction
    """

    df = pd.read_csv(CASH_LOG)
    cash_column_total = df["cash_ammount"].sum()

    if cash_column_total < cash_ammount:
        return "You dont have enough funds to withdraw that ammount"

    else:
        success = _withdraw_cash(cash_ammount)
        return success


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

# endregion

## region Tools for Trades Agent
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
    description="Automatically checks if there is enough cash, and if so, uses it to buy the stock."
)
def add_to_portfolio(ticket_symbol : str, number_of_stocks : float, individual_price_bought : float) -> str:
    """
    Description:
        Automatically checks if there is enough cash, and if so, uses it to buy the stock.

    Args:
        ticket_symbol (str) : the initials of the stock bought, number_of_stocks (float) : the quantity of stocks bought in this transaction, individual_price_bought (float): the proce of each individual stock, total_cost_trade (float): the result of the multiplication of number of stocks bough by the cost per individual stock, date_bought: todays date.

    Returns:
        Lets the user know a new buy has been done and gives the details of that new transaction

    Raises:
        Lets the user know if there is a lack of information to create this transaction
    """
    df = pd.read_csv(CASH_LOG)
    cash_column_total = df["cash_ammount"].sum()
    total_cost_trade = number_of_stocks * individual_price_bought

    if cash_column_total < total_cost_trade:
        return "You dont have enough funds to buy this ammount of this stock"

    else:

        date_bought = datetime.now()
        
        df = pd.read_csv(TRADE_LOG)
        new_row = pd.DataFrame([{
            "buy_or_sell" : "buy",
            "ticket_symbol" : ticket_symbol,
            "number_of_stocks" : number_of_stocks,
            "individual_price" : +individual_price_bought,
            "total_cost_trade" : -total_cost_trade,
            "date_transaction" : date_bought
        }])

        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(TRADE_LOG, index=False)

        _withdraw_cash(total_cost_trade)
        
        return f"New trade saved with these details:\n 'ticket_symbol': {ticket_symbol}\n'number_of_stocks': {number_of_stocks}\n'individual_price_bought': {individual_price_bought}\n'total_cost_trade': {total_cost_trade}\n'date_bought': {date_bought}. Cash already taken form the available cash"
    

@tool(
    "remove_from_portfolio",
    parse_docstring=True,
    description="Sells stock and automatically adds that cash to the total cash balance."
)
def remove_from_portfolio(ticket_symbol : str, number_of_stocks : float, individual_price_sold : float) -> str:
    """
    Description:
        Sells stock and automatically adds that cash to the total cash balance.

    Args:
        ticket_symbol (str) : the initials of the stock sold, number_of_stocks (float) : the quantity of stocks sold in this transaction, individual_price_sold (float): the price of each individual stock, total_return_trade (float): the result of the multiplication of number of stocks bough by the cost per individual stock, date_sold: todays date.

    Returns:
        Lets the user know a new sell has been done and gives the details of that new transaction

    Raises:
        Lets the user know if there is a lack of information to create this transaction
    """
    date_sold = datetime.now()

    total_cost_trade = number_of_stocks * individual_price_sold
    
    df = pd.read_csv(TRADE_LOG)
    new_row = pd.DataFrame([{
        "buy_or_sell" : "sell",
        "ticket_symbol" : ticket_symbol,
        "number_of_stocks" : number_of_stocks,
        "individual_price" : -individual_price_sold,
        "total_cost_trade" : +total_cost_trade,
        "date_transaction" : date_sold
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(TRADE_LOG, index=False)

    _add_cash(total_cost_trade)
    print(f"Added {total_cost_trade} cash to the cash position")
    
    return f"New trade saved with these details:\n 'ticket_symbol': {ticket_symbol}\n'number_of_stocks': {number_of_stocks}\n'individual_price_sold': {individual_price_sold}\n'total_cost_trade': {total_cost_trade}\n'date_bought': {date_sold}. The cash already was added to the cash balance"


@tool(
    "stock_market_data",
    parse_docstring=True,
    description="gives you the stock market prices necessary to answer, alongside the p/e ratio of the company"
)
def stock_market_data(ticket_symbol : str) -> str:
    """
    Description:
        Gets you the lowest and highest price of a stock in the last 2 years and the pe ratio

    Args:
        ticket_symbol (str): The ticket symbol to research

    Returns:
        ticket symbols highest and lowest price in the lasz 2 years, plus the pe ratio

    Raises:
        If there is not wnough information about the symbol and or an error in the API Call
    """

    ticket_symbol = ticket_symbol.upper()
    #Sadly, API calls are only 25 per day, so will be using mocking data for this exercise:
    lower_price = random.randint(10 , 200)
    higher_price = random.randint(201 , 500)

    pe_ratio = random.randint(10 , 40)

    return f"the ticket symbol {ticket_symbol} has a lowest price of {lower_price}, and highest of {higher_price}, with a pe ratio of {pe_ratio} times per sales"

# endregion

## region Tools for Councel

def _save_stock_evals(ticket_symbol : str, LLM_1 : str, LLM_2 : str, LLM_3 : str, price : float, price_description : str,  p_e : str, selected_reason : str) -> str:
    """
    Description:
        Saves the stock evals in a csv file

    Args:
        ticket_symbol (str): The ticket symbol to research
        recommendations_list (list): The list of recommendations
        price (float): The price of the stock
        p_e (float): The p/e ratio of the stock
        selected_reason (str): The reason for the recommendation

    Returns:
        Lets the user know a new sell has been done and gives the details of that new transaction

    Raises:
        Lets the user know if there is a lack of information to create this transaction
    """
    

    df = pd.read_csv(STOCK_EVALS)
    new_row = pd.DataFrame({
        "stock": [ticket_symbol],
        "LLM_1": [LLM_1],
        "LLM_2" :[LLM_2],
        "LLM_3" : [LLM_3],
        "price": [price],
        "price_description": [price_description],
        "p/e": [p_e],
        "one_sentence_reasoning": [selected_reason]
    })
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(STOCK_EVALS, index=False)

    return "Succcessfully saved the stock recommendation into the stock evaluations database"


def ticker_admin_tool(ticker_symbol):
    """
    A function that checks if the ticker symbol requested is in the database already or not.
    If it is, returns True, if it is not, returns False.
    """
    #Check the first column in db
    df = pd.read_csv(STOCK_EVALS)
    ticker_column  = df["stock"].values

    if ticker_symbol in ticker_column:
        print(f"The ticker symbol {ticker_symbol} its already in the db")
        return True
    else:
        print(f"The ticker symbol {ticker_symbol} it not in the DB.\nGathering info from the SEC now...")
        return False

def ticker_info_db(ticker_symbol):
    df = pd.read_csv(STOCK_EVALS)
    
    return f"Here the info about {ticker_symbol} "

# endregion

##  Add all info to agents
class FinancialInformation(TypedDict):
    stock: str
    financials: str
    growth: str
    lower_stock_price : str
    higher_stock_price : str
    price_description: str
    price_to_earnings: str
    recommendation: str
    reason: str

checker_agent = create_agent(
    model="openai:gpt-5-mini",
    system_prompt=checker_prompt,
)

openai_finance_boy = create_agent(
    model="openai:gpt-5-mini",
    system_prompt=quarter_results_prompt,
    checkpointer=InMemorySaver(),
    tools=[retriever_tool, stock_market_data],
    response_format=FinancialInformation
)


anthropic_finance_boy = create_agent(
    model="anthropic:claude-haiku-4-5",
    system_prompt=quarter_results_prompt,
    checkpointer=InMemorySaver(),
    tools=[retriever_tool, stock_market_data],
    response_format=FinancialInformation
)

google_finance_boy = create_agent(
    model="google_genai:gemini-2.5-flash",
    system_prompt=quarter_results_prompt,
    checkpointer=InMemorySaver(),
    tools=[retriever_tool, stock_market_data],
    response_format=FinancialInformation
) 


my_portfolio_agent = create_agent(
    model="openai:gpt-5-mini",
    system_prompt=my_portfolio_prompt,
    checkpointer=InMemorySaver(),
    tools=[read_my_portfolio, add_to_portfolio, remove_from_portfolio,  add_cash_tool, withdraw_cash_tool, cash_position_count],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "add_to_portfolio": {
                    "allowed_decisions": ["approve", "reject"],
                    "description": "Confirm addition of new stock to portfolio"},

                "remove_from_portfolio" :{
                    "allowed_decisions": ["approve", "reject"],
                    "description": "Confirm removal of a stock from portfolio"}
                }
            )
        ])


# TODO This agent saves the information in a csv and shows it. gets the info form 2 years and shows a 10-25-50 % disscount, and
#if it is a good, bad and X price for the ticket.

# endregion

## region Gradio-ing

"""
Agent that reads the vector stores, and gives you info about the quaterly information
"""
async def response_quaterly(message, history):

    check_ticker = checker_agent.invoke(
        {"messages": [{"role": "user", "content": message}]}
    )

    print("before IF statements:")
    print(check_ticker["messages"][-1].content)
    print()

    if check_ticker["messages"][-1].content == 'No clear symbol or company mentioned, could you try again please?':
        print(check_ticker["messages"][-1].content)
        return 'No clear symbol or company mentioned, could you try again please?'
    
    else:
        ticker_symbol = check_ticker["messages"][-1].content
        print(f"checker found the ticker symbol {ticker_symbol} in the users query")

        if ticker_admin_tool(ticker_symbol):
            return ticker_info_db(ticker_symbol)
        
        else:
             # TODO: Function to get info form the ticker into the vector_store
                

            
            #OPENAI Research
            response_openai = await openai_finance_boy.ainvoke(
                {"messages": [{"role": "user", "content": f"Research {ticker_symbol}"}]},
                {"configurable": {"thread_id": "thread_001"}}
            )
            data_openai = _extract_structured_data(response_openai["messages"][-1].content)
            print(f"OpenAi Says:{data_openai}")
            print(f"OpenAI recommends: {data_openai["recommendation"]}\n\n")


            #CLAUDE Research
            response_claude = await anthropic_finance_boy.ainvoke(
                {"messages": [{"role": "user", "content": message}]},
                {"configurable": {"thread_id": "thread_001"}}
            )
            data_claude = _extract_structured_data(response_claude["messages"][-1].content)
            #print(f"Claude says: {data_claude}")
            print(f"Claude recommends: {data_claude["recommendation"]}\n\n")


            #Gemini Research
            response_gemini = await google_finance_boy.ainvoke(
                {"messages": [{"role": "user", "content": message}]},
                {"configurable": {"thread_id": "thread_001"}}
            )
            data_gemini = _extract_structured_data(response_gemini["messages"][-1].content)
            #print(f"Google says: {data_claude}")
            print(f"Gemini recommends: {data_claude["recommendation"]}\n\n")


            AI1 = data_openai["recommendation"]
            AI2 = data_claude["recommendation"]
            AI3 = data_gemini["recommendation"]

            AI1_reason = data_openai["reason"]
            AI2_reason = data_claude["reason"]
            AI3_reason = data_gemini["reason"]

            AI1_price = data_openai["higher_stock_price"]
            AI2_price = data_claude["higher_stock_price"]
            AI3_price = data_gemini["higher_stock_price"]

            AI1_pri_de = data_openai["price_description"]
            AI2_pri_de = data_claude["price_description"]
            AI3_pri_de = data_gemini["price_description"]


            AI1_pe = data_openai["price_to_earnings"]
            AI2_pe = data_claude["price_to_earnings"]
            AI3_pe = data_gemini["price_to_earnings"]


            recommendations_list = [AI1, AI2, AI3]
            reasons_list = [AI1_reason, AI2_reason, AI3_reason]
            price_list = [AI1_price, AI2_price, AI3_price]
            price = random.choice(price_list)
            p_e_list = [AI1_pe, AI2_pe, AI3_pe]
            p_e = random.choice(p_e_list)
            price_des_list = [AI1_pri_de, AI2_pri_de, AI3_pri_de]
            price_description = random.choice(price_des_list)

            selected_reason = random.choice(reasons_list)

            ticket_symbol = data_openai["stock"]

            saved_database = _save_stock_evals(ticket_symbol, AI1, AI2, AI3, price, price_description,  p_e, selected_reason)

            if recommendations_list.count("Buy") >= 2:
                return f"The councel of LLMS recommends to BUY this stock, the reason:\n\n{selected_reason}\n\n{saved_database}"

            elif recommendations_list.count("Sell") >= 2:
                return f"The councel of LLMS recommends to SELL this stock, the reason:\n{selected_reason}\n\n{saved_database}"

            else:
                return f"The councel of LLMS recommends to HOLD this stock, the reason:\n{selected_reason}\n\n{saved_database}"
            
   
"""
Agent that manages the portfolio
"""
def response_my_portfolio(message, history, waiting_for_approval):

    if waiting_for_approval:

        decision = message.lower().strip()

        if decision in ["yes" , "y", "approve", "buy", "sell"]:
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

        return response["messages"][-1].content, False, PORTFOLIO

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
                f"⚠️ **Approval Required** ⚠️\n\n"
                f"The agent wants BUY or SELL stock\n"
                f"Do you approve? (yes/no)"
                )

            return approval_message, True, PORTFOLIO

        return response["messages"][-1].content, False, PORTFOLIO


# TODO grab the data in the interrupt__ value, and use it to selfpopulate correctly aproval_message
# with the stock, if it is BUY or SELL, quantity and price

company_ticker_state = gr.State("")
waiting_for_approval_state = gr.State(False)
_update_portfolio_info()

with gr.Blocks() as demo:
    with gr.Tabs():
        
        #Stock Evaluation Tab, saved into stock_evaluation.csv
        with gr.Tab("Stock Research Counsel"):
            gr.Markdown("# The Stock Councel's Panel") 
            gr.ChatInterface(
                fn=response_quaterly,
                type="messages"
            )
            gr.Markdown("### NOTE: Answer based on mock stock price and pe ratio because of API Cost. Please dont use this Tech-Demo as financial advice") 

        #Manages and takes action on the current portfolio
        with gr.Tab("Trade Assistant"):
            gr.Markdown("# Your Portfolio") 
            portfolio_display = gr.DataFrame(PORTFOLIO)
            gr.Markdown("## Manage your Portfolio:") 
            gr.ChatInterface(
                fn=response_my_portfolio,
                additional_inputs=[waiting_for_approval_state],
                additional_outputs=[waiting_for_approval_state, portfolio_display],
                type="messages"
            )

        with gr.Tab("Find Opportunities"):
            gr.Markdown("# Decide What to Buy.. or not to...")
            gr.Markdown("## Decide Your Risk Tolerance:") 
            gr.Dropdown(
                label="Rick Tolerance",
                interactive=True,
                choices=["Y.O.L.O", "I tolerate a lot of RISK", "I tolerate little risk", "Lets take NO risks", "I'm Too Young to Die"]
                )
            gr.ChatInterface(
                fn=response_quaterly,   #Temp while building
                type="messages"
            )

demo.launch()

# endregion