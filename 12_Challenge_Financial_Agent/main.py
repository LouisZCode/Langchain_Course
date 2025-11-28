from config import (
    TRADE_LOG, PORTFOLIO, CASH_LOG, STOCK_EVALS, 
    DB_PATH, load_prompts, EMBEDDING_MODEL
)

from database_init import initialize_databases

from cash_management_tools import (
    add_cash_tool, withdraw_cash_tool,
    cash_position_count
    )

from cash_management_helper_functions import (
    _update_portfolio_info, _extract_structured_data
)

from portfolio_operations import (
    read_my_portfolio, add_to_portfolio, remove_from_portfolio
    )

from langchain.agents import create_agent


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

import asyncio

import re
import random
import json
import ast

from edgar import set_identity, Company

from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from tqdm import tqdm

import time
import asyncio


set_identity("Juan Perez juan.perezzgz@hotmail.com")

# we initialize the databases form the database_init.py
initialize_databases(TRADE_LOG, PORTFOLIO, CASH_LOG, STOCK_EVALS)


prompts = load_prompts()
quarter_results_prompt = prompts["QUATERLY_RESULTS_EXPERT"]
my_portfolio_prompt = prompts["MY_PORTFOLIO_EXPERT"]
checker_prompt = prompts["CHECKER"]
explainer_prompt = prompts["EXPLAINER"]
OPPORTUNITY_FINDER_PROMPT_TEMPLATE = prompts["PORTFOLIO_RECOMMENDATOR"]

##  Create Retriever RAG Tool
"""
This created the retriever for the RAG and gives a retriever_tool to be used bxy an agent
"""

embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_store = FAISS.load_local(DB_PATH, embedding, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

retriever_tool = create_retriever_tool(
    retriever,
    name="retriever_tool",
    description="Search through the document knowledge base to find relevant information."
)




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
    tools=[retriever_tool],
    response_format=FinancialInformation
)


anthropic_finance_boy = create_agent(
    model="anthropic:claude-haiku-4-5",
    system_prompt=quarter_results_prompt,
    checkpointer=InMemorySaver(),
    tools=[retriever_tool],
    response_format=FinancialInformation
)

google_finance_boy = create_agent(
    model="google_genai:gemini-2.5-flash",
    system_prompt=quarter_results_prompt,
    checkpointer=InMemorySaver(),
    tools=[retriever_tool],
    response_format=FinancialInformation
)

simple_explaining_agent = create_agent(
    model="anthropic:claude-haiku-4-5",
    system_prompt=explainer_prompt,
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


opportunity_agent = create_agent(
    model="anthropic:claude-haiku-4-5",
    system_prompt=explainer_prompt,
    tools=[read_my_portfolio, review_stock_data],
    checkpointer=InMemorySaver()
)


## LLM Councel Response Function

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
        yield "No clear symbol or company mentioned, could you try again please?"
        return 
    
    else:
        ticker_symbol = check_ticker["messages"][-1].content
        yield f"I have found the ticker symbol {ticker_symbol} in the users query, thinking..."
        time.sleep(1)

        if ticker_admin_tool(ticker_symbol):
            yield "The councel already had researched this Ticker, gathering the info form the database..."
            time.sleep(2)
            #agent that explains the info:
            ticker_info = ticker_info_db(ticker_symbol)

            explainer_agent = simple_explaining_agent.invoke(
            {"messages": [{"role": "user", "content": f"{ticker_info}"}]}
            )
            explainer_agent["messages"][-1].content

            yield explainer_agent["messages"][-1].content
            return
        
        else:

            yield "Getting data for this company from the SEC directly, this will take 1 minute..."
            await asyncio.sleep(1)
            download_clean_filings(ticker_symbol)
            
            #OPENAI Research
            yield "Data received, now the councel with review the data and come with a veridict, just a moment..."
            time.sleep(2)

            prices_pe_data = _stock_market_data(ticker_symbol)

            response_openai = await openai_finance_boy.ainvoke(
                {"messages": [{"role": "user", "content": f"Research {ticker_symbol}, more info: {prices_pe_data}"}]},
                {"configurable": {"thread_id": "thread_001"}}
            )
            data_openai = _extract_structured_data(response_openai["messages"][-1].content)
            #print(f"OpenAi Says:{data_openai}")
            yield f"OpenAI recommends: {data_openai["recommendation"]}\n\n"
            time.sleep(2)


            #CLAUDE Research
            response_claude = await anthropic_finance_boy.ainvoke(
                {"messages": [{"role": "user", "content": f"Research {ticker_symbol}, more info: {prices_pe_data}"}]},
                {"configurable": {"thread_id": "thread_001"}}
            )
            data_claude = _extract_structured_data(response_claude["messages"][-1].content)
            #print(f"Claude says: {data_claude}")
            yield f"Claude recommends: {data_claude["recommendation"]}\n\n"


            #Gemini Research
            response_gemini = await google_finance_boy.ainvoke(
                {"messages": [{"role": "user", "content": f"Research {ticker_symbol}, more info: {prices_pe_data}"}]},
                {"configurable": {"thread_id": "thread_001"}}
            )
            data_gemini = _extract_structured_data(response_gemini["messages"][-1].content)
            #print(f"Google says: {data_claude}")
            yield f"Gemini recommends: {data_claude["recommendation"]}\n\n"


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
                yield f"The councel of LLMS recommends to BUY this stock, the reason:\n\n{selected_reason}\n\n{saved_database}"

            elif recommendations_list.count("Sell") >= 2:
                yield f"The councel of LLMS recommends to SELL this stock, the reason:\n{selected_reason}\n\n{saved_database}"

            else:
                yield f"The councel of LLMS recommends to HOLD this stock, the reason:\n{selected_reason}\n\n{saved_database}"
            
## Portfolio Manager reposne_function
   
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

## Find Tailored Opportunities Agent   response function
    
async def find_opportunities(message, history, risk_state):

    response = await opportunity_agent.ainvoke(
        {"messages": [{"role": "user", "content": f"my risk restuls are: {risk_state}, and here is my query: {message}"}]},
        config={"configurable": {"thread_id": "opp_thread_1"}}
    )

    return response["messages"][-1].content


def update_risk_state(risk_value):
    print(f"Model changed to: {risk_value}")
    
    if risk_value == "Y.O.L.O":
        risk_value = "I prefer you to Identify high-volatility, speculative micro-cap stocks with massive upside potential. Ignore standard safety metrics. Focus on aggressive growth narratives."
    if risk_value =="I tolerate a lot of RISK":
        risk_value ="I prefer you to Focus on growth stocks with high beta. Accept significant volatility for the chance of market-beating returns."
    if risk_value =="I tolerate little risk":
        risk_value ="I prefer you to Balance growth and stability. Look for established companies with decent growth prospects and reasonable valuations."
    if risk_value =="Lets take NO risks":
        risk_value ="I prefer you to Prioritize capital preservation and steady income. Focus on blue-chip, dividend-paying aristocrats with low volatility."


    return risk_value  # Return new state value

## Gradio - ing

waiting_for_approval_state = gr.State(False)
risk_state = gr.State("")
_update_portfolio_info()

with gr.Blocks() as demo:
    with gr.Tabs():
        
        #Stock Evaluation Tab, saved into stock_evaluation.csv
        with gr.Tab("Counsel of LLMs"):
            gr.Markdown("# The Councel that will research and categorize a stock for you...") 
            gr.ChatInterface(
                fn=response_quaterly,
                type="messages"
            )
            gr.Markdown("### NOTE: Answer based on real SEC data, but on mock stock price and P/E ratio because of API Costs.\nPlease dont use this Tech-Demo as financial advice") 

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
            risk_dropdown = gr.Dropdown(
                label="Select your Risk Tolerance",
                interactive=True,
                choices=["Y.O.L.O", "I tolerate a lot of RISK", "I tolerate little risk", "Lets take NO risks"],
                )

            risk_dropdown.change(
                fn=update_risk_state,
                inputs=risk_dropdown,
                outputs=risk_state
                )

            gr.ChatInterface(
                fn=find_opportunities,
                additional_inputs=[risk_state],
                type="messages"
            )

demo.launch(share=False)
