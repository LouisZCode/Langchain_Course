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

import gradio as gr
import yaml

from dotenv import load_dotenv

MY_PORTOLIO = "my_portfolio.csv"


## Create or check if the databases exist, if not, create an empty one

if not os.path.exists(MY_PORTOLIO):
    new_dataframe = pd.DataFrame(columns=["ticket_symbol", "date_bought", "price_bought"])
    new_dataframe.to_csv(MY_PORTOLIO, index=False)


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
    description="reads the current potfolio information"
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




# TODO Portfolio needs to have:  Ticket Symbol / Number of Stocks / Date bought / Proce bought at / Total Cost
# TODO   1: Portfolio LOGS and ways to sum etc.. 
# TODO - Tool to read the portfolio
# TODO - Tool to add to porfolio
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
    tools=[read_my_portfolio]
)


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

def response_my_portfolio(message, history):

    response = my_portfolio_agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        {"configurable": {"thread_id": "thread_001"}}
    )

    for i, msg in enumerate(response["messages"]):
        msg.pretty_print()

    return response["messages"][-1].content

with gr.Blocks() as demo:
    with gr.Tabs():

        with gr.Tab("Quaterly Reports Expert"):
            gr.ChatInterface(
                fn=response_quaterly
            )

        with gr.Tab("My Portfolio Management"):
            gr.ChatInterface(
                fn=response_my_portfolio
            )

demo.launch()