from langchain.agents import create_agent
#Ollama agents not able to use tools, to figure out later.
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import create_retriever_tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import gradio as gr
import yaml

from dotenv import load_dotenv

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
    tools=[]
)

choice = input("So do you want to:\n10:")


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