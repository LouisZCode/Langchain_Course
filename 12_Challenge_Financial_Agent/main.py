from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import create_retriever_tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import gradio as gr
import yaml


THREAD_ID = "thread_001"

with open("prompts.yaml", "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)
    prompt = prompts["QUATERLY_RESULTS_EXPERT"]

#print(prompt)

model = ChatOllama(
    model="gemma3:27b"
)

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
agent = create_agent(
    model=model,
    system_prompt=prompt,
    checkpointer=InMemorySaver(),
    tools=[retriever_tool]
)



##Gradio-ing
def response(message, history):

    response = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        {"configurable": {"thread_id": THREAD_ID}}
    )
    return response["messages"][-1].content

with gr.Blocks() as demo:
    with gr.Tabs():

        with gr.Tab("Tab 1"):
            gr.ChatInterface(
                fn=response
            )

demo.launch()