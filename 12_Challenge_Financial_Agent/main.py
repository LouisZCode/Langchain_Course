from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

import gradio as gr
import yaml


THREAD_ID = "thread_001"

with open("prompts.yaml", "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)
    prompt = prompts["HISTORIAN"]

#print(prompt)

model = ChatOllama(
    model="gemma3:27b"
)

agent = create_agent(
    model=model,
    system_prompt=prompt,
    checkpointer=InMemorySaver()
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