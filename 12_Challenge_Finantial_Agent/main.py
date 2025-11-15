from ollama_agent import call_ollama
import gradio as gr


import yaml


with open("prompts.yaml", "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)

    prompt = prompts["HISTORIAN"]


def response (message, history):

    messages = history + [{"role": "user", "content": message}]

    response = call_ollama(prompt, messages)

    return response

with gr.Blocks() as demo:
    with gr.Tabs():

        with gr.Tab("Tab 1"):
            gr.ChatInterface(
                fn=response
            )

demo.launch()