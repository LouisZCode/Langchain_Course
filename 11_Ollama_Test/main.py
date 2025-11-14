from langchain_ollama import ChatOllama
from langchain.agents import create_agent


SYSTEM_PROMPT="Youa re a full stack comedian"

model = ChatOllama(
    model="gemma3:27b"
)

#Safe and normal great LLM installed locally: gemma3:27b
#have fun with the uncensored version of Gemma:   gemma-3-27b-it-abliterated-GGUF = (hf.co/mlabonne/gemma-3-27b-it-abliterated-GGUF:latest)

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT
)

query = input("Your message:\n")

result = agent.invoke({
    "role" : "user",
    "messages" : query
    }
)

#print(result)

for i,msg in enumerate(result["messages"]):
    msg.pretty_print()