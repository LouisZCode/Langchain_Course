from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
import asyncio

load_dotenv()

async def main():
    # Connect to MCP Server
    mcp_client = MultiServerMCPClient({
        "demo": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-everything"]  # ✅ Reliable
        }
    })
    
    # Load tools
    mcp_tools = await mcp_client.get_tools()
    print(f"Loaded {len(mcp_tools)} MCP tools: {[t.name for t in mcp_tools]}")
    
    # Create agent with the tools
    agent = create_agent(
        model = "openai:gpt-4",
        tools = mcp_tools,
        system_prompt = "You are a weather expert, and always check your tools to get a better response."
    )
    
    # Use the agent
    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": "What time is it in Tokyo?"}]
    })
    
    print(response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())