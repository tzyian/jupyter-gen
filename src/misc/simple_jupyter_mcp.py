import asyncio
import json
import logging
from typing import Dict, TypedDict

from langchain.agents import create_agent
from langchain_mcp_adapters.client import Connection, MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from config import settings
from prompts import GENERATIVE_SYSTEM
from utils.telemetry import callbacks_config


gen_servers: Dict[str, Connection] = {
    # "jupyter": {
    #     "command": "uvx",
    #     "args": ["jupyter-mcp-server@latest"],
    #     "transport": "stdio",
    #     "env": {
    #         "JUPYTER_URL": "http://localhost:8888",
    #         "JUPYTER_TOKEN": "MY_TOKEN",
    #         "ALLOW_IMG_OUTPUT": "true",
    #     },
    # },
    "Jupyter-MCP": {
        "command": "uvx",
        "args": [
            "--from",
            # "your/path/to/jupyter-mcp-server/dist/jupyter_mcp_server-x.x.x-py3-none-any.whl",
            "C:/Users/Ian/Documents/GitHub/fyp/jupyter-mcp-server-modified/dist/jupyter_mcp_server-0.18.1-py3-none-any.whl",
            "jupyter-mcp-server",
        ],
        "transport": "stdio",
        "env": {
            "JUPYTER_URL": "http://localhost:8888",
            "JUPYTER_TOKEN": "MY_TOKEN",
            "ALLOW_IMG_OUTPUT": "true",
        },
    },
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    client = MultiServerMCPClient(gen_servers)
    MODEL = "gpt-4o-mini"
    gen_llm = ChatOpenAI(model=MODEL, api_key=settings.openai_api_key, temperature=0)
    
    class AgentState(TypedDict):
        input: str

    async with client.session("Jupyter-MCP") as session:
        tools = await load_mcp_tools(session)
        gen_agent = create_agent(gen_llm, tools=tools, system_prompt=GENERATIVE_SYSTEM)



        graph = StateGraph(AgentState)
        graph.add_node("agent", gen_agent)
        graph.add_edge(START, "agent")
        graph.add_edge("agent", END)
        compiled = graph.compile()

        # Example input for the agent/graph
        input_data: AgentState = {"input": "What is 2 + 2?"}
        logger.info("Running agent graph with input: %s", input_data)
        result = await compiled.ainvoke(input_data, config=callbacks_config())
        logger.info("Result: %s", result)
        print(json.dumps(result, indent=2))


# Entry point for running the script
if __name__ == "__main__":
    print("Starting simple")
    asyncio.run(main())
    print("Ended simple")
