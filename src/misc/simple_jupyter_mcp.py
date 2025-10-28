import asyncio
import json
import logging
from typing import Dict, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import ToolException
from langchain_mcp_adapters.client import Connection, MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from config import settings
from prompts import GENERATIVE_SYSTEM
from utils.telemetry import callbacks_config

gen_servers: Dict[str, Connection] = {
    "jupyter": {
        "command": "uvx",
        "args": ["jupyter-mcp-server@latest"],
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
    gen_llm = ChatOpenAI(model=MODEL, api_key=settings.openai_api_key, temperature=0.1)
    echo_llm = ChatOpenAI(model=MODEL, api_key=settings.openai_api_key, temperature=0.1)

    class AgentState(TypedDict):
        input: str
        intermediate: str
        turn: int

    async with client.session("Jupyter-MCP") as session:
        tools = await load_mcp_tools(session)
        lc_agent = create_agent(gen_llm, tools=tools, system_prompt=GENERATIVE_SYSTEM)

        async def gen_agent(state: AgentState) -> AgentState:
            logger.info("Gen agent received: %s", state)
            messages = [
                SystemMessage(content=GENERATIVE_SYSTEM),
                HumanMessage(content=state["input"]),
            ]
            try:
                result = await lc_agent.ainvoke(
                    {"messages": messages}, config=callbacks_config()
                )
                last_msg = result["messages"][-1]
                content = getattr(last_msg, "content", str(last_msg))
            except ToolException as exc:
                logger.warning("Generative agent tool error: %s", exc)
                content = f"Tool error: {exc}"
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Generative agent unexpected error")
                content = f"Unexpected error: {exc}"

            return {
                "input": state["input"],
                "intermediate": state["intermediate"] + f" | gen: {content}",
                "turn": state["turn"] + 1,
            }

        async def echo_agent(state: AgentState) -> AgentState:
            logger.info("Echo agent received: %s", state)
            echo_response = await echo_llm.ainvoke(
                [
                    SystemMessage(content="You are an echo bot."),
                    HumanMessage(content=state["intermediate"] or state["input"]),
                ],
                config=callbacks_config(),
            )
            echo_content = getattr(echo_response, "content", echo_response)
            return {
                "input": state["input"],
                "intermediate": state["intermediate"] + f" | echo: {echo_content}",
                "turn": state["turn"] + 1,
            }

        def next_node(state: AgentState) -> str:
            if state["turn"] >= 3:
                return END
            return "gen_agent" if state["turn"] % 2 == 0 else "echo_agent"

        graph = StateGraph(AgentState)
        graph.add_node("gen_agent", gen_agent)
        graph.add_node("echo_agent", echo_agent)
        graph.add_edge(START, "gen_agent")
        graph.add_conditional_edges(
            "gen_agent",
            next_node,
            {"gen_agent": "gen_agent", "echo_agent": "echo_agent", END: END},
        )
        graph.add_conditional_edges(
            "echo_agent",
            next_node,
            {"gen_agent": "gen_agent", "echo_agent": "echo_agent", END: END},
        )
        compiled = graph.compile()

        # Example input for the agent/graph
        input_data: AgentState = {
            "input": "Write a function calculating fibonacci then execute it for varying fibonacci numbers",
            "intermediate": "",
            "turn": 0,
        }
        logger.info("Running agent graph with input: %s", input_data)
        result = await compiled.ainvoke(input_data, config=callbacks_config())
        logger.info("Result: %s", result)
        print(json.dumps(result, indent=2))


# Entry point for running the script
if __name__ == "__main__":
    print("Starting simple")
    asyncio.run(main())
    print("Ended simple")
