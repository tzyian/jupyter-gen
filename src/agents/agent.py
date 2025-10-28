# main.py
import asyncio
import json
import logging
import time
from contextlib import AsyncExitStack
from typing import Dict, TypedDict, cast

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.tools import ToolException
from langchain_mcp_adapters.client import Connection, MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from config import settings
from prompts import (
    DEFAULT_NOTEBOOK_PATH,
    GENERATIVE_SYSTEM,
    RESEARCH_SYSTEM,
    SUPERVISOR_SYSTEM,
)
from utils.telemetry import callbacks_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Custom state: messages + loop metadata
class AppState(TypedDict, total=False):
    messages: list
    score: float
    notes: str
    rounds: int


MAX_REVISION_ROUNDS = 2


class MCPResources(TypedDict):
    research_client: MultiServerMCPClient
    generative_client: MultiServerMCPClient
    generative_session: object
    exit_stack: AsyncExitStack


async def build_graph() -> tuple[CompiledStateGraph, MCPResources]:
    OPENAI_API_KEY = settings.openai_api_key
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in environment variables.")

    # Setup MCP clients
    research_servers: Dict[str, Connection] = {
        "mcp2-search-tools": {
            "command": "uv",
            "args": ["run", "python", "-m", "src.tools.search"],
            "transport": "stdio",
        }
    }
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
    research_mcp = MultiServerMCPClient(research_servers)
    gen_mcp = MultiServerMCPClient(gen_servers)
    # Keep the Jupyter MCP session open across tool calls.
    exit_stack = AsyncExitStack()
    try:
        research_tools = await research_mcp.get_tools()
        gen_session = await exit_stack.enter_async_context(
            gen_mcp.session("Jupyter-MCP")
        )
        gen_tools = await load_mcp_tools(gen_session)
    except Exception:
        await exit_stack.aclose()
        raise

    resources: MCPResources = {
        "research_client": research_mcp,
        "generative_client": gen_mcp,
        "generative_session": gen_session,
        "exit_stack": exit_stack,
    }

    MODEL = "gpt-4o-mini"

    research_llm = ChatOpenAI(model=MODEL, api_key=OPENAI_API_KEY, temperature=0)
    gen_llm = ChatOpenAI(model=MODEL, api_key=OPENAI_API_KEY, temperature=0)
    critic_llm = ChatOpenAI(model=MODEL, api_key=OPENAI_API_KEY, temperature=0)

    # Create specialized agents
    research_agent = create_agent(
        research_llm, research_tools, system_prompt=RESEARCH_SYSTEM
    )
    gen_agent = create_agent(gen_llm, gen_tools, system_prompt=GENERATIVE_SYSTEM)
    gen_history: list = []
    notebook_initialized = False

    def _should_drop_instruction(message: object) -> bool:
        content = getattr(message, "content", None)
        role = getattr(message, "role", None) or getattr(message, "type", None)
        if isinstance(message, dict):
            content = message.get("content", content)
            role = message.get("role", role)
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text_parts.append(str(part.get("text", "")))
                else:
                    text_parts.append(str(part))
            content_text = "".join(text_parts)
        else:
            content_text = str(content) if content is not None else ""
        return role == "system" and content_text.startswith("Active notebook path: ")

    def _detect_use_notebook(messages: list) -> bool:
        for message in messages:
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls is None and isinstance(message, dict):
                tool_calls = message.get("tool_calls")
            if not tool_calls:
                continue
            for call in tool_calls:
                name = getattr(call, "name", None)
                if name is None and isinstance(call, dict):
                    name = call.get("name")
                if name == "use_notebook":
                    return True
        return False

    # Wrap agents as tools for supervisor
    @tool
    async def research_task(request: str) -> str:
        """Research information using arxiv, web search, and other research tools.

        Use this when you need to gather information, find papers, or research a topic.
        """
        result = await research_agent.ainvoke(
            {"messages": [{"role": "user", "content": request}]},
            config=callbacks_config(),
        )
        last_message = result["messages"][-1]
        return getattr(last_message, "content", str(last_message))

    @tool
    async def generate_notebook(request: str) -> str:
        """Generate and execute Jupyter notebook cells.

        Use this to create notebooks, write code, run cells, and work with Jupyter.
        """
        nonlocal gen_history, notebook_initialized
        try:
            messages = list(gen_history)
            if notebook_initialized:
                path_instruction = (
                    "Active notebook path: "
                    f"{DEFAULT_NOTEBOOK_PATH}. Include this path on all Jupyter tool calls. "
                    "Notebook session already active; continue working in the same notebook and do not call `use_notebook` again unless you are switching files."
                )
            else:
                path_instruction = (
                    "Active notebook path: "
                    f"{DEFAULT_NOTEBOOK_PATH}. Include this path on all Jupyter tool calls. "
                    "Call the `use_notebook` tool once (mode 'connect') before other notebook actions if it has not been run in this session."
                )
            messages.append({"role": "system", "content": path_instruction})
            messages.append({"role": "user", "content": request})
            result = await gen_agent.ainvoke(
                {"messages": messages},
                config=callbacks_config(),
            )
            conversation = result["messages"]
            gen_history = [
                msg for msg in conversation if not _should_drop_instruction(msg)
            ]
            if not notebook_initialized and _detect_use_notebook(conversation):
                notebook_initialized = True
            if conversation:
                last_message = conversation[-1]
                return getattr(last_message, "content", str(last_message))
            return ""
        except ToolException as exc:
            logger.warning("Notebook tool failed: %s", exc, exc_info=True)
            return (
                "Notebook tool failed to execute. "
                "Review the request, adjust parameters, and try again."
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Unexpected notebook tool error")
            return f"Unexpected notebook tool error: {exc}"

    # Create supervisor agent with sub-agents as tools
    supervisor_agent = create_agent(
        ChatOpenAI(model=MODEL, api_key=OPENAI_API_KEY, temperature=0),
        [research_task, generate_notebook],
        system_prompt=SUPERVISOR_SYSTEM,
    )

    # Custom nodes for critic loop
    async def supervisor_node(state: AppState) -> AppState:
        """Run the supervisor agent"""
        result = await supervisor_agent.ainvoke(
            {"messages": state.get("messages", [])}, config=callbacks_config()
        )
        # Preserve other state keys (rounds, score, notes) when returning.
        new_state = dict(state)
        new_state["messages"] = result["messages"]
        return cast(AppState, new_state)

    async def critic_node(state: AppState) -> AppState:
        """Evaluate the supervisor's output"""
        msgs = state.get("messages", [])
        if len(msgs) < 2:
            new_state = dict(state)
            new_state.setdefault("messages", msgs)
            new_state["score"] = 0.0
            new_state["notes"] = "No output to evaluate"
            new_state["rounds"] = state.get("rounds", 0)
            return cast(AppState, new_state)

        # Track how many revision cycles have executed so far.
        rounds = state.get("rounds", 0)

        # Get last assistant message
        last_msg = msgs[-1]
        content = getattr(last_msg, "content", "")

        critique_prompt = [
            {
                "role": "system",
                "content": 'Evaluate the response. Return JSON: {"score": 0..1, "notes": "feedback"}.',
            },
            {"role": "user", "content": f"Evaluate this response:\n{content}"},
        ]

        critique = await critic_llm.ainvoke(critique_prompt, config=callbacks_config())

        try:
            critique_content = getattr(critique, "content", "{}")
            if isinstance(critique_content, str):
                r = json.loads(critique_content)
            else:
                r = {}
            score = float(r.get("score", 0))
            notes = r.get("notes", "")
        except Exception as e:
            logger.warning(f"Failed to parse critic response: {e}")
            score = 0.0
            notes = "Invalid JSON from critic"
        # Preserve messages and other state keys when returning updated critique info.
        new_state = dict(state)
        new_state["score"] = score
        new_state["notes"] = notes
        new_state["rounds"] = rounds + 1
        new_state["messages"] = msgs
        return cast(AppState, new_state)

    # Routing functions
    def after_critic(state: AppState) -> str:
        """Decide whether to continue or finish based on critic score"""
        score = state.get("score", 0)
        rounds = state.get("rounds", 0)

        if score >= 0.8 or rounds >= MAX_REVISION_ROUNDS:
            return END

        # Add revision request to messages
        notes = state.get("notes", "No specific feedback")
        msgs = state.get("messages", [])
        msgs.append(
            {
                "role": "user",
                "content": f"Please revise based on this feedback:\n{notes}",
            }
        )

        return "supervisor"

    # Build the graph
    builder = StateGraph(AppState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("critic", critic_node)
    # Define flow
    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "critic")
    builder.add_conditional_edges(
        "critic", after_critic, {END: END, "supervisor": "supervisor"}
    )

    return builder.compile(), resources


async def main():
    try:
        out = await setup_and_run()
        messages = out.get("messages", [])
        if messages:
            final_msg = messages[-1]
            print(getattr(final_msg, "content", str(final_msg)))
        else:
            print(str(out))
    finally:
        logger.info("Agent execution completed")


async def setup_and_run(
    initial_messages: list[dict[str, str]] | None = None,
    initial_rounds: int = 0,
) -> AppState:
    graph, resources = await build_graph()
    try:
        payload = {
            "messages": initial_messages
            or [
                {"role": "system", "content": SUPERVISOR_SYSTEM},
                {
                    "role": "user",
                    "content": "Use Jupyter to introduce linear regression.",
                },
            ],
            "rounds": initial_rounds,
        }
        result = await graph.ainvoke(payload, config=callbacks_config())
        return cast(AppState, result)
    finally:
        await resources["exit_stack"].aclose()


if __name__ == "__main__":
    start_time = time.perf_counter()
    asyncio.run(main())
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time} seconds")
