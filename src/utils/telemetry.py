from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langfuse import get_client
from langfuse.langchain import CallbackHandler

# Load environment variables from .env file
load_dotenv()


langfuse = get_client()


def callbacks_config() -> RunnableConfig:
    """Return a LangChain invoke config with Langfuse callbacks if available."""
    handler = CallbackHandler()

    if handler is None:
        return {}
    return {"callbacks": [handler]}
