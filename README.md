## Quickstart
- Install deps (uv): `uv sync`
- Configure `.env` from `.env.example` with your OpenAI and Tavily API keys
- Run the following manually (required by [jupyter-mcp-server](https://github.com/datalayer/jupyter-mcp-server))
  - `uv pip uninstall pycrdt datalayer_pycrdt`
  - `uv pip install datalayer_pycrdt==0.12.17`
- Run Jupyter lab `jupyter lab --port 8888 --IdentityProvider.token MY_TOKEN --ip 0.0.0.0`
- `python src/agents/agent.py`

This project uses the [arXiv API](https://info.arxiv.org/help/api/index.html): Thank you to arXiv for use of its open access interoperability.

