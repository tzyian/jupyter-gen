```
# Project Outline

## 1. Core Features
- The first cell when created should automatically be the following in the markdown cell:
"""
<img src="images/logo/selene-logo-640.png" style="max-height:75px;" alt="SELENE Logo" />

**Disclaimer:** This Jupyter Notebook contains content generated with the assistance of AI. While every effort has been made to review and validate the outputs, users should independently verify critical information before relying on it. The SELENE notebook repository is constantly evolving. We recommend downloading or pulling the latest version of this notebook from Github.
"""

- If the user requests a new notebook to be generated but without stating what topic it wants, it should:
  - RAG over existing notebooks to locate related `src/` code
  - RAG over existing jupyter notebook topics to see what has already been done
  - Suggest a few notebook ideas based on gaps in the existing notebooks and codebase

- These notebook ideas should be inserted into the second cell cell as markdown
- If the user already knows what they want to generate, instead: do research on that topic. 
- The second cell should then be a markdown cell containing the research findings and relevant URLs

- Then, the agent should generate an outline for the notebook, get user feedback, and then generate the actual code cells
- The third cell should contain a brief description of the notebook's purpose and structure.

- Use jupyter-mcp-server to interact with the notebook 
- Use supervisor, research, generative, and reflective agent roles
- support voice commands using ElevenLabs STT, transcribe to words, then get an agent to do its task


- Package everything together in a jupyterlab extension for easy installation
- including jupyterlab 4.49, python LSP server, and optional AI autocomplete


- Next Edit Prediction
- ElevenLabs voice controls

- Notebook summarizer with mindmap/Table of Contents generation
- Generate an outline, get feedback, and then the actual code
- If possible, integrate with Inkscape MCP as well

## 2. Research Integrations
- Tavily tool-calls for literature search
- ArXiv retrieval (via LangGraph retriever)

## 3. Agent Roles
- Supervisor (routing/coordination)
- Research (Tavily + ArXiv external knowledge)
- Generative (memory + planning)
- Reflective (meta-learning from errors)

## 4. Notebook Utilities
- RAG over existing notebooks to locate related `src/` code
- Detect duplicate/overlapping notebooks in the workspace
- Summarize notebooks and surface key entities/links

## 5. Environment & DX
- JupyterLab 4.49
- Python LSP server for completion and diagnostics
- Optional AI autocomplete
```

Core Feature Areas
- Next Edit Prediction: proactively suggest the user’s likely next code change.
- ElevenLabs voice controls: voice input/output for hands-free interactions.
- Notebook summarizer: generate concise summaries and a mindmap/ToC per notebook.

Research Integrations
- Tavily tool-calls to search research papers, emphasizing survey/meta-analysis papers.
- ArXiv retriever in LangGraph for targeted literature RAG.

Agent Roles
- Supervisor: routes requests and coordinates sub-agents.
- Research: queries Tavily/ArXiv and compiles findings.
- Generative: plans tasks, maintains working memory, and produces drafts.
- Reflective: reviews errors/outcomes and updates strategies.

Notebook Utilities
- RAG notebooks to find whether referenced Python code already exists in `src/`.
- Detect duplicates or near-duplicates among notebooks in the folder.
- Summarize notebooks and surface cross-references or missing links.

Environment & Developer Experience
- Target: JupyterLab 4.49.
- Bundle Python LSP server for autocompletion and error checking.
- Include an AI autocomplete option if feasible.

Suggested Research and Readings
- Agents and orchestration
  - Search queries: "agentic workflows survey", "LLM multi-agent systems survey", "supervisor agent coordination"
- RAG and retrieval
  - Search queries: "retrieval augmented generation best practices", "RAG for notebooks", "arxiv retriever langgraph"
- Planning and reflection
  - Search queries: "LLM planning with memory", "reflective agents meta-learning", "error-driven self-reflection LLM"
- Next edit prediction
  - Search queries: "next edit prediction code", "code change prediction LLM", "intent-aware code suggestion"
- Voice interfaces
  - Search queries: "LLM voice control pipeline", "text-to-speech ElevenLabs integration", "speech-to-text notebook workflow"
- Notebook summarization
  - Search queries: "notebook summarization arxiv", "mindmap generation from documents", "entity extraction notebooks"

Immediate Next Steps
- Define minimal LangGraph graph: supervisor, research, generative, reflective nodes.
- Add ArXiv retriever + Tavily tools and test end-to-end query flow.
- Implement notebook RAG index and duplicate detection for the workspace.
- Prototype Next Edit Prediction as a background suggestion service.
- Integrate Python LSP and confirm JupyterLab 4.49 setup.
