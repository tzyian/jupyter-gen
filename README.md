## Quickstart

- Install deps (uv): `uv sync`
- Configure `.env` from `.env.example` with your API keys
- Run server: `uv run python -m mcp2.cli run-server`
- Summarize a notebook: `uv run python -m mcp2.cli summarize path/to.ipynb`

Core MCP tools exposed:
- `summarize_notebook(notebook)`
- `next_edit_prediction(notebook, focus_cell_id)`
- `search_arxiv(query, surveys_only, max_results)`
- `search_tavily(query, surveys_only, max_results)`
- `transcribe_audio(audio_path, model)`
- `transcribe_audio_openai(audio_path, model)`
- `process_transcript(transcript, notebook?)`

Voice flow (STT only):
- Record in Jupyter UI, save to a temp file, call `transcribe_audio` (ElevenLabs scribe_v1).
- Pass the returned `text` to `process_transcript` (optionally include the active notebook JSON) to get next actions.

CLI examples:
- ElevenLabs STT: `uv run python -m mcp2.cli transcribe path/to.wav`
- OpenAI Whisper STT: `uv run python -m mcp2.cli transcribe-openai path/to.wav`
