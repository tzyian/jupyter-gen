# Repository Guidelines

## Project Structure & Module Organization
- Root contains project files: `pyproject.toml`, `prompts.py`, `features.md`, and notebooks (e.g., `Untitled1.ipynb`).
- Source code lives at the repo root today; prefer adding new modules under `src/` with an importable package name (e.g., `src/mcp2/`).
- Add tests under `tests/` mirroring the module path (e.g., `tests/test_prompts.py`).
- Keep large datasets and generated artifacts out of the repo; use `data/` (gitignored) if needed.

## Build, Test, and Development Commands
- Setup environment (uv): `uv sync` (installs from `pyproject.toml` / `uv.lock`).
- Run JupyterLab: `uv run jupyter lab` (target JupyterLab 4.49).
- Run a script/module: `uv run python -m module_name` (or `uv run python prompts.py`).
- Tests (pytest): `uv run pytest -q` (from repo root).
- Alternative without uv: create a venv and `pip install -e .`.

## Coding Style & Naming Conventions
- Imports should be at the top of the file
- Avoid `#noqa`, `#type ignore`, `#ignore`, `#pragma no cover`, `# pyright: ignore` or similar ignores unless absolutely necessary; prefer fixing lint/type issues.
- Avoid `typing.Any` where possible; prefer precise types.
- Don't use lazy imports, put all imports at the top of the file.
- Python: 4-space indentation, type hints for public functions, docstrings for modules/classes.
- Names: modules `snake_case.py`, classes `PascalCase`, functions/vars `snake_case`, constants `UPPER_SNAKE`.
- Formatting/linting: no enforced tools in-repo; prefer Black (line length 88) and Ruff. Keep imports deterministic.

## Testing Guidelines
- Framework: `pytest` with files named `test_*.py` in `tests/`.
- Aim for meaningful unit tests around prompt logic and any agents/utilities; prefer fast, hermetic tests.
- Run tests: `uv run pytest -q`; add `-k name` to filter. Consider `pytest-cov` for coverage if needed.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise subject (<= 72 chars), focused diffs.
- Prefer Conventional Commits when practical: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- PRs: include a clear description, linked issues, reproduction/validation notes, and screenshots or CLI output when relevant.

## Agent-Specific Instructions
- Keep changes minimal and localized; do not modify `.venv/` or generated files.
- Avoid network calls from notebooks unless explicitly approved. Respect the constraints in `prompts.py`.
- Target Python `>=3.13`. For new code, favor modular components under `src/` and add tests.
