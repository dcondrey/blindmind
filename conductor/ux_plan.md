# Zero-Onboarding UX Upgrade Plan

## Objective
Transform `blindmind` from a standard sub-command CLI into an interactive, wizard-driven application. A user should be able to type `blindmind` (or `make start`) and be guided entirely via interactive prompts.

## Core Features to Implement

### 1. Interactive Setup Wizard (`src/setup.py` or within `src/cli.py`)
- **API Key Check & Auto-Discovery**: 
  - Before prompting, aggressively scan for `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.
  - Check current process environment (`os.environ`).
  - Check global dotfiles (`~/.env`, `~/.bashrc`, `~/.zshrc`, `~/.bash_profile`).
  - If found, automatically copy to the project's local `.env` and proceed silently.
  - If *not* found anywhere, *then* prompt the user to paste it. Write this to the `.env` file automatically.
- **Database Initialization Check**: Check if the database exists. If not, automatically run the initialization logic.
- **Seed Check**: Check if Generation 0 concepts exist. If not, offer to automatically load the default "Quick Seed" concepts.

### 2. Interactive Main Menu (`src/cli.py`)
- Change the default behavior of the `blindmind` command (when run without arguments) to launch an interactive menu.
- Use `rich.prompt` (or a library like `questionary` or `InquirerPy`, but we'll stick to `rich` or standard `typer` prompts to avoid extra dependencies if possible) to present options:
  1. 🧬 **Evolve** (Start a Run)
  2. 🌱 **Seed** (Add new concepts)
  3. 📜 **List** (View Latent Space)
  4. 🌳 **Lineage** (Trace a Concept)
  5. 🕸️ **Graph** (Export Network)
  6. ❌ **Exit**

### 3. Guided Prompts for Sub-actions
- When **Evolve** is selected, interactively ask: "How many generations?" and "Population size?" with smart defaults.
- When **Lineage** is selected, ask for the Short ID interactively.

## Implementation Steps
1. **Env Management:** Add a utility in `src/config.py` to write/update the `.env` file.
2. **CLI Refactor:** 
   - Add an `@app.callback(invoke_without_command=True)` in `src/cli.py` to act as the main entry point when no subcommands are provided.
   - Implement the `interactive_menu()` loop.
   - Extract the logic of existing commands (`run`, `list`, `seed`, `graph`, `tree`) into standalone async functions that can be called by both the interactive menu and the traditional CLI subcommands.
3. **Setup Flow:** Implement the `ensure_setup()` function that runs before the main menu, handling API keys and DB initialization.
4. **Update Makefile/README:** Add a simple `make start` command and update instructions to reflect the new "just run it" philosophy.

This refactor will completely remove the friction of manual configuration and command-line syntax. Shall we proceed?