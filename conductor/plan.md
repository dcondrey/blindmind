# Comprehensive Implementation Plan: BlindMind Evolutionary Algorithm

## 1. Objective
Develop a robust, Python-based CLI application that implements a "Blind Variation and Selective Retention" evolutionary algorithm. The system uses LLMs to crossover, mutate, and filter concepts across generations, creating a compounding fitness landscape of novel ideas.

## 2. Technical Stack & Tooling
- **Core Language:** Python 3.12+ (managed via `uv`).
- **CLI & UX:** `typer` for command routing, `rich` for live progress bars, panels, and syntax-highlighted terminal UI.
- **Validation & State:** `pydantic` v2 for JSON schema enforcement and data modeling; `pydantic-settings` for `.env` management.
- **LLM Orchestration:** `litellm` (async API wrapper), `tenacity` (retry logic & exponential backoff).
- **Data Persistence:** `sqlite3` + `sqlmodel` (SQLAlchemy + Pydantic ORM) for relational tracking of concepts and their evolutionary lineage.
- **Concurrency:** Native `asyncio` with bounded semaphores for rate-limited parallel LLM execution.
- **Testing:** `pytest`, `pytest-asyncio`, and `respx` (for mocking LLM HTTP responses) following a strict Test-Driven Development (TDD) approach.

## 3. Evolutionary Mechanics & Data Model

### 3.1. Database Schema (`src/models.py`)
- **`Concept` (Table):**
  - `id` (UUID, Primary Key)
  - `domain` (String): e.g., "Quantum Physics".
  - `title` (String): A punchy thesis title.
  - `description` (Text): The core mechanics of the concept.
  - `generation` (Int): 0 for seed, N for evolved.
  - `fitness_score` (Float, Nullable): Final composite score.
  - `created_at` (Datetime)
- **`Lineage` (Table):**
  - Tracks relationships. A concept can have 1 parent (Point Mutation) or N parents (Crossover).
  - `child_id` (UUID, Foreign Key)
  - `parent_id` (UUID, Foreign Key)
  - `mutation_type` (Enum: CROSSOVER, POINT_MUTATION, INVERSION)
- **`EvolutionRun` (Table):**
  - `id` (UUID), `status` (Enum: IN_PROGRESS, COMPLETED, FAILED).
  - Tracks current generation and active configuration.

### 3.2. Variation Engine (Mutation & Crossover)
- **Crossover (Combinatorial):** Samples 2-3 distant concepts. LLM prompt forces synthesis of disparate domains.
- **Point Mutation (Refinement):** Samples 1 concept. LLM prompt forces a lateral thinking shift or domain translation (e.g., applying a biology concept to economics).
- **Control:** System configurable mutation vs. crossover rates (e.g., 70% crossover, 30% point mutation).

### 3.3. Retention Engine (Fitness Function)
- **Stage 1: AI Critic (Zero-Shot Grading):** Low temperature (0.1). Analyzes mutation and outputs a strict JSON schema: `{"conceptual_novelty": int, "feasibility": int, "utility": int, "fatal_flaws": list[str], "rationale": str}`. 
- **Stage 2: Human Operator:** `rich` interactive layout. Shows parents -> mutation -> critic scores. Human inputs final binary (Keep/Discard) or scalar (1-5) fitness score.

---

## 4. TDD Task Breakdown & Success Criteria

### Phase 1: Foundation & Data Layer
**Task 1.1: Environment & Config Scaffolding**
- **Action:** Initialize `uv` project. Setup `src/config.py` using `pydantic-settings`.
- **Tests (`tests/test_config.py`):** Assert missing API keys raise `ValidationError`. Assert default values load correctly.
- **Success Criteria:** `pytest tests/test_config.py` passes.

**Task 1.2: Relational Schema & SQLite Engine**
- **Action:** Implement `src/models.py` (SQLModel classes) and `src/db.py` (Engine initialization, session management).
- **Tests (`tests/test_db.py`):** Use an in-memory SQLite DB fixture. Test creating concepts, linking lineages, and querying by generation.
- **Success Criteria:** 100% coverage on CRUD operations and Lineage relationship traversal.

### Phase 2: LLM Integration & Prompt Chains
**Task 2.1: Resilience & LLM Wrapper**
- **Action:** Implement `src/llm.py`. Wrap `litellm.acompletion` with `@retry` from `tenacity` (handling `RateLimitError` and `APIConnectionError`). Implement a bounded `asyncio.Semaphore` (e.g., max 5 concurrent calls).
- **Tests (`tests/test_llm.py`):** Use `respx` to mock 429 Too Many Requests and verify `tenacity` backoff works before succeeding.
- **Success Criteria:** Wrapper correctly handles transient network errors and enforces rate limits.

**Task 2.2: Variation Prompts & Output Parsing**
- **Action:** Implement `src/prompts.py` (Jinja2 templates or formatted strings for Crossover and Point Mutation). Create `Pydantic` schema `MutationOutput`.
- **Tests (`tests/test_variation.py`):** Mock a successful LLM JSON response and assert it parses perfectly into `MutationOutput`.
- **Success Criteria:** Pydantic strictly validates LLM JSON; raises specific errors for malformed output.

**Task 2.3: Critic Prompt & Scoring Engine**
- **Action:** Implement AI Critic prompt. Create `Pydantic` schema `CriticScore`.
- **Tests (`tests/test_critic.py`):** Provide mock "good" and "bad" JSON responses. Assert the engine correctly flags mutations below the composite threshold.
- **Success Criteria:** Critic parser reliably extracts numeric scores and rationale.

### Phase 3: The Generational Loop & Orchestration
**Task 3.1: State Machine & Evolution Engine**
- **Action:** Implement `src/engine.py`. Create the `run_generation` async function. It must:
  1. Query active seeds.
  2. Generate N mutation pairs.
  3. `asyncio.gather` the Variation Engine calls.
  4. `asyncio.gather` the Critic Engine calls on successful variations.
- **Tests (`tests/test_engine.py`):** Mock all LLM calls. Run a full 10-item generation loop. Assert that exactly 10 variations are requested, and only those passing the mock critic are returned.
- **Success Criteria:** The async pipeline executes concurrently without deadlocks and correctly filters results.

### Phase 4: CLI UX & Human-in-the-Loop
**Task 4.1: Database Seeding CLI**
- **Action:** Implement `src/cli.py` commands: `init`, `seed`, `seed-batch` (from JSON).
- **Success Criteria:** User can comfortably populate Generation 0 concepts via terminal.

**Task 4.2: Interactive Generational Run UI**
- **Action:** Implement `blindmind run --generations X --population-size Y`.
- **UX Details:**
  - Use `rich.progress.Progress` to show a live progress bar during the async LLM Variation and Critic phases (e.g., "[10/50] Mutating concepts...").
  - Use `rich.layout.Layout` and `rich.panel.Panel` to present surviving concepts to the user in a beautiful terminal dashboard (showing Lineage, Description, and Critic Scores).
  - Use `rich.prompt.Prompt` to capture human fitness feedback.
- **Success Criteria:** A flawless, crash-free terminal experience that gracefully handles user interrupts (`Ctrl+C`) and saves current progress to `EvolutionRun` state.

## 5. Deployment & Execution
- System will provide a `Makefile` or `uv` task runner scripts for standard operations (`make test`, `make run`).
- Standardize on `gpt-4o-mini` or `claude-3-5-haiku` for cost-effective Variation/Critic loops, configurable via `.env`.
