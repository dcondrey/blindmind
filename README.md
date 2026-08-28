# BlindMind

_Evolutionary concept refinement: Blind Variation and Selective Retention (BVSR), driven by LLMs._

BlindMind is a command-line application that replicates the biological mechanism of "Blind Variation and Selective Retention" (BVSR) to generate and refine novel concepts using Large Language Models (LLMs).

## Core Architecture

1.  **Seed Database**: Relational storage (SQLite) tracking concepts, generations, and parent-child lineage.
2.  **Variation Engine**: Smashing disparate concepts together (Crossover) or laterally shifting a single idea (Mutation) using high-temperature LLM sampling.
3.  **Retention Engine**:
    *   **Stage 1 (AI Critic)**: Low-temperature analytical pass scoring novelty, feasibility, and utility.
    *   **Stage 2 (Human-in-the-Loop)**: Interactive scored retention (1-10) blended with AI critique scores.
4.  **Generational Loop**: Survivors of both filters become the seeds for the next generation, allowing ideas to compound and evolve.

## Getting Started

1.  **Install dependencies:**
    ```bash
    make install
    ```
2.  **Launch interactive menu:**
    ```bash
    make start
    ```
3.  **Guided Setup:** On first launch, the app will auto-discover API keys, initialize the database, and offer to load seed concepts.

## Commands

Every command runs through uv (`uv run blindmind …`); the bare name works too once
the venv is active.

| Command | Description |
|---------|-------------|
| `blindmind run` | Start an evolutionary run |
| `blindmind list` | List concepts with filtering |
| `blindmind search <query>` | Search by keyword, domain, fitness |
| `blindmind view <ID>` | Inspect a concept |
| `blindmind tree <ID>` | Trace ancestry lineage |
| `blindmind stats` | Latent space metrics dashboard |
| `blindmind export [file]` | Export all concepts to JSON |
| `blindmind import <file>` | Import concepts from JSON |
| `blindmind graph [file]` | Export lineage to GraphViz DOT |
| `blindmind delete <ID>` | Remove a concept |
| `blindmind settings` | View current configuration |

## Evolution Parameters

The `run` command supports CLI flags for tuning:

```bash
uv run blindmind run 3 10 --threshold 6.5 --temperature 1.2 --model gpt-4o
```

| Flag | Description | Default |
|------|-------------|---------|
| `--threshold` / `-t` | Minimum composite score to survive | 7.0 |
| `--temperature` / `-T` | LLM sampling temperature for mutations | 1.0 |
| `--model` / `-m` | Override LLM model | gpt-4o-mini |

## Scored Retention

Instead of binary accept/reject, rate each mutation 1-10 during the human-in-the-loop phase. Your score is blended with the AI critic's composite score to produce a final fitness value. Enter 0 to skip a concept.

## Tags & Search

Concepts can be tagged with comma-separated labels during seeding or retention. Search by keyword, domain, fitness range, or tags:

```bash
uv run blindmind search "quantum" --domain physics --min-fitness 7.0
```

## JSON Import/Export

Export your entire latent space for backup or sharing:

```bash
uv run blindmind export my_ideas.json
uv run blindmind import seed_concepts.json
```

Import accepts a JSON array of objects with `domain`, `title`, `description` fields, or a full export file.

## Ease of Use Features

- **Short IDs**: Use the first few characters of a UUID (e.g., `8a2f`)
- **Pre-flight Checks**: Automatic API key validation and seed verification
- **Graceful Interrupts**: Ctrl+C during evolution saves progress as CANCELLED
- **LLM Stats**: Token counts and latency shown after each run
- **File Logging**: Detailed logs with rotation in `data/blindmind.log`

## Developer Tooling

```
make install    Sync dependencies
make start      Launch interactive menu
make test       Run the test suite (36 tests)
make clean      Remove venv, cache, and database
```
