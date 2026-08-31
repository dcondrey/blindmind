<h1 align="center">BlindMind</h1>

<p align="center">
  <strong>Evolutionary concept refinement — Blind Variation and Selective Retention (BVSR), driven by LLMs.</strong>
</p>

<p align="center">
  <a href="https://github.com/dcondrey/blindmind/actions/workflows/ci.yml"><img src="https://github.com/dcondrey/blindmind/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/dcondrey/blindmind/actions/workflows/codeql.yml"><img src="https://github.com/dcondrey/blindmind/actions/workflows/codeql.yml/badge.svg" alt="CodeQL"></a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/dcondrey/blindmind"><img src="https://api.securityscorecards.dev/projects/github.com/dcondrey/blindmind/badge" alt="OpenSSF Scorecard"></a>
  <a href="https://github.com/dcondrey/blindmind/blob/main/LICENSE"><img src="https://img.shields.io/github/license/dcondrey/blindmind" alt="license"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="python version">
  <a href="https://github.com/dcondrey/blindmind/issues"><img src="https://img.shields.io/github/issues/dcondrey/blindmind" alt="issues"></a>
</p>

<p align="center">
  <a href="#getting-started">Getting Started</a> &middot;
  <a href="#core-architecture">Architecture</a> &middot;
  <a href="#commands">Commands</a> &middot;
  <a href="#evolution-parameters">Parameters</a> &middot;
  <a href="#contributing">Contributing</a>
</p>

---

BlindMind is a command-line application that replicates the biological mechanism of "Blind Variation and Selective Retention" (BVSR) to generate and refine novel concepts using Large Language Models (LLMs).

## Core Architecture

1. **Seed Database**: Relational storage (SQLite, via SQLModel/SQLAlchemy async) tracking concepts, generations, and parent-child lineage — scoped per named `--project`.
2. **Variation Engine**: Smashing disparate concepts together (Crossover) or laterally shifting a single idea (Mutation, Wildcard) using high-temperature LLM sampling.
3. **Retention Engine**:
   - **Stage 1 (AI Critic)**: Low-temperature analytical pass scoring novelty, feasibility, and utility; survivors are persisted as soon as they clear the adaptive threshold, not just at the end of a generation.
   - **Stage 2 (Human-in-the-Loop)**: Interactive scored retention (1-10) blended with AI critique scores.
4. **Generational Loop**: Survivors of both filters become the seeds for the next generation, allowing ideas to compound and evolve. The evolutionary directive itself adapts each generation based on what scored well.

## Getting Started

Requires **Python 3.11+** and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/dcondrey/blindmind
cd blindmind
make install       # uv sync
make start         # launch interactive menu
```

On first launch, the app auto-discovers API keys, initializes the database, and offers to load seed concepts.

## Commands

Every command runs through uv (`uv run blindmind …`); the bare name works too once
the venv is active. Most commands accept `--project`/`-p` to scope to a named project (default: `default`).

| Command | Description |
|---------|-------------|
| `blindmind init` | Initialize the database |
| `blindmind run` | Start an evolutionary run |
| `blindmind list` | List concepts with filtering |
| `blindmind search <query>` | Search by keyword, domain, fitness |
| `blindmind view <ID>` | Inspect a concept |
| `blindmind tree <ID>` | Trace ancestry lineage |
| `blindmind stats` | Latent space metrics dashboard |
| `blindmind projects` | List all projects with summary stats |
| `blindmind export [file]` | Export all concepts to JSON |
| `blindmind import <file>` | Import concepts from JSON |
| `blindmind export-v1 <file> -p <project>` | Export via the `crosstalk.blindmind.v1` contract |
| `blindmind graph [file]` | Export lineage to GraphViz DOT |
| `blindmind delete <ID>` | Remove a concept |
| `blindmind settings` | View current configuration |

## Evolution Parameters

The `run` command supports CLI flags for tuning:

```bash
uv run blindmind run 3 10 --threshold 6.5 --temperature 1.2 --model gpt-4o --project myidea
```

| Flag | Description | Default |
|------|-------------|---------|
| `--threshold` / `-t` | Minimum composite score to survive | 7.0 |
| `--temperature` / `-T` | LLM sampling temperature for mutations | 1.0 |
| `--model` / `-m` | Override LLM model | gpt-4o-mini |
| `--project` / `-p` | Named project to scope this run to | `default` |

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
make test       Run the test suite
make clean      Remove venv, cache, and database
```

```bash
uv run ruff check .           # lint
uv run ruff format --check .  # formatting
uv run pytest -v              # tests
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and the PR process. Please read our [Code of Conduct](CODE_OF_CONDUCT.md). Security issues should be reported per [SECURITY.md](SECURITY.md), not as public issues.

## License

[Apache-2.0](LICENSE)
