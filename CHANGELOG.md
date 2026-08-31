# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) +
[Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased]

### Fixed
- Survivors found mid-generation are now saved to the database as soon as they clear the critic threshold, instead of only at the end of `run_generation_cycle`. A per-generation timeout used to discard all work found before it fired.
- Serialize concurrent `AsyncSession` access in `EvolutionEngine` (crossover/tournament parent selection) behind a lock; SQLAlchemy's async session is not safe for concurrent coroutine use and `run_generation_cycle` fans candidate creation out via `asyncio.gather`.
- Raise the LLM subprocess timeout in `blindmind/llm.py`; the previous value was too tight for slower `claude-cli` calls.

## [0.1.0] - 2026-08-13

### Added
- Initial release: evolutionary concept refinement CLI (Blind Variation and Selective Retention) with SQLite-backed lineage tracking, LLM-driven crossover/mutation/critic stages, and human-in-the-loop scored retention.
