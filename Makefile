.PHONY: start install test init seed clean man help

help:
	@echo "BlindMind Elite - Evolutionary Concept Refinement"
	@echo ""
	@echo "Usage:"
	@echo "  make start      Launch the interactive menu (Recommended)"
	@echo "  make install    Install dependencies using uv"
	@echo "  make test       Run the full test suite"
	@echo "  make man        Generate man pages (docs/man/) from the live CLI"
	@echo "  make clean      Remove virtual environment and database"

start:
	export PYTHONPATH=$$PYTHONPATH:. && ./.venv/bin/python -m blindmind.cli

install:
	uv sync

init:
	uv run blindmind init

test:
	./.venv/bin/python -m pytest -v

man:
	uv run python scripts/generate_man_pages.py

clean:
	rm -rf .venv .uv_cache data/blindmind.db data/blindmind.log .env
