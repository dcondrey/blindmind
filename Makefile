.PHONY: start install test init seed clean help

help:
	@echo "BlindMind Elite - Evolutionary Concept Refinement"
	@echo ""
	@echo "Usage:"
	@echo "  make start      Launch the interactive menu (Recommended)"
	@echo "  make install    Install dependencies using uv"
	@echo "  make test       Run the full test suite"
	@echo "  make clean      Remove virtual environment and database"

start:
	export PYTHONPATH=$$PYTHONPATH:. && ./.venv/bin/python -m blindmind.cli

install:
	uv sync

init:
	uv run blindmind init

test:
	./.venv/bin/python -m pytest -v

clean:
	rm -rf .venv .uv_cache data/blindmind.db data/blindmind.log .env
