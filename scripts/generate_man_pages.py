"""Generate man pages for the `blindmind` CLI from its live Typer command tree.

Run via `make man` (see Makefile). Never hand-edit the generated .1 files under
docs/man/ -- they are derived output and will drift from the actual commands the
moment cli.py changes; regenerate instead.

click-man discovers commands via console_scripts entry points, expecting an actual
click.Command there. A typer.Typer() instance (blindmind.cli:app) is not itself a
click.Command, so we convert it with typer.main.get_command() and call click-man's
Python API (write_man_pages) directly rather than its CLI, which sidesteps both that
mismatch and an unrelated click-man 0.5.1 bug where its CLI crashes reading
entry_point.version under Python 3.12+/newer setuptools (AttributeError:
'EntryPoint' object has no attribute 'version').
"""

import sys
from pathlib import Path

import typer.main
from click_man.core import write_man_pages

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from blindmind.cli import app

    try:
        from importlib.metadata import version

        blindmind_version = version("blindmind")
    except Exception:
        blindmind_version = None

    target_dir = REPO_ROOT / "docs" / "man"
    target_dir.mkdir(parents=True, exist_ok=True)

    command = typer.main.get_command(app)
    write_man_pages("blindmind", command, target_dir=str(target_dir), version=blindmind_version)
    print(f"Wrote man pages to {target_dir}")


if __name__ == "__main__":
    main()
