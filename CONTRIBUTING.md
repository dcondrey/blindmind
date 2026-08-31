# Contributing to blindmind

Thanks for your interest in improving blindmind. This document describes how to
report issues, set up a development environment, and submit changes.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating, you are expected to uphold it.

## How to Contribute

### Reporting Issues

- Use the [issue templates](.github/ISSUE_TEMPLATE) for bugs and feature requests.
- **Do not** report security vulnerabilities in public issues — see
  [SECURITY.md](SECURITY.md).

### Development Setup

Requires **Python 3.11+** and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/dcondrey/blindmind
cd blindmind
make install
make test
```

### Making Changes

1. Create a topic branch off `main`.
2. Make focused, minimal changes; keep commits as single logical units.
3. Before opening a PR, the quality gate must be green:

   ```sh
   uv run ruff check .          # lint
   uv run ruff format --check . # formatting
   uv run pytest -v             # tests
   ```

4. Add a regression test for every fix or new behavior where feasible.

### Code Style

- Match the surrounding code's idiom, naming, and comment density.
- `uv run ruff format .` is the source of truth for formatting.
- Conventional commit subjects: `<type>: <description>` where
  `type ∈ fix | feat | refactor | test | docs | perf | security | chore`.

## Pull Request Process

- Fill out the [pull request template](.github/pull_request_template.md).
- Keep PRs scoped to one concern; link related issues.
- All CI checks must pass and at least one maintainer must approve.

## License and Contributor Agreement

blindmind is licensed under [Apache-2.0](LICENSE). By contributing, you agree that
your contributions are licensed under the same terms.

For questions about the contributor agreement, contact: davidcondrey@me.com
