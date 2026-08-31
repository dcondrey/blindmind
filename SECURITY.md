# Security Policy

## Supported Versions

Security fixes are applied to `main`; there is no long-term support branch yet.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Open a private advisory via
[GitHub Security Advisories](https://github.com/dcondrey/blindmind/security/advisories/new),
or email **davidcondrey@me.com**.

Please include a description of the vulnerability and its impact, steps to reproduce (without real
secrets or credentials), and the affected version/commit. You can expect an initial response within a
few days; coordinated disclosure is appreciated — please give a reasonable window to ship a fix before
publishing details.

## Supply-chain security

- CodeQL and Dependency Review actions are pinned to full commit SHAs.
- **CodeQL**, **Dependency Review**, and **Dependabot** run in CI.
- LLM API keys are read from environment/`.env` only; never commit `.env` or logs containing keys.
