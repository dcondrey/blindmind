import logging
import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_log = logging.getLogger("blindmind.config")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM Settings
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    mistral_api_key: str | None = Field(default=None, alias="MISTRAL_API_KEY")
    perplexity_api_key: str | None = Field(default=None, alias="PERPLEXITY_API_KEY")
    cohere_api_key: str | None = Field(default=None, alias="COHERE_API_KEY")

    litellm_model: str = Field(default="gpt-4o-mini")
    variation_temperature: float = Field(default=1.0)
    critic_temperature: float = Field(default=0.1)

    # Evolutionary Settings
    critic_threshold: float = Field(default=7.0)
    # This applies to real API-key providers (openai/anthropic/etc via litellm) only.
    # The claude-cli subprocess provider is always capped at concurrency=1 in llm.py
    # regardless of this setting (concurrent CLI subprocesses were observed to collide
    # and blow the 240s per-call timeout; see headless_evolve_rule30.py). 4 is a
    # moderate default: most API providers' free/starter tiers comfortably handle
    # 4-5 concurrent calls without hitting per-minute rate limits, while still leaving
    # headroom below tighter tiers. Raise it if your provider/tier supports more.
    max_concurrent_calls: int = Field(default=4)
    crossover_rate: float = Field(default=0.5)
    point_mutation_rate: float = Field(default=0.3)
    inversion_rate: float = Field(default=0.2)

    # Database Settings
    database_url: str = Field(default="sqlite+aiosqlite:///data/blindmind.db")

    # Logging
    log_level: str = Field(default="INFO")

    @property
    def sync_database_url(self) -> str:
        return self.database_url.replace("sqlite+aiosqlite://", "sqlite://")

    def save_local_env(self):
        """Save current keys to local .env file, preserving unrelated vars."""
        existing = {}
        if os.path.exists(".env"):
            with open(".env") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, val = line.partition("=")
                        existing[key.strip()] = val.strip()

        api_keys = {
            "OPENAI_API_KEY": self.openai_api_key,
            "ANTHROPIC_API_KEY": self.anthropic_api_key,
            "GEMINI_API_KEY": self.gemini_api_key,
            "OPENROUTER_API_KEY": self.openrouter_api_key,
            "GROQ_API_KEY": self.groq_api_key,
            "MISTRAL_API_KEY": self.mistral_api_key,
            "PERPLEXITY_API_KEY": self.perplexity_api_key,
            "COHERE_API_KEY": self.cohere_api_key,
        }
        for key, val in api_keys.items():
            if val:
                existing[key] = val
            elif key in existing and not val:
                pass  # keep existing value

        with open(".env", "w") as f:
            f.write("\n".join(f"{k}={v}" for k, v in existing.items()))
        os.chmod(".env", 0o600)

def discover_api_keys():
    """Aggressively search for API keys in environment and dotfiles."""
    s = Settings()
    if any([s.openai_api_key, s.anthropic_api_key, s.gemini_api_key, s.openrouter_api_key, s.groq_api_key, s.mistral_api_key]):
        return s

    home = Path.home()
    files_to_scan = [".env", ".zshrc", ".bashrc", ".bash_profile", ".profile"]

    key_map = {
        "OPENAI_API_KEY=": "openai_api_key",
        "ANTHROPIC_API_KEY=": "anthropic_api_key",
        "GEMINI_API_KEY=": "gemini_api_key",
        "OPENROUTER_API_KEY=": "openrouter_api_key",
        "GROQ_API_KEY=": "groq_api_key",
        "MISTRAL_API_KEY=": "mistral_api_key",
        "PERPLEXITY_API_KEY=": "perplexity_api_key",
        "COHERE_API_KEY=": "cohere_api_key",
    }

    for filename in files_to_scan:
        p = home / filename
        if p.exists():
            try:
                content = p.read_text(errors="ignore")
                for line in content.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("#") or stripped.startswith("//"):
                        continue
                    for prefix, attr in key_map.items():
                        if prefix in stripped:
                            val = stripped.split(prefix, 1)[1].strip("'\" \t")
                            if val:
                                setattr(s, attr, val)
                                break
            except Exception as e:
                _log.debug(f"Failed to scan {p} for API keys: {e}")
                continue

    return s

settings = discover_api_keys()
