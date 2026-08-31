import asyncio
import json
import os
import shutil
import time
from typing import TYPE_CHECKING, TypeVar

from litellm import acompletion, exceptions
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from blindmind.config import settings
from blindmind.logging import logger

if TYPE_CHECKING:
    from blindmind.llm_schemas import CriticScore, MutationOutput

CLAUDE_CLI_LABEL = "Claude (subscription)"

T = TypeVar("T", bound=BaseModel)


class LLMStats:
    def __init__(self):
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_latency_ms = 0

    def record(self, input_tokens: int = 0, output_tokens: int = 0, latency_ms: float = 0):
        self.total_calls += 1
        self.successful_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_latency_ms += latency_ms

    def record_failure(self):
        self.total_calls += 1
        self.failed_calls += 1

    @property
    def summary(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "successful": self.successful_calls,
            "failed": self.failed_calls,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "avg_latency_ms": round(self.total_latency_ms / max(self.successful_calls, 1)),
        }


FATAL_ERROR_KEYWORDS = ("credit", "balance", "budget", "quota", "not found")


class ClaudeCLIOutputError(RuntimeError):
    """The `claude` CLI exited 0 but its stdout could not be turned into structured data.

    Deliberately carries a fixed, keyword-free message: _completion() classifies a
    failure as permanent by substring-matching str(e) against FATAL_ERROR_KEYWORDS,
    so raw model prose must never reach the exception text -- a response that happened
    to contain "budget" or "not found" would blacklist the provider for the whole
    session, and claude-cli is the only provider in headless runs. The raw text,
    stderr and exit code go to the log instead, where they are diagnostic but inert.
    """


def _extract_structured_output(payload: dict) -> dict:
    """Pull the schema-conforming object out of a `claude --output-format json` payload.

    The CLI delivers structured output via a tool call (stop_reason "tool_use") in
    payload["structured_output"]. When the model ends its turn with prose instead
    (stop_reason "end_turn") that key is null and payload["result"] holds plain text.
    The previous code fed that text straight to json.loads(), which rejected it with a
    bare "Expecting value: line 1 column 1 (char 0)" -- the uninformative error seen in
    production headless runs. Reproduced against claude CLI 2.1.251: subtype "success",
    is_error false, exit code 0, empty stderr, so no other check catches it.
    """
    if payload.get("is_error"):
        logger.warning(f"claude CLI error result: {str(payload.get('result'))[:500]}")
        # Plain RuntimeError, not ClaudeCLIOutputError: is_error is the CLI reporting a
        # failure it already knows about (usage cap, upstream API error), so re-asking
        # it is wasted wall clock. Kept keyword-free anyway so the raw text can't reach
        # _completion()'s FATAL_ERROR_KEYWORDS blacklist match.
        raise RuntimeError(f"claude CLI reported an error result (subtype={payload.get('subtype')!r})")

    structured = payload.get("structured_output")
    if structured is not None:
        return structured

    context = f"stop_reason={payload.get('stop_reason')!r} subtype={payload.get('subtype')!r}"
    result = payload.get("result")
    if not isinstance(result, str) or not result.strip():
        raise ClaudeCLIOutputError(f"claude CLI returned no structured output and an empty result field ({context})")

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        logger.warning(f"claude CLI returned unstructured prose instead of schema output ({context}): {result[:500]}")
        raise ClaudeCLIOutputError(
            f"claude CLI returned unstructured prose instead of schema output ({context}); see log for the text"
        ) from None


def _resolve_key_for_override(model: str) -> str | None:
    """Guess which configured API key an explicit --model override needs, by provider prefix."""
    if model == "claude-cli":
        return None
    if model.startswith("openrouter/"):
        return settings.openrouter_api_key
    if model.startswith("anthropic/"):
        return settings.anthropic_api_key
    if model.startswith("gemini/"):
        return settings.gemini_api_key
    if model.startswith("groq/"):
        return settings.groq_api_key
    if model.startswith("mistral/"):
        return settings.mistral_api_key
    return settings.openai_api_key


class LLMEngine:
    def __init__(self):
        # Provider-aware concurrency: real API providers (openai/anthropic/etc via
        # litellm) share settings.max_concurrent_calls. The claude-cli subprocess
        # provider gets its own semaphore fixed at 1 -- it spawns a real subprocess
        # per call with up to a 240s timeout, and concurrent subprocess invocations
        # were observed to collide and blow that timeout (see the comment in
        # headless_evolve_rule30.py, which hit this in a live run before pinning
        # max_concurrent_calls=1 globally). Splitting the semaphores means that
        # workaround is no longer necessary for correctness (it's still harmless if
        # left in place) and a provider pool that mixes claude-cli with API
        # providers no longer serializes the API calls unnecessarily.
        self.api_semaphore = asyncio.Semaphore(settings.max_concurrent_calls)
        self.claude_cli_semaphore = asyncio.Semaphore(1)
        self.providers = []
        self._blacklisted = set()
        self._explicit_override = None
        self._initialized = False
        self.stats = LLMStats()

    def set_model_override(self, model: str | None):
        """Force a specific model (e.g. from --model) to the front of the provider pool.
        Rebuilds the pool on the next call so the override actually takes effect."""
        self._explicit_override = model
        self._initialized = False
        self._blacklisted = set()

    async def _lazy_init(self):
        if self._initialized:
            return
        self.providers = await self._get_available_providers()
        self._initialized = True

    async def _get_available_providers(self) -> list[dict]:
        pool = []

        if self._explicit_override:
            pool.append(
                {
                    "model": self._explicit_override,
                    "api_key": _resolve_key_for_override(self._explicit_override),
                    "label": f"Override ({self._explicit_override})",
                }
            )

        # Quality-first: strongest/most-trusted providers before cheaper fallbacks.
        if shutil.which("claude"):
            pool.append({"model": "claude-cli", "api_key": None, "label": CLAUDE_CLI_LABEL})
        if settings.anthropic_api_key:
            pool.append(
                {
                    "model": "anthropic/claude-3-5-haiku-20241022",
                    "api_key": settings.anthropic_api_key,
                    "label": "Anthropic",
                }
            )
        if settings.openai_api_key:
            pool.append({"model": "gpt-4o-mini", "api_key": settings.openai_api_key, "label": "OpenAI"})
        if settings.groq_api_key:
            pool.append(
                {
                    "model": "groq/llama-3.3-70b-versatile",
                    "api_key": settings.groq_api_key,
                    "label": "Groq",
                }
            )
        if settings.gemini_api_key:
            pool.append({"model": "gemini/gemini-1.5-flash", "api_key": settings.gemini_api_key, "label": "Gemini"})
        if settings.mistral_api_key:
            pool.append(
                {
                    "model": "mistral/mistral-small-latest",
                    "api_key": settings.mistral_api_key,
                    "label": "Mistral",
                }
            )

        if settings.openrouter_api_key:
            pool.append(
                {
                    "model": "openrouter/openai/gpt-4o-mini",
                    "api_key": settings.openrouter_api_key,
                    "label": "OpenRouter (Paid)",
                }
            )

            # Free-tier models are last-resort only: unranked for quality, so they never
            # outrank a paid/subscription provider. Appended at the end of the pool.
            try:
                import httpx

                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get("https://openrouter.ai/api/v1/models")
                    if resp.status_code == 200:
                        or_models = resp.json().get("data", [])
                        free_models = [m for m in or_models if m.get("pricing", {}).get("prompt") == "0"]
                        for fm in free_models[:5]:
                            pool.append(
                                {
                                    "model": f"openrouter/{fm['id']}",
                                    "api_key": settings.openrouter_api_key,
                                    "label": f"OpenRouter (Free: {fm['id']})",
                                }
                            )
            except Exception as e:
                logger.warning(f"Could not fetch dynamic OpenRouter free-tier models: {e}. Skipping free fallback.")

        return pool

    async def _completion(self, messages: list[dict], temperature: float, response_format: type[T]) -> T:
        await self._lazy_init()
        if not self.providers:
            raise RuntimeError("No API keys found for any supported provider.")

        order = [i for i in range(len(self.providers)) if i not in self._blacklisted] or list(
            range(len(self.providers))
        )

        last_exception = None
        for idx in order:
            provider = self.providers[idx]
            try:
                return await self._try_provider(provider, messages, temperature, response_format)
            except Exception as e:
                err_str = str(e).lower()
                err_desc = str(e) or f"{type(e).__name__} (no message)"
                is_validation_error = "validation error" in err_str or "pydantic" in err_str
                is_timeout = isinstance(e, (asyncio.TimeoutError, exceptions.Timeout))
                # Only genuinely permanent failures (bad creds, missing model/quota) blacklist
                # a provider for the rest of the session. Everything else (rate limits,
                # timeouts, one-off validation misses) falls back for THIS call only, so the
                # next call still starts from the best provider instead of staying downgraded.
                is_permanent = isinstance(e, (exceptions.AuthenticationError, exceptions.NotFoundError)) or any(
                    word in err_str for word in FATAL_ERROR_KEYWORDS
                )

                self.stats.record_failure()
                last_exception = e
                if is_permanent:
                    logger.warning(
                        f"Provider {provider['label']} permanently unavailable this session ({err_desc}); blacklisting."
                    )
                    self._blacklisted.add(idx)
                else:
                    reason = (
                        "low quality response"
                        if is_validation_error
                        else ("timeout" if is_timeout else "transient error")
                    )
                    logger.warning(
                        f"Provider {provider['label']} failed ({reason}): {err_desc}; falling back for this call."
                    )
                continue

        last_desc = str(last_exception) or (
            f"{type(last_exception).__name__} (no message)" if last_exception else "unknown"
        )
        raise RuntimeError(f"All LLM providers failed. Last error: {last_desc}")

    # litellm.exceptions.RateLimitError is NOT a subclass of litellm.exceptions.APIError
    # (they're siblings mirroring openai's own hierarchy: both derive from
    # openai.APIStatusError, but litellm.APIError only wraps openai.APIError directly).
    # It was previously uncovered here, so a rate limit propagated straight to
    # _completion()'s per-provider fallback loop with no backoff at all -- which, with
    # a single active provider, meant hammering the same rate limit again on the very
    # next candidate. Retrying it here first (same provider, with backoff) is the right
    # layer for a typically-short rate-limit window; _completion()'s fallback loop is
    # the second-tier safety net if it's still failing after these retries are
    # exhausted, so this isn't double-handling the same failure at the same layer.
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=20),
        # ClaudeCLIOutputError is retried here for the same reason as a rate limit:
        # it is nondeterministic, not a bug in the request. The same directive prompt
        # is reused across a whole generation and ~3% of calls under it (6/207 in one
        # observed run) came back as prose while the rest returned schema output, so
        # the model's choice to answer in prose instead of calling the structured-output
        # tool clears on a re-ask. Without this the candidate is lost outright, because
        # headless runs have claude-cli as their only provider and _completion()'s
        # cross-provider fallback has nowhere to fall back to.
        retry=retry_if_exception_type(
            (
                exceptions.ServiceUnavailableError,
                exceptions.APIError,
                exceptions.RateLimitError,
                ClaudeCLIOutputError,
            )
        ),
    )
    async def _try_provider(
        self, provider: dict, messages: list[dict], temperature: float, response_format: type[T]
    ) -> T:
        if provider["model"] == "claude-cli":
            logger.debug(
                f"claude-cli provider has no temperature flag; requested temperature={temperature} will be ignored."
            )
            return await self._try_claude_cli(messages, response_format)
        async with self.api_semaphore:
            start = time.monotonic()
            response = await asyncio.wait_for(
                acompletion(
                    model=provider["model"],
                    api_key=provider["api_key"],
                    messages=messages,
                    temperature=temperature,
                    response_format=response_format,
                ),
                timeout=60,
            )
            elapsed_ms = (time.monotonic() - start) * 1000

            content = response.choices[0].message.content
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
            output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
            self.stats.record(input_tokens=input_tokens, output_tokens=output_tokens, latency_ms=elapsed_ms)
            logger.debug(f"LLM call to {provider['label']}: {input_tokens}in/{output_tokens}out in {elapsed_ms:.0f}ms")

            return response_format.model_validate_json(content)

    async def _try_claude_cli(self, messages: list[dict], response_format: type[T]) -> T:
        """Structured generation via the local Claude Code CLI, authenticated by
        subscription (not per-token API billing). Strips ANTHROPIC_API_KEY from the
        subprocess env so the CLI falls back to its keychain/OAuth subscription auth
        instead of billing the (possibly empty) API key.

        Note: the `claude` CLI has no --temperature flag, so settings.variation_temperature
        / settings.critic_temperature have no effect when this provider is active."""
        prompt = "\n\n".join(m["content"] for m in messages)
        schema = response_format.model_json_schema()
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

        async with self.claude_cli_semaphore:
            start = time.monotonic()
            proc = await asyncio.create_subprocess_exec(
                "claude",
                "-p",
                prompt,
                "--output-format",
                "json",
                "--json-schema",
                json.dumps(schema),
                "--model",
                "claude-haiku-4-5-20251001",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                # 240s, not 120s: a long directive (register digest, forbidden-domain
                # lists) pushes extended-thinking haiku calls past 120s on a large
                # fraction of calls even with no concurrency, observed on the rule30
                # project (0/1 succeeded at 120s serially; a bare call with the same
                # prompt took 73s but varied well past that under real load).
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=240)
            except TimeoutError:
                proc.kill()
                await proc.wait()
                raise
            elapsed_ms = (time.monotonic() - start) * 1000

            if proc.returncode != 0:
                raise RuntimeError(f"claude CLI exited {proc.returncode}: {stderr.decode(errors='replace')[:500]}")

            raw = stdout.decode(errors="replace")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                # Exit code 0 with unparseable stdout. Log everything that could
                # explain it -- previously this surfaced as a bare JSONDecodeError
                # indistinguishable from the prose case handled below.
                logger.warning(
                    f"claude CLI exited 0 but stdout was not JSON "
                    f"({len(raw)} chars): {raw[:500]!r}; stderr: {stderr.decode(errors='replace')[:500]!r}"
                )
                raise ClaudeCLIOutputError(
                    f"claude CLI exited 0 but stdout was not JSON ({len(raw)} chars); see log for stdout/stderr"
                ) from None

            structured = _extract_structured_output(payload)

            usage = payload.get("usage", {})
            self.stats.record(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                latency_ms=elapsed_ms,
            )
            logger.debug(
                f"LLM call to {CLAUDE_CLI_LABEL}: {elapsed_ms:.0f}ms, cost~${payload.get('total_cost_usd', 0):.4f}"
            )

            return response_format.model_validate(structured)

    async def generate_mutation(self, prompt: str, temperature: float = None) -> "MutationOutput":
        from blindmind.llm_schemas import MutationOutput

        temp = temperature if temperature is not None else settings.variation_temperature
        messages = [{"role": "user", "content": prompt}]
        return await self._completion(messages, temp, MutationOutput)

    async def critique_mutation(self, prompt: str) -> "CriticScore":
        from blindmind.llm_schemas import CriticScore

        messages = [{"role": "user", "content": prompt}]
        return await self._completion(messages, settings.critic_temperature, CriticScore)


llm_engine = LLMEngine()
