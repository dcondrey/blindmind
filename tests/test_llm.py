from unittest.mock import AsyncMock, patch

import pytest
from litellm import exceptions

from blindmind.llm import LLMEngine, LLMStats
from blindmind.llm_schemas import CriticScore, MutationOutput


def _engine_with_providers(providers):
    """Fresh LLMEngine pre-seeded with a fixed provider list, bypassing real host
    discovery (live OpenRouter fetch, real claude CLI subprocess) so tests are
    hermetic and deterministic instead of depending on the machine's actual
    installed tools and API keys."""
    engine = LLMEngine()
    engine.providers = providers
    engine._initialized = True
    return engine


TWO_PROVIDERS = [
    {"model": "gpt-4o-mini", "api_key": "k1", "label": "P1"},
    {"model": "gpt-4o-mini", "api_key": "k2", "label": "P2"},
]
THREE_PROVIDERS = TWO_PROVIDERS + [{"model": "gpt-4o-mini", "api_key": "k3", "label": "P3"}]


def _mock_response(content: str, in_tok=100, out_tok=50):
    resp = AsyncMock()
    resp.choices = [AsyncMock(message=AsyncMock(content=content))]
    resp.usage = AsyncMock(prompt_tokens=in_tok, completion_tokens=out_tok)
    return resp


@pytest.mark.asyncio
async def test_generate_mutation_success():
    engine = _engine_with_providers(TWO_PROVIDERS)
    mock_response = _mock_response(
        '{"title": "Quantum Logistics", "domain": "Logistics", '
        '"description": "Using entanglement for instant tracking", "justification": "Novel combo"}'
    )

    with patch("blindmind.llm.acompletion", return_value=mock_response) as mock_acompletion:
        result = await engine.generate_mutation("Test prompt")
        assert isinstance(result, MutationOutput)
        assert result.title == "Quantum Logistics"
        mock_acompletion.assert_called_once()

@pytest.mark.asyncio
async def test_generate_mutation_with_retries():
    engine = _engine_with_providers(THREE_PROVIDERS)
    mock_response = _mock_response('{"title": "Success", "domain": "D", "description": "Desc", "justification": "J"}')

    with patch("blindmind.llm.acompletion", side_effect=[
        exceptions.RateLimitError("Rate limit reached", model="gpt-4o-mini", llm_provider="openai"),
        exceptions.RateLimitError("Rate limit reached", model="gpt-4o-mini", llm_provider="openai"),
        mock_response
    ]) as mock_acompletion:
        result = await engine.generate_mutation("Test prompt")
        assert result.title == "Success"
        assert mock_acompletion.call_count == 3

@pytest.mark.asyncio
async def test_critique_mutation():
    engine = _engine_with_providers(TWO_PROVIDERS)
    mock_response = _mock_response(
        '{"conceptual_novelty": 9, "feasibility": 5, "utility": 8, "semantic_jump": 7, '
        '"rationale": "High novelty but hard to implement", "evolutionary_directive": "Focus on stability"}',
        in_tok=80,
        out_tok=40,
    )

    with patch("blindmind.llm.acompletion", return_value=mock_response):
        result = await engine.critique_mutation("Test prompt")
        assert isinstance(result, CriticScore)
        assert result.semantic_jump == 7

def test_llm_stats_tracking():
    stats = LLMStats()
    stats.record(input_tokens=100, output_tokens=50, latency_ms=500)
    stats.record(input_tokens=200, output_tokens=100, latency_ms=700)
    stats.record_failure()

    summary = stats.summary
    assert summary["total_calls"] == 3
    assert summary["successful"] == 2
    assert summary["failed"] == 1
    assert summary["input_tokens"] == 300
    assert summary["output_tokens"] == 150
    assert summary["avg_latency_ms"] == 600


@pytest.mark.asyncio
async def test_transient_failure_falls_back_without_blacklisting():
    """A rate limit on the best provider should fail over for this call only.
    The next call must still try the best provider first rather than staying
    downgraded for the rest of the session (this was the rotation-never-resets bug:
    one transient error used to permanently exile a good provider)."""
    engine = _engine_with_providers(TWO_PROVIDERS)
    good_response = _mock_response(
        '{"title": "T", "domain": "D", "description": "Desc", "justification": "J"}', in_tok=10, out_tok=10
    )

    with patch("blindmind.llm.acompletion", side_effect=[
        exceptions.RateLimitError("limited", model="m", llm_provider="p"),
        good_response,
    ]) as mock_acompletion:
        await engine.generate_mutation("prompt 1")
        assert mock_acompletion.call_count == 2
        assert engine._blacklisted == set()

    with patch("blindmind.llm.acompletion", return_value=good_response) as mock_acompletion_2:
        await engine.generate_mutation("prompt 2")
        assert mock_acompletion_2.call_count == 1


@pytest.mark.asyncio
async def test_auth_failure_blacklists_provider_for_session():
    engine = _engine_with_providers(TWO_PROVIDERS)
    good_response = _mock_response(
        '{"title": "T", "domain": "D", "description": "Desc", "justification": "J"}', in_tok=10, out_tok=10
    )

    with patch("blindmind.llm.acompletion", side_effect=[
        exceptions.AuthenticationError("bad key", model="m", llm_provider="p"),
        good_response,
    ]):
        await engine.generate_mutation("prompt 1")
        assert engine._blacklisted == {0}

    with patch("blindmind.llm.acompletion", return_value=good_response) as mock_acompletion_2:
        await engine.generate_mutation("prompt 2")
        assert mock_acompletion_2.call_count == 1


def test_set_model_override_puts_override_first_and_forces_rebuild():
    engine = _engine_with_providers(TWO_PROVIDERS)
    engine.set_model_override("gpt-4o")
    assert engine._initialized is False
    assert engine._blacklisted == set()
