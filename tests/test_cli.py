import difflib

from blindmind.cli import MENU_WORDS, _friendly_llm_error


def test_menu_unknown_choice_suggests_close_match():
    """Mirrors the interactive menu's unknown-choice branch: a near-miss on a
    long-form word should surface a suggestion instead of silently doing nothing."""
    assert difflib.get_close_matches("evlove", MENU_WORDS, n=1, cutoff=0.6) == ["evolve"]
    assert difflib.get_close_matches("expor", MENU_WORDS, n=1, cutoff=0.6) == ["export"]


def test_menu_unknown_choice_no_suggestion_for_garbage():
    assert difflib.get_close_matches("zzz", MENU_WORDS, n=1, cutoff=0.6) == []


def test_friendly_llm_error_adds_hint_for_no_api_keys():
    result = _friendly_llm_error("No API keys found for any supported provider.")
    assert "No API keys found" in result
    assert "-> " in result


def test_friendly_llm_error_adds_hint_for_all_providers_failed():
    result = _friendly_llm_error("All LLM providers failed. Last error: RateLimitError")
    assert "All LLM providers failed" in result
    assert "--model" in result


def test_friendly_llm_error_passthrough_for_unknown_message():
    assert _friendly_llm_error("some other error") == "some other error"
