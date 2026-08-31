import os
import tempfile

from blindmind.config import Settings


def test_config_defaults():
    settings = Settings(_env_file=None)
    assert settings.litellm_model == "gpt-4o-mini"
    assert settings.variation_temperature == 1.0
    assert settings.critic_threshold == 7.0
    assert settings.crossover_rate == 0.5
    assert settings.point_mutation_rate == 0.3
    assert settings.inversion_rate == 0.2

def test_config_env_override(monkeypatch):
    monkeypatch.setenv("LITELLM_MODEL", "claude-3-opus")
    monkeypatch.setenv("CRITIC_THRESHOLD", "8.5")

    settings = Settings(_env_file=None)
    assert settings.litellm_model == "claude-3-opus"
    assert settings.critic_threshold == 8.5

def test_mutation_rates_sum_to_one():
    settings = Settings(_env_file=None)
    total = settings.crossover_rate + settings.point_mutation_rate + settings.inversion_rate
    assert abs(total - 1.0) < 0.001

def test_save_local_env_preserves_existing():
    settings = Settings(_env_file=None)
    settings.openai_api_key = "sk-test-key"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False, dir=".") as f:
        f.write("CUSTOM_VAR=my_value\n")
        f.write("OTHER_VAR=other_value\n")
        tmpname = f.name

    try:
        orig_name = ".env"
        backup = None
        if os.path.exists(orig_name):
            backup = orig_name + ".bak"
            os.rename(orig_name, backup)

        os.rename(tmpname, orig_name)
        settings.save_local_env()

        with open(orig_name) as f:
            content = f.read()
        assert "CUSTOM_VAR=my_value" in content
        assert "OPENAI_API_KEY=sk-test-key" in content
    finally:
        if os.path.exists(orig_name):
            os.remove(orig_name)
        if backup and os.path.exists(backup):
            os.rename(backup, orig_name)
        if os.path.exists(tmpname):
            os.remove(tmpname)
