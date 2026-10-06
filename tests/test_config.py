from app.config import Settings


def test_settings_defaults():
    """Verifica que la configuración cargue los valores por defecto esperados."""
    s = Settings()
    assert s.llm_provider in ("openai", "anthropic")
    assert s.app_env is not None
    assert s.log_level is not None
