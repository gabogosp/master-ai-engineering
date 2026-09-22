from app.config import Settings
from app.context.examples import ESTIMATION_EXAMPLES
from app.services.llm_service import SYSTEM_INSTRUCTIONS, build_system_prompt


def test_settings_defaults():
    """Verifica que la configuración cargue los valores por defecto esperados."""
    s = Settings()
    assert s.llm_provider in ("openai", "anthropic")
    assert s.app_env is not None
    assert s.log_level is not None


def test_build_system_prompt_structure():
    """Verifica que el prompt del sistema CAG incluya las instrucciones y todos los ejemplos."""
    prompt = build_system_prompt()

    # Debe contener las instrucciones base
    assert SYSTEM_INSTRUCTIONS in prompt
    assert "50 EUR/h" in prompt
    assert "Riesgo de plazo:" in prompt

    # Debe contener la sección de ejemplos
    assert "## Previous estimation examples" in prompt

    # Cada ejemplo en ESTIMATION_EXAMPLES debe estar formateado con sus tags XML
    for i, example in enumerate(ESTIMATION_EXAMPLES, start=1):
        assert f'<example number="{i}">' in prompt
        assert example["meeting_summary"] in prompt
        assert example["estimation"] in prompt
