from app.services.cache import build_cache_key


def test_cache_key_is_deterministic():
    """La misma combinación de parámetros siempre produce la misma clave."""
    key_a = build_cache_key(system="s", user="u", model="gpt-4o-mini", temperature=0.3)
    key_b = build_cache_key(system="s", user="u", model="gpt-4o-mini", temperature=0.3)
    assert key_a == key_b


def test_cache_key_changes_with_any_relevant_field():
    """Si cambia el system prompt, el usuario, el modelo o la temperatura, la clave cambia.

    Esto es lo que da la invalidación implícita: si agregamos un ejemplo CAG
    nuevo al system prompt, las claves viejas quedan huérfanas solas.
    """
    base = {"system": "s", "user": "u", "model": "gpt-4o-mini", "temperature": 0.3}
    base_key = build_cache_key(**base)

    variants = [
        {**base, "system": "s2"},
        {**base, "user": "u2"},
        {**base, "model": "claude-haiku-4-5-20251001"},
        {**base, "temperature": 0.5},
    ]
    for variant in variants:
        assert build_cache_key(**variant) != base_key
