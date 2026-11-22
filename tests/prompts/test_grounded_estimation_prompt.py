import pytest
from jinja2 import UndefinedError

from app.prompts.loader import render_grounded_estimation_prompt

CHUNKS = [
    {"chunk_id": "7", "document_id": "BUD-Z", "content": "Component: Auth\nEstimated hours: 120"},
    {"chunk_id": "12", "document_id": "BUD-Y", "content": "Component: Ledger\nEstimated hours: 90"},
]


def test_system_prompt_states_the_attribution_rules():
    system, _ = render_grounded_estimation_prompt(query="q", chunks=CHUNKS)

    assert "grounded=false" in system
    assert "VERBATIM" in system
    assert "Insufficient data:" in system
    assert "hours=0" in system
    assert "chunk_id" in system and "document_id" in system


def test_system_prompt_forbids_totals_and_unrequested_components():
    system, _ = render_grounded_estimation_prompt(query="q", chunks=CHUNKS)

    assert "Do not add totals" in system
    assert "Do not add components the request does not call for" in system


def test_user_prompt_contains_the_query():
    _, user = render_grounded_estimation_prompt(query="Mobile banking app", chunks=CHUNKS)

    assert "Mobile banking app" in user


def test_user_prompt_exposes_every_chunk_with_its_ids_and_text():
    _, user = render_grounded_estimation_prompt(query="q", chunks=CHUNKS)

    assert '<chunk chunk_id="7" document_id="BUD-Z">' in user
    assert '<chunk chunk_id="12" document_id="BUD-Y">' in user
    assert "Estimated hours: 120" in user and "Estimated hours: 90" in user
    assert user.count("</chunk>") == 2


def test_chunks_appear_in_the_given_order():
    _, user = render_grounded_estimation_prompt(query="q", chunks=CHUNKS)

    assert user.index('chunk_id="7"') < user.index('chunk_id="12"')


def test_user_prompt_still_renders_with_no_chunks():
    _, user = render_grounded_estimation_prompt(query="q", chunks=[])

    assert "<chunk" not in user


def test_a_chunk_missing_an_id_fails_instead_of_rendering_blank():
    with pytest.raises(UndefinedError):
        render_grounded_estimation_prompt(query="q", chunks=[{"content": "no ids here"}])


def test_unknown_prompt_version_is_rejected():
    with pytest.raises(ValueError, match="v99"):
        render_grounded_estimation_prompt(query="q", chunks=CHUNKS, version="v99")
