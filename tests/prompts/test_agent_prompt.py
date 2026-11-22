from app.prompts.loader import render_agent_system_prompt


def test_prompt_names_the_three_tools_and_the_turn_budget():
    prompt = render_agent_system_prompt(max_iterations=7)
    for tool in ("search_budgets", "calculate_estimate", "validate_estimate"):
        assert tool in prompt
    assert "7 turns" in prompt


def test_prompt_tells_the_model_to_search_in_english_and_never_invent_hours():
    prompt = render_agent_system_prompt(max_iterations=10)
    assert "write queries in English" in prompt
    assert "Never make up reference hours" in prompt
