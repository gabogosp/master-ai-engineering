import pytest

from evals.stress.scenarios import Scenario, ScenarioTurn, all_scenarios, get_scenario
from app.schemas import DetailLevel, OutputFormat, ProjectType


def test_all_scenarios_returns_the_three_profiles_in_order():
    names = [s.name for s in all_scenarios()]
    assert names == ["growing", "pivot", "contradiction"]


def test_get_scenario_raises_on_unknown_name():
    with pytest.raises(ValueError, match="desconocido"):
        get_scenario("does-not-exist")


@pytest.mark.parametrize("name", ["growing", "pivot", "contradiction"])
def test_each_scenario_has_at_least_20_sequential_turns(name):
    scenario = get_scenario(name)
    assert len(scenario.turns) >= 20
    assert [t.turn_index for t in scenario.turns] == list(range(1, len(scenario.turns) + 1))


@pytest.mark.parametrize("name", ["growing", "pivot", "contradiction"])
def test_first_turn_always_introduces_the_project_name(name):
    """Es el hecho que todos los perfiles van a necesitar que
    MemoryDriftMetric pueda buscar en el turno 20."""
    first = get_scenario(name).turns[0]
    assert first.fact_field == "project_name"
    assert first.fact_to_remember is not None


def test_pivot_scenario_swaps_stack_at_turn_5():
    pivot = get_scenario("pivot")
    turn5 = pivot.turns[4]
    assert turn5.turn_index == 5
    assert turn5.fact_to_remember == "Flutter"
    assert turn5.fact_field == "technologies"
    # Turnos previos mencionan el stack original -- si no, no hay nada de
    # qué "pivotar".
    assert any("React Native" in t.transcript for t in pivot.turns[:4])


def test_contradiction_scenario_disagrees_at_turns_3_and_8():
    contradiction = get_scenario("contradiction")
    turn3, turn8 = contradiction.turns[2], contradiction.turns[7]
    assert turn3.turn_index == 3 and turn8.turn_index == 8
    assert turn3.fact_to_remember != turn8.fact_to_remember
    # El valor del turno 8 (el más reciente) es el que debería prevalecer
    # según la regla de merge de ProjectMetadata -- lo que confirmará
    # MemoryDriftMetric en el Bloque 4.
    assert turn8.fact_to_remember == "75000"
    assert turn3.fact_to_remember == "28000"


def test_turns_up_to_truncates_without_mutating_the_scenario():
    scenario = get_scenario("growing")
    prefix = scenario.turns_up_to(6)
    assert [t.turn_index for t in prefix] == [1, 2, 3, 4, 5, 6]
    assert len(scenario.turns) == 20  # el original no se tocó


def test_scenario_rejects_fewer_than_20_turns():
    with pytest.raises(ValueError, match="al menos 20"):
        Scenario(
            name="growing",
            project_type=ProjectType.WEB_SAAS,
            detail_level=DetailLevel.MEDIUM,
            output_format=OutputFormat.PHASES_TABLE,
            turns=[ScenarioTurn(1, "demasiado corto")],
        )


def test_scenario_turn_is_an_actual_tuple():
    """La consigna pide 'una lista de tuplas' -- ScenarioTurn es un
    NamedTuple, así que sigue siendo una tupla de verdad (acceso
    posicional incluido), con nombres de campo para legibilidad."""
    turn = get_scenario("growing").turns[0]
    assert isinstance(turn, tuple)
    assert turn[0] == turn.turn_index
    assert turn[1] == turn.transcript
