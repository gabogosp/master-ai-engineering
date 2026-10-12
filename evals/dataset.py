"""Casos dorados tipados + loader del JSON."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

ProjectTypeStr = Literal["mobile_app", "web_saas", "internal_tool", "data_pipeline"]
DetailLevelStr = Literal["summary", "medium", "detailed"]
OutputFormatStr = Literal["phases_table", "line_items", "narrative"]
TierStr = Literal["auto", "executive", "pm", "developer", "default"]


class GoldenCase(BaseModel):
    """Un transcript + el sobre esperado alrededor de la respuesta del
    modelo.

    La mayoría de los chequeos son rangos laxos en vez de igualdad exacta:
    un LLM tiene margen legítimo en cómo dimensiona una fase. El objetivo
    es atrapar regresiones obvias (fuera de presupuesto, falta tecnología,
    tier equivocado), no clavar cada dígito."""

    id: str = Field(min_length=3, max_length=64)
    transcript: str = Field(min_length=20, max_length=80_000)
    project_type: ProjectTypeStr
    detail_level: DetailLevelStr = "medium"
    output_format: OutputFormatStr = "phases_table"
    tier: TierStr = "auto"

    expected_out_of_scope: bool = False
    expected_in_summary: list[str] = Field(default_factory=list)
    expected_technologies_any_of: list[str] = Field(default_factory=list)
    expected_phase_count_range: tuple[int, int] | None = None
    expected_cost_range_eur: tuple[int, int] | None = None
    expected_duration_weeks_range: tuple[int, int] | None = None
    expected_tier: TierStr | None = None  # se compara contra last_resolved_tier


_DEFAULT_PATH = Path(__file__).resolve().parent / "golden_dataset.json"


def load_dataset(path: Path | str | None = None) -> list[GoldenCase]:
    target = Path(path) if path else _DEFAULT_PATH
    raw = json.loads(target.read_text(encoding="utf-8"))
    return [GoldenCase.model_validate(entry) for entry in raw]
