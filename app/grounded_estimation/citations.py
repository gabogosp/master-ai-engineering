import structlog

from app.grounded_estimation.schemas import (
    CitationReport,
    Estimate,
    LineStatus,
    LineVerdict,
)

logger = structlog.get_logger(__name__)


def verify_citations(
    estimate: Estimate,
    retrieved_chunk_ids: set[str],
) -> CitationReport:
    """Flag any line whose cited chunk_id was never in the retrieved context.

    A line is DANGLING as soon as ONE of its sources points to an id the LLM
    was never given, even if its other sources are valid: a partly invented
    citation is still an invented citation.

    `request_id` is not a parameter: it travels through structlog contextvars,
    so every log emitted here is correlated with the request automatically.
    """
    verdicts: list[LineVerdict] = []

    for line in estimate.lines:
        if not line.grounded:
            verdicts.append(
                LineVerdict(
                    component=line.component,
                    status=LineStatus.INSUFFICIENT_DATA,
                    dangling_chunk_ids=[],
                )
            )
            continue

        dangling = [
            source.chunk_id
            for source in line.sources
            if source.chunk_id not in retrieved_chunk_ids
        ]
        verdicts.append(
            LineVerdict(
                component=line.component,
                status=LineStatus.DANGLING_CITATION if dangling else LineStatus.GROUNDED,
                dangling_chunk_ids=dangling,
            )
        )

    report = CitationReport(verdicts=verdicts)

    logger.info(
        "citation_report",
        lines=len(report.verdicts),
        grounded=report.grounded_count,
        dangling=report.dangling_count,
        insufficient=report.insufficient_count,
    )
    for verdict in report.verdicts:
        if verdict.status is LineStatus.DANGLING_CITATION:
            logger.error(
                "dangling_citation",
                component=verdict.component,
                dangling_chunk_ids=verdict.dangling_chunk_ids,
            )

    return report
