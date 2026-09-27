import logging

import structlog


def configure_logging(app_env: str, log_level: str) -> None:
    """Configura structlog: consola legible en desarrollo, JSON en producción.

    Los mismos eventos sirven para leer en la terminal durante el desarrollo
    y para ser ingeridos por una herramienta de observabilidad en producción.
    """
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer = (
        structlog.processors.JSONRenderer()
        if app_env == "production"
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
