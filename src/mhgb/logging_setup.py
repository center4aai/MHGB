"""
Настройка loguru для всего проекта MHGB.

Использование:
    from mhgb.logging_setup import logger
    logger.info("Граф загружен: {n} нод", n=len(G))

Вызов setup_logging() происходит при первом импорте модуля.
Повторный импорт безопасен — logger.remove() сбрасывает предыдущие обработчики.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

from loguru import logger

from mhgb.config import cfg


def setup_logging() -> None:
    logger.remove()  # сбросить обработчики по умолчанию

    # Консоль — INFO и выше
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True,
    )

    # Файл — DEBUG и выше, имя = YYYY-MM-DD.log
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = cfg.logs_dir / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.log"
    logger.add(
        log_file,
        level="DEBUG",
        encoding="utf-8",
        rotation="00:00",      # новый файл каждую полночь UTC
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
    )


setup_logging()

__all__ = ["logger"]
