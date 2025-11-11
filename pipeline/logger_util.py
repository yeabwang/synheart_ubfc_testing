"""Lightweight logging utility with optional Rich integration and file logging.

Usage:
    from pipeline.logger_util import get_logger
    log = get_logger(__name__)
    log.info("message")
    log.warning("warn")
    log.error("error")
"""

from __future__ import annotations

import logging
import sys
from typing import Optional
from pathlib import Path
from datetime import datetime

_RICH_AVAILABLE = False
try:
    from rich.console import Console
    from rich.logging import RichHandler

    _RICH_AVAILABLE = True
except Exception:
    _RICH_AVAILABLE = False


def _configure_root_logger(level: int = logging.INFO) -> None:
    # Ensure logs directory exists at repo root
    try:
        root = Path(__file__).resolve().parents[2]
    except Exception:
        root = Path.cwd()
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logfile = logs_dir / f"run_{timestamp}.log"

    # Console handler 
    if _RICH_AVAILABLE:
        console = Console(stderr=True)
        console_handler = RichHandler(console=console, markup=True, rich_tracebacks=True)
        console_format = logging.Formatter("%(message)s", datefmt="%H:%M:%S")
        console_handler.setFormatter(console_format)
    else:
        console_handler = logging.StreamHandler(sys.stderr)
        console_format = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_format)

    # File handler
    file_handler = logging.FileHandler(logfile, encoding="utf-8")
    file_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    # Avoid duplicating handlers if reconfigured
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


_configured = False


def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    global _configured
    if not _configured:
        _configure_root_logger(level=level)
        _configured = True
    logger = logging.getLogger(name if name else "ubfc")
    logger.setLevel(level)
    return logger
