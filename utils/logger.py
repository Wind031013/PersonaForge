import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logger(name: str = "logger",
                 log_path: Path = Path(__file__).parent,
                 console_level: int = logging.DEBUG,
                 file_level: int = logging.DEBUG) -> logging.Logger:
    log_path.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = RotatingFileHandler(
        log_path / "app.log",
        maxBytes=1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger