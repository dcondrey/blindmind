import logging
import os
import warnings
from logging.handlers import RotatingFileHandler

from rich.logging import RichHandler

from blindmind.config import settings


def setup_logging():
    # Suppress noisy third-party loggers
    for noisy in ["litellm", "LiteLLM", "httpx", "httpcore", "openai", "anthropic"]:
        logging.getLogger(noisy).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*botocore.*")
    warnings.filterwarnings("ignore", message=".*sagemaker.*")

    log = logging.getLogger("blindmind")
    log.setLevel(settings.log_level)
    log.propagate = False

    # Console: only show errors to keep terminal clean
    console_handler = RichHandler(rich_tracebacks=True, show_path=False, show_time=False, show_level=False)
    console_handler.setLevel(logging.ERROR)
    log.addHandler(console_handler)

    # File: capture everything for debugging
    log_dir = "data"
    os.makedirs(log_dir, exist_ok=True)
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "blindmind.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(file_handler)

    return log


logger = setup_logging()
