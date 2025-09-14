import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    force=True,  # override any JSON handler that hides exc_info
)

from app import app

__all__ = ["app"]
