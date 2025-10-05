"""Service layer package with shared configuration exports."""

from config import settings as settings
from . import favorites_sim as favorites_sim

__all__ = ["settings", "favorites_sim"]
