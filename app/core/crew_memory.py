"""
OrthoLink CrewAI Long-Term Memory — SQLite storage adapter.

PRD: Cross-session learning for DVA and RSA. Device profiles and prior regulatory
decisions are persisted and recalled so agents avoid redundant inputs.
"""

import logging
from pathlib import Path
from typing import Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_ltm_instance: Optional[object] = None


def get_ltm_memory():
    """
    Return a CrewAI LongTermMemory instance with SQLite storage.
    Used in Crew(..., memory=True, long_term_memory=get_ltm_memory()).
    """
    global _ltm_instance
    if _ltm_instance is not None:
        return _ltm_instance
    try:
        from crewai.memory import LongTermMemory
        from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage

        settings = get_settings()
        base = getattr(settings, "faiss_index_path", "data/embeddings")
        data_dir = Path(base).parent if base else Path("data")
        data_dir = data_dir.resolve() if data_dir else Path("data").resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(data_dir / "crewai_memory.db")
        storage = LTMSQLiteStorage(db_path=db_path)
        _ltm_instance = LongTermMemory(storage=storage)
        logger.info("CrewAI LongTermMemory initialized: %s", db_path)
        return _ltm_instance
    except ImportError as e:
        logger.warning("CrewAI memory not available: %s. Running without LTM.", e)
        return None
