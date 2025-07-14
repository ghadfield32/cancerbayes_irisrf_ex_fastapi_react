# api/app/db.py
from contextlib import asynccontextmanager
import os, logging, asyncio
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from .models import Base
from .services.ml.model_service import model_service
from .core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database engine & session factory (module-level singletons – cheap & safe)
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./app.db")
engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

# Global readiness flag
_app_ready: bool = False

def get_app_ready():
    """Get the current app ready status."""
    return _app_ready

# ---------------------------------------------------------------------------
# FastAPI lifespan – runs ONCE at startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app):
    """Open & dispose engine at app startup/shutdown; create all tables."""
    global _app_ready

    logger.info("🗄️  Initializing database…  URL=%s", DATABASE_URL)
    try:
        async with engine.begin() as conn:
            # DDL is safe here; it blocks startup until complete
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Database tables created/verified successfully")

        # Initialize application readiness
        logger.info("🚀 Startup event starting - _app_ready=%s", _app_ready)

        if settings.UNIT_TESTING:
            logger.info("🔒 UNIT_TESTING=1 – startup hooks bypassed")
            _app_ready = True
            logger.info("✅ _app_ready set to True (unit testing)")
        else:
            try:
                # Initialize ModelService first
                logger.info("🔧 Initializing ModelService")
                await model_service.initialize()
                logger.info("✅ ModelService initialized successfully")

                # Start background training tasks
                logger.info("🔄 Starting background training tasks")
                asyncio.create_task(model_service.startup())
                logger.info("✅ Background training tasks started")

                # Set ready to true after initialization (models will load in background)
                _app_ready = True
                logger.info("🚀 FastAPI ready – _app_ready=%s, health probes will pass immediately", _app_ready)

            except Exception as e:
                logger.error("❌ Startup event failed: %s", e)
                import traceback
                logger.error("❌ Startup traceback: %s", traceback.format_exc())
                # Set ready to true anyway so the API can serve requests
                _app_ready = True
                logger.warning("⚠️  Setting _app_ready=True despite startup errors")

        logger.info("🎯 Lifespan startup complete - _app_ready=%s", _app_ready)
        yield
    finally:
        logger.info("🔒 Disposing database engine…")
        await engine.dispose()

# ---------------------------------------------------------------------------
# Dependency injection helper
# ---------------------------------------------------------------------------
async def get_db() -> AsyncSession:
    """Yield a new DB session per request."""
    async with AsyncSessionLocal() as session:
        yield session

