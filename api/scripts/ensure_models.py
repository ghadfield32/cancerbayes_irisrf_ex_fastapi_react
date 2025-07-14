#!/usr/bin/env python3
"""
Ensure models script - pre-trains all models before starting the API.
This can be used in development or CI to ensure models are ready.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the api directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml.model_service import TRAINERS, ModelService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Ensure all models are trained and loaded."""
    logger.info("🚀 Starting model ensure script...")

    svc = ModelService()

    # Start the self-healing process
    await svc.startup(auto_train=True)

    # Wait until all models are loaded
    max_wait = 300  # 5 minutes max
    start_time = asyncio.get_event_loop().time()

    while len(svc.models) < len(TRAINERS):
        if asyncio.get_event_loop().time() - start_time > max_wait:
            logger.error("❌ Timeout waiting for models to load")
            return False

        logger.info(f"⏳ Waiting for models... ({len(svc.models)}/{len(TRAINERS)} loaded)")

        # Check for failed models
        failed = [name for name, status in svc.status.items() if status == "failed"]
        if failed:
            logger.error(f"❌ Models failed to train: {failed}")
            return False

        await asyncio.sleep(5)

    logger.info("✅ All models loaded successfully!")
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("⏹️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1) 
