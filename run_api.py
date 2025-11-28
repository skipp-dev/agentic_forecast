#!/usr/bin/env python3
"""
Production-ready script to run the Agentic Forecast API server.
This script provides proper error handling and logging for production use.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for running the API server."""
    try:
        # Import FastAPI app
        from api import app
        logger.info("FastAPI app imported successfully")

        # Use uvicorn to run the server
        import uvicorn

        # Get configuration from environment
        host = os.getenv("HOST", "127.0.0.1")
        port = int(os.getenv("PORT", "8000"))

        logger.info(f"Starting server on {host}:{port}")

        # Run the server
        uvicorn.run(
            "api:app",
            host=host,
            port=port,
            reload=False,  # Disable reload in production
            log_level="info"
        )

    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()