#!/usr/bin/env python3
"""
Run the API server programmatically for testing.
"""

import uvicorn
from api import app

if __name__ == "__main__":
    print("Starting API server on http://127.0.0.1:8002")
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")