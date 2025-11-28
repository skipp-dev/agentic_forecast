#!/bin/bash

# Try to install textblob with timeout
pip install --default-timeout=30 textblob 2>/dev/null || echo "Warning: textblob install failed or timed out"

# Start the main application
python main.py
