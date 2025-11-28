#!/usr/bin/env python3
"""
Simple test script for OpenAI Research Agent
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

try:
    from agents.openai_research_agent import OpenAIResearchAgent
    print("✓ Import successful")

    agent = OpenAIResearchAgent()
    print("✓ Agent initialized")

    insights = agent.conduct_market_research()
    print("✓ Research completed")
    print(f"Confidence score: {insights.confidence_score}")
    print(f"Market sentiment: {insights.market_sentiment}")
    print(f"Trading signals: {len(insights.trading_signals)}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()