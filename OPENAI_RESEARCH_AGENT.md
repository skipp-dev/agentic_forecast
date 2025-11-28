# OpenAI Research Agent

## Overview

The OpenAI Research Agent is a fully autonomous agent that gathers external market intelligence and sentiment analysis from sources outside the internal data pipeline. **No manual prompts required** - the agent autonomously researches across all market segments.

## Autonomous Operation

The agent operates completely autonomously and researches the following market segments without any manual intervention:

### 🔍 **Autonomous Research Scope**

1. **Individual Symbols**: All symbols from watchlist + major stocks (AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, etc.)
2. **Major Indices**: S&P 500 (SPY), NASDAQ (QQQ), DOW (DIA), Russell 2000 (IWM)
3. **Commodities**: Gold (GLD), Oil (USO), Natural Gas (UNG), Agriculture (DBA), Silver (SLV)
4. **Cryptocurrency**: Bitcoin, Ethereum, crypto regulation, blockchain developments
5. **Economic Policy**: Federal Reserve decisions, interest rates, monetary policy
6. **Economic Indicators**: Inflation (CPI), employment data, GDP, labor market
7. **Geopolitical**: International trade, tariffs, global economic events
8. **Sector News**: AI/tech innovation, electric vehicles, renewable energy

### 🤖 **Autonomous Prompt Generation**

The agent internally generates and executes these research queries:

- `"Research latest news and sentiment for [all watchlist symbols]"`
- `"Gather external market intelligence for portfolio"`
- `"Analyze breaking news impact on [all symbols] futures"`
- `"Monitor social sentiment for crypto markets"`
- `"Monitor social sentiment for DOW, SP500, NASDAQ"`
- `"Track Fed policy and interest rate decisions"`
- `"Monitor commodity price movements and drivers"`
- `"Analyze geopolitical events and market impact"`

## API Requirements

The agent supports multiple news sources for maximum coverage:

### 1. OpenAI API (Required)
```bash
export OPENAI_API_KEY="your-openai-api-key"
```
- Used for sentiment analysis and market impact assessment
- Requires GPT-4 access for best results

### 2. NewsAPI.org (Optional)
```bash
export NEWS_API_KEY="your-newsapi-key"
```
- Provides comprehensive news coverage
- Free tier available: https://newsapi.org/

### 3. Alpha Vantage (Optional)
```bash
export ALPHA_VANTAGE_API_KEY="your-alpha-vantage-key"
```
- Specialized financial news with sentiment scores
- Free tier available: https://www.alphavantage.co/

## Integration

The OpenAI Research Agent is fully integrated into the LangGraph workflow and operates autonomously:

### 🔄 **Workflow Integration**

```python
# Agent operates autonomously - no manual parameters needed
research_insights = await openai_research_node()
```

### 📊 **Data Flow**

1. **Autonomous Execution**: Agent runs automatically during forecasting pipeline
2. **Comprehensive Research**: Gathers intelligence across all 8 market segments
3. **Sentiment Analysis**: Applies GPT-4 analysis to all gathered content
4. **Impact Assessment**: Evaluates market impact of all news and events
5. **Signal Generation**: Produces trading signals based on comprehensive analysis
6. **Risk Integration**: Feeds risk assessments into portfolio optimization

### 🎯 **Output Structure**

```python
{
    "research_summary": "Comprehensive market intelligence across all segments",
    "sentiment_scores": {
        "overall_market": 0.75,
        "individual_symbols": {...},
        "commodities": {...},
        "crypto": {...},
        "economic_indicators": {...}
    },
    "trading_signals": [...],
    "risk_assessment": {...},
    "key_insights": [...]
}
```

## Usage

### Automatic Integration
The agent runs automatically as part of the main forecasting pipeline:

```bash
python main.py --task full
```

### Direct Usage
You can also use the agent directly:

```python
from agents.openai_research_agent import OpenAIResearchAgent

# Initialize agent
research_agent = OpenAIResearchAgent()

# Conduct autonomous research (no parameters needed)
insights = research_agent.conduct_market_research()

print(f"Market Sentiment: {insights.market_sentiment}")
print(f"Risk Assessment: {insights.risk_assessment}")
print(f"Trading Signals: {len(insights.trading_signals)}")
```

## Output Data

The agent stores results in the graph state:

- `research_insights`: Summary of market sentiment, risk assessment, and confidence
- `external_news`: Detailed news articles with sentiment analysis

## Sample Research Output

```
Market Sentiment: bullish (Overall market sentiment across all segments)
Risk Assessment: Moderate volatility expected from tech sector earnings and Fed policy
Trading Signals: 15 signals generated across all market segments
Confidence Score: 0.82

Key Insights by Segment:
- Individual Symbols: AAPL, MSFT, NVDA showing strong momentum
- Commodities: Gold prices stable amid economic uncertainty
- Crypto: Bitcoin consolidating above $60K, Ethereum upgrades positive
- Economic: Fed signals potential rate pause, inflation moderating
- Geopolitical: Trade tensions easing, positive for global markets

Research Coverage: 8 market segments analyzed autonomously
External Sources: 45 articles processed, 120 sentiment scores calculated
```

## Configuration

Add to your `config/settings.toml`:

```toml
[openai]
api_key = "your-key-here"

[research]
max_articles = 50
sentiment_threshold = 0.1
confidence_threshold = 0.7
days_back = 7
```

## Health Monitoring

Check agent status:

```python
status = research_agent.get_health_status()
print(status)
# {'agent_type': 'OpenAIResearchAgent', 'openai_available': True, 'status': 'operational'}
```

## Fallback Behavior

If external APIs are unavailable, the agent:
1. Uses sample data for development/testing
2. Continues with reduced functionality
3. Logs warnings but doesn't break the pipeline

## Best Practices

1. **API Keys**: Set up multiple news sources for comprehensive autonomous coverage
2. **Rate Limits**: Monitor API usage as autonomous research increases request volume
3. **Cost Management**: OpenAI API costs scale with comprehensive market analysis
4. **Data Quality**: External news supplements, doesn't replace, internal data
5. **Integration**: Research insights are combined with technical analysis across all segments
6. **Autonomous Monitoring**: Regularly review autonomous research quality and coverage

## Troubleshooting

### Common Issues

1. **OpenAI API Key Missing**
   ```
   Warning: OpenAI API key not found. LLM features will fail.
   ```
   Solution: Set `OPENAI_API_KEY` environment variable

2. **News APIs Unavailable**
   ```
   No external news APIs available, using sample data
   ```
   Solution: Configure `NEWS_API_KEY` or `ALPHA_VANTAGE_API_KEY`

3. **Rate Limit Exceeded**
   ```
   Failed to fetch news: 429 Client Error
   ```
   Solution: Implement retry logic or reduce request frequency

### Testing

Test the agent independently:

```bash
python -c "
from agents.openai_research_agent import OpenAIResearchAgent
agent = OpenAIResearchAgent()
insights = agent.conduct_market_research()  # Autonomous research
print('Test successful:', insights.confidence_score > 0)
print('Segments covered:', len(insights.sentiment_scores))
"
```</content>
<parameter name="filePath">c:\Users\spreu\Documents\agentic_forecast\OPENAI_RESEARCH_AGENT.md