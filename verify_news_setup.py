import os
import sys
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env
load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.newsapi_client import NewsAPIClient
    from src.agents.news_data_agent import NewsDataAgent
except ImportError as e:
    logger.error(f"Import failed: {e}")
    sys.exit(1)

def test_news_fetch():
    api_key = os.environ.get("NEWSAPI_AI_KEY")
    if not api_key:
        logger.warning("NEWSAPI_AI_KEY not found in env. Checking NEWS_API_KEY...")
        api_key = os.environ.get("NEWS_API_KEY")
    
    if not api_key:
        logger.error("No News API key found in environment variables.")
        return

    logger.info(f"Found API Key: {api_key[:4]}...{api_key[-4:]}")

    # Config mock
    config = {
        "newsapi_ai": {
            "enabled": True,
            "api_key": api_key
        },
        "news_api": {
            "enabled": False
        }
    }

    try:
        agent = NewsDataAgent(config)
        symbols = ["AAPL"]
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        
        logger.info(f"Fetching news for {symbols} from {start_date} to {end_date}...")
        news = agent.fetch_news(symbols, start_date, end_date)
        
        if news:
            logger.info(f"Successfully fetched {len(news.get('AAPL', []))} articles for AAPL.")
            if news['AAPL']:
                logger.info(f"Sample headline: {news['AAPL'][0].get('headline')}")
        else:
            logger.warning("No news fetched.")
            
    except Exception as e:
        logger.error(f"News fetch failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_news_fetch()
