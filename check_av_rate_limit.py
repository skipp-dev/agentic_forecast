import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
if not api_key:
    print("Error: ALPHA_VANTAGE_API_KEY not found.")
    exit(1)

print(f"API Key found: {api_key[:4]}...")

symbols = ["IBM", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "ADBE"]
base_url = "https://www.alphavantage.co/query"

print("Testing Alpha Vantage rate limits with 10 rapid requests...")

success_count = 0
for i, symbol in enumerate(symbols):
    params = {
        'function': 'GLOBAL_QUOTE',
        'symbol': symbol,
        'apikey': api_key
    }
    
    try:
        start = time.time()
        response = requests.get(base_url, params=params)
        duration = time.time() - start
        
        data = response.json()
        
        if 'Global Quote' in data and data['Global Quote']:
            print(f"[{i+1}/{len(symbols)}] {symbol}: Success ({duration:.2f}s)")
            success_count += 1
        elif 'Note' in data:
            print(f"[{i+1}/{len(symbols)}] {symbol}: RATE LIMITED - {data['Note']}")
        elif 'Error Message' in data:
            print(f"[{i+1}/{len(symbols)}] {symbol}: Error - {data['Error Message']}")
        else:
            print(f"[{i+1}/{len(symbols)}] {symbol}: Unknown response - {data.keys()}")
            
    except Exception as e:
        print(f"[{i+1}/{len(symbols)}] {symbol}: Exception - {e}")

print("-" * 20)
if success_count == len(symbols):
    print("✅ No rate limiting detected in this short burst.")
else:
    print("⚠️ Rate limiting or errors detected.")
