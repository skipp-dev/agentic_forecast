import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('FMP_API_KEY')
symbol = 'AAPL'

print(f"Testing FMP API with key: {api_key[:4]}...{api_key[-4:]}")

endpoints = [
    ("Quote (Usually Free)", f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={api_key}"),
    ("Profile", f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={api_key}"),
    ("Income Statement", f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?period=annual&limit=10&apikey={api_key}"),
]

for name, url in endpoints:
    print(f"\nTesting {name}...")
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                print("✅ Success! Data received.")
            else:
                print(f"⚠️  Response 200 but empty/unexpected data: {data}")
        else:
            print(f"❌ Failed.")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
