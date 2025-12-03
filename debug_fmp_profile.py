import os
from dotenv import load_dotenv
from src.fmp_client import FMPClient
import json

load_dotenv()

def debug_profile():
    client = FMPClient()
    symbol = 'AAPL'
    print(f"Fetching profile for {symbol}...")
    profile = client.get_company_profile(symbol)
    
    if profile:
        print("Profile received:")
        print(json.dumps(profile, indent=2))
        
        # Check for market cap keys
        keys = profile[0].keys() if isinstance(profile, list) and len(profile) > 0 else profile.keys()
        print("\nKeys found:", list(keys))
        
        if isinstance(profile, list) and len(profile) > 0:
            p = profile[0]
            print(f"\nmktCap: {p.get('mktCap')}")
            print(f"marketCap: {p.get('marketCap')}")
    else:
        print("No profile received.")

if __name__ == "__main__":
    debug_profile()