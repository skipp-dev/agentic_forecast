import argparse
import sys
from src.data.model_registry import ModelRegistry

def main():
    parser = argparse.ArgumentParser(description="Manage the Global Trading Kill Switch")
    parser.add_argument("action", choices=["status", "enable", "disable"], help="Action to perform")
    parser.add_argument("--db", default="data/model_registry.db", help="Path to registry DB")
    
    args = parser.parse_args()
    
    registry = ModelRegistry(args.db)
    
    if args.action == "status":
        status = registry.get_trading_status()
        print(f"Trading Enabled: {status}")
        sys.exit(0 if status else 1)
        
    elif args.action == "enable":
        registry.set_trading_status(True)
        print("Trading ENABLED.")
        
    elif args.action == "disable":
        registry.set_trading_status(False)
        print("Trading DISABLED (Kill Switch Activated).")

if __name__ == "__main__":
    main()
