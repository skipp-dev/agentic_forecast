import os
from langsmith import Client
from langsmith import traceable

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def show_env():
    print("LANGCHAIN_TRACING_V2:", os.getenv("LANGCHAIN_TRACING_V2"))
    print("LANGCHAIN_API_KEY   :", "set" if os.getenv("LANGCHAIN_API_KEY") else "NOT set")
    print("LANGCHAIN_PROJECT   :", os.getenv("LANGCHAIN_PROJECT"))
    print("LANGCHAIN_ENDPOINT  :", os.getenv("LANGCHAIN_ENDPOINT"))


@traceable(name="debug_langsmith_traceable")
def dummy_fn(x: int) -> int:
    return x * 2


def main():
    show_env()
    print("\nCreating LangSmith client explicitly...")
    client = Client()
    print("Client base_url:", client.api_url)

    print("\nCalling dummy_fn(21)...")
    res = dummy_fn(21)
    print("Result:", res)
    print("\nIf tracing works, you should see a run named 'debug_langsmith_traceable' in your LangSmith project.")


if __name__ == "__main__":
    main()