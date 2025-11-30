import os
print("LANGSMITH_ENDPOINT:", os.getenv("LANGSMITH_ENDPOINT"))
print("LANGSMITH_API_KEY set:", bool(os.getenv("LANGSMITH_API_KEY")))

from langsmith import Client

client = Client(
    api_url=os.getenv("LANGSMITH_ENDPOINT"),
    api_key=os.getenv("LANGSMITH_API_KEY"),
)

print("Listing first few projects:")
projects = list(client.list_projects())[:5]
for p in projects:
    print("-", p.name, getattr(p, "id", None))