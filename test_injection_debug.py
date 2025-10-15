"""Debug test to see if client injection is working."""
import os
from pathlib import Path

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

print("Testing DDGS client injection...")
print()

from codeless_orchestrator.tools.builtin.web_search import _DefaultDDGSClient

# Create the client
print("Creating _DefaultDDGSClient...")
client_wrapper = _DefaultDDGSClient()

# Check the inner client
print(f"DDGS client type: {type(client_wrapper._client)}")
print(f"Inner primp.Client type: {type(client_wrapper._client.client)}")

# Check if it has the attributes we set
inner_client = client_wrapper._client.client
print(f"Inner client: {inner_client}")
print(f"Inner client attributes: {dir(inner_client)}")

# Now try to use it
print("\nTrying to perform a search...")
try:
    results = list(client_wrapper.text(q="Python programming", max_results=2))
    print(f"SUCCESS! Got {len(results)} results")
    for r in results:
        print(f"  - {r.get('title', 'N/A')}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
