"""Test with different backends to isolate the issue."""
import os
from dotenv import load_dotenv
load_dotenv()

from codeless_orchestrator.tools.builtin.web_search import _DefaultDDGSClient

client_wrapper = _DefaultDDGSClient()

print("Testing different backends...")
print()

# Test with html backend
print("1. Testing with backend='html'...")
try:
    results = list(client_wrapper.text(q="Python", max_results=2, backend="html"))
    print(f"   SUCCESS! Got {len(results)} results")
    for r in results:
        print(f"     - {r.get('title', 'N/A')[:60]}")
except Exception as e:
    print(f"   FAILED: {type(e).__name__}: {str(e)[:100]}")

print()

# Test with lite backend
print("2. Testing with backend='lite'...")
try:
    results = list(client_wrapper.text(q="Python", max_results=2, backend="lite"))
    print(f"   SUCCESS! Got {len(results)} results")
    for r in results:
        print(f"     - {r.get('title', 'N/A')[:60]}")
except Exception as e:
    print(f"   FAILED: {type(e).__name__}: {str(e)[:100]}")

print()

# Test with bing backend
print("3. Testing with backend='bing'...")
try:
    results = list(client_wrapper.text(q="Python", max_results=2, backend="bing"))
    print(f"   SUCCESS! Got {len(results)} results")
    for r in results:
        print(f"     - {r.get('title', 'N/A')[:60]}")
except Exception as e:
    print(f"   FAILED: {type(e).__name__}: {str(e)[:100]}")
