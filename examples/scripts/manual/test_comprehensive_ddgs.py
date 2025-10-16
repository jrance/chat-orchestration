"""Comprehensive test suite for ddgs 9.6.1 with proxy and CA bundle."""
from dotenv import load_dotenv
load_dotenv()

from codeless_orchestrator.tools.builtin.web_search import WebSearchTool

print("="*70)
print("COMPREHENSIVE TESTS FOR DDGS 9.6.1+ WITH PROXY + CA BUNDLE")
print("="*70)
print()

tool = WebSearchTool()

# Test 1: Basic search
print("TEST 1: Basic search")
try:
    results = tool.invoke({"q": "Python programming", "max_results": 2})
    print(f"? PASSED - Got {len(results)} results")
    if results:
        print(f"   First result: {results[0]['title'][:60]}...")
except Exception as e:
    print(f"? FAILED - {str(e)[:100]}")
print()

# Test 2: Search with region
print("TEST 2: Search with region parameter")
try:
    results = tool.invoke({"q": "news", "max_results": 2, "region": "us-en"})
    print(f"? PASSED - Got {len(results)} results")
except Exception as e:
    print(f"? FAILED - {str(e)[:100]}")
print()

# Test 3: Search with safesearch
print("TEST 3: Search with safesearch parameter")
try:
    results = tool.invoke({"q": "test", "max_results": 2, "safesearch": "moderate"})
    print(f"? PASSED - Got {len(results)} results")
except Exception as e:
    print(f"? FAILED - {str(e)[:100]}")
print()

# Test 4: Search with time filter
print("TEST 4: Search with time filter")
try:
    results = tool.invoke({"q": "technology", "max_results": 2, "time": "d"})
    print(f"? PASSED - Got {len(results)} results")
except Exception as e:
    print(f"? FAILED - {str(e)[:100]}")
print()

# Test 5: Different backends
print("TEST 5: Testing different backends")
for backend in ["html", "lite"]:
    try:
        results = tool.invoke({"q": "test", "max_results": 1, "backend": backend})
        print(f"? {backend:10s} - Got {len(results)} results")
    except Exception as e:
        print(f"? {backend:10s} - {str(e)[:60]}")
print()

print("="*70)
print("ALL TESTS COMPLETED!")
print("="*70)

