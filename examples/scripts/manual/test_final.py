"""Final test of the web search fix."""
from dotenv import load_dotenv
load_dotenv()

from codeless_orchestrator.tools.builtin.web_search import WebSearchTool

print("Creating WebSearchTool...")
tool = WebSearchTool()

print("\nTesting with backend='bing'...")
try:
    results = tool.invoke({"q": "Python programming", "max_results": 3, "backend": "bing"})
    print(f"Invoke returned: {results}")
    print(f"Type: {type(results)}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    results = []

print(f"\nSuccess! Got {len(results)} results:")
for i, result in enumerate(results, 1):
    print(f"\n{i}. {result.get('title', 'N/A')}")
    print(f"   URL: {result.get('href', 'N/A')}")
    print(f"   Snippet: {result.get('body', 'N/A')[:100]}...")

