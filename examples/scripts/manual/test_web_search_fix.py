"""Test to confirm the TLS certificate error is fixed."""
from dotenv import load_dotenv
load_dotenv()

from codeless_orchestrator.tools.builtin.web_search import WebSearchTool

print("="*70)
print("TESTING WEB SEARCH WITH PROXY + CA BUNDLE")
print("="*70)
print()

tool = WebSearchTool()

print("Attempting search (this should NOT raise certificate errors)...")
print()

try:
    results = tool.invoke({"q": "test query", "max_results": 2, "backend": "bing"})
    print(f"? SUCCESS! No certificate errors.")
    print(f"   Search completed and returned {len(results)} results.")
    print()
    print("The TLS certificate verification is now working correctly!")
    print("If you're getting 0 results, that's a separate issue (likely")
    print("DuckDuckGo blocking, rate limiting, or Fiddler configuration).")
    
except Exception as e:
    error_msg = str(e)
    if "CERTIFICATE_VERIFY_FAILED" in error_msg or "cert verification failed" in error_msg:
        print(f"? FAILED! Certificate error still occurring:")
        print(f"   {error_msg[:200]}")
    else:
        print(f"? Different error occurred:")
        print(f"   {error_msg[:200]}")

