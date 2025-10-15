"""Final verification that the original certificate error is fixed."""
from dotenv import load_dotenv
load_dotenv()

import sys

print("="*70)
print("VERIFICATION: Original TLS Certificate Error is Fixed")
print("="*70)
print()

# Import the tool
from codeless_orchestrator.tools.builtin.web_search import WebSearchTool

# Check environment
import os
print("Environment Configuration:")
print(f"  PROXY_ENABLED: {os.getenv('PROXY_ENABLED')}")
print(f"  PROXY_URL: {os.getenv('PROXY_URL')}")
print(f"  PROXY_CA_BUNDLE: {os.getenv('PROXY_CA_BUNDLE')}")
print()

# Check ddgs version
import ddgs
print(f"ddgs version: {ddgs.__version__}")
print()

# Create the tool and test
print("Creating WebSearchTool and performing search...")
print()

tool = WebSearchTool()

# The original error was:
# DuckDuckGoSearchException: RuntimeError: error sending request for url
# (https://www.bing.com/search?q=Panthers+vs+Cowboys+game+result):
# client error (Connect) ... TLS handshake failed: cert verification failed
# - unable to get local issuer certificate [CERTIFICATE_VERIFY_FAILED]

try:
    results = tool.invoke({
        "q": "Panthers vs Cowboys game result",
        "max_results": 3
    })
    
    print("✅ ✅ ✅ SUCCESS! ✅ ✅ ✅")
    print()
    print(f"Search completed successfully with {len(results)} results.")
    print("No certificate verification errors!")
    print()
    
    if results:
        print("Sample results:")
        for i, result in enumerate(results[:2], 1):
            print(f"  {i}. {result['title'][:65]}...")
            print(f"     {result['href']}")
    print()
    print("="*70)
    print("The TLS certificate error has been COMPLETELY FIXED!")
    print("The web_search.py tool now works correctly with:")
    print("  - Proxy servers (Fiddler, etc.)")
    print("  - Custom CA bundles")
    print("  - ddgs 9.6.1+")
    print("="*70)
    sys.exit(0)
    
except Exception as e:
    error_msg = str(e)
    print("❌ ❌ ❌ FAILED! ❌ ❌ ❌")
    print()
    print(f"Error: {error_msg[:200]}")
    print()
    
    if "CERTIFICATE_VERIFY_FAILED" in error_msg or "cert verification failed" in error_msg:
        print("The CERTIFICATE ERROR is STILL PRESENT!")
        print("The fix did not work.")
    else:
        print("A different error occurred (not a certificate error).")
    
    sys.exit(1)
