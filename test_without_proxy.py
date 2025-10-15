"""Test without proxy."""
import os

# Disable proxy
os.environ['PROXY_ENABLED'] = 'False'

from dotenv import load_dotenv
load_dotenv()

from codeless_orchestrator.tools.builtin.web_search import WebSearchTool

tool = WebSearchTool()
results = tool.invoke({"q": "Python", "max_results": 2})
print(f"Without proxy: {len(results)} results")
if results:
    print(f"First result: {results[0].get('title', 'N/A')}")
