"""Test creating DDGS directly and then manually replacing its client."""
import os
from dotenv import load_dotenv
load_dotenv()

import primp
from duckduckgo_search import DDGS

proxy = os.getenv('PROXY_URL')
ca_bundle = os.getenv('PROXY_CA_BUNDLE')

print(f"DEBUG: proxy = {repr(proxy)}")
print(f"DEBUG: ca_bundle = {repr(ca_bundle)}")

print("Creating DDGS with verify=True...")
ddgs = DDGS(proxy=proxy, verify=True)

print(f"DDGS client before: {ddgs.client}")
print()

# Now replace the client
print("Replacing client with one that has ca_cert_file...")
new_client = primp.Client(
    proxy=proxy,
    verify=True,
    ca_cert_file=ca_bundle,
    timeout=10,
)

print(f"New client: {new_client}")

# Test the new client directly first
print("\nTesting new client directly...")
try:
    resp = new_client.get("https://www.bing.com")
    print(f"Direct client test: SUCCESS (status {resp.status_code})")
except Exception as e:
    print(f"Direct client test: FAILED - {e}")

# Now replace it in DDGS
ddgs.client = new_client
print(f"\nDDGS client after: {ddgs.client}")

# Try a search
print("\nTrying DDGS search with backend='bing'...")
try:
    results = ddgs.text(keywords="Python", backend="bing", max_results=2)
    print(f"SUCCESS! Got {len(results)} results")
    for r in results:
        print(f"  - {r.get('title', 'N/A')}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}")
    print(f"Error: {e}")

