"""Test if importing DDGS affects primp."""
import os
from dotenv import load_dotenv
load_dotenv()

proxy = os.getenv('PROXY_URL')
ca_bundle = os.getenv('PROXY_CA_BUNDLE')

# Test 1: primp BEFORE importing DDGS
print("TEST 1: primp.Client BEFORE importing DDGS")
import primp
client1 = primp.Client(proxy=proxy, verify=True, ca_cert_file=ca_bundle, timeout=30)
try:
    resp = client1.get("https://www.bing.com")
    print(f"  Result: SUCCESS (status {resp.status_code})")
except Exception as e:
    print(f"  Result: FAILED - {str(e)[:100]}")

print()

# Test 2: primp AFTER importing DDGS
print("TEST 2: Importing DDGS...")
from duckduckgo_search import DDGS
print("  DDGS imported")

print("\nTEST 3: Creating primp.Client AFTER importing DDGS")
client2 = primp.Client(proxy=proxy, verify=True, ca_cert_file=ca_bundle, timeout=30)
try:
    resp = client2.get("https://www.bing.com")
    print(f"  Result: SUCCESS (status {resp.status_code})")
except Exception as e:
    print(f"  Result: FAILED - {str(e)[:100]}")

print()

# Test 3: Try the first client again
print("TEST 4: Retrying first client (created before DDGS import)")
try:
    resp = client1.get("https://www.bing.com")
    print(f"  Result: SUCCESS (status {resp.status_code})")
except Exception as e:
    print(f"  Result: FAILED - {str(e)[:100]}")

