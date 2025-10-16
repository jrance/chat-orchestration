"""Test if creating a DDGS instance affects subsequent primp.Client creation."""
import os
from dotenv import load_dotenv
load_dotenv()

import primp
from duckduckgo_search import DDGS

proxy = os.getenv('PROXY_URL')
ca_bundle = os.getenv('PROXY_CA_BUNDLE')

print("="*60)
print("STEP 1: Create DDGS instance FIRST")
print("="*60)
ddgs = DDGS(proxy=proxy, verify=True)
print(f"DDGS created: {ddgs}")
print(f"DDGS client: {ddgs.client}")

print()
print("="*60)
print("STEP 2: Create primp.Client AFTER DDGS")
print("="*60)
client = primp.Client(
    proxy=proxy,
    verify=True,
    ca_cert_file=ca_bundle,
    timeout=10,
)
print(f"Client created: {client}")

print()
print("="*60)
print("STEP 3: Test the primp.Client")
print("="*60)
try:
    resp = client.get("https://www.bing.com")
    print(f"[OK] Client test SUCCEEDED (status {resp.status_code})")
except Exception as e:
    print(f"[FAIL] Client test FAILED: {str(e)[:150]}")

