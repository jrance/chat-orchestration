"""Debugging test - primp before and after DDGS."""
import os
from dotenv import load_dotenv
load_dotenv()

proxy = os.getenv('PROXY_URL')
ca_bundle = os.getenv('PROXY_CA_BUNDLE')

print("="*60)
print("STEP 1: Test primp.Client BEFORE any DDGS code")
print("="*60)
import primp

client1 = primp.Client(
    proxy=proxy,
    verify=True,
    ca_cert_file=ca_bundle,
    timeout=30,
)
print(f"Client 1 created: {client1}")

try:
    resp = client1.get("https://www.bing.com")
    print(f"✓ Client 1 test: SUCCESS (status {resp.status_code})")
except Exception as e:
    print(f"✗ Client 1 test: FAILED - {str(e)[:100]}")

print()
print("="*60)
print("STEP 2: Import and create DDGS")
print("="*60)
from duckduckgo_search import DDGS
ddgs = DDGS(proxy=proxy, verify=True)
print(f"DDGS created: {ddgs}")
print(f"DDGS inner client: {ddgs.client}")

print()
print("="*60)
print("STEP 3: Create new primp.Client AFTER DDGS")
print("="*60)
client2 = primp.Client(
    proxy=proxy,
    verify=True,
    ca_cert_file=ca_bundle,
    timeout=30,
)
print(f"Client 2 created: {client2}")

try:
    resp = client2.get("https://www.bing.com")
    print(f"✓ Client 2 test: SUCCESS (status {resp.status_code})")
except Exception as e:
    print(f"✗ Client 2 test: FAILED - {str(e)[:100]}")

print()
print("="*60)
print("STEP 4: Try client 1 again")
print("="*60)
try:
    resp = client1.get("https://www.bing.com")
    print(f"✓ Client 1 retest: SUCCESS (status {resp.status_code})")
except Exception as e:
    print(f"✗ Client 1 retest: FAILED - {str(e)[:100]}")
