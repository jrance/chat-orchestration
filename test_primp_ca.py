"""Test primp.Client with ca_cert_file parameter."""
import os
from pathlib import Path

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

ca_bundle = os.getenv('PROXY_CA_BUNDLE')
proxy = os.getenv('PROXY_URL')

print(f"CA Bundle: {ca_bundle}")
print(f"Proxy: {proxy}")
print(f"CA file exists: {Path(ca_bundle).exists()}")
print()

# Test 1: Create primp.Client with ca_cert_file
try:
    import primp
    
    print("Creating primp.Client with ca_cert_file...")
    client = primp.Client(
        proxy=proxy,
        verify=True,
        ca_cert_file=ca_bundle,
        timeout=30,
    )
    print(f"Client created successfully: {client}")
    print(f"Client type: {type(client)}")
    
    # Try a simple request
    print("\nAttempting HTTPS request through proxy...")
    response = client.get("https://www.bing.com")
    print(f"Response status: {response.status_code}")
    print("SUCCESS!")
    
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
