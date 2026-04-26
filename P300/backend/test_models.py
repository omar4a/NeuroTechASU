"""Quick test of both SPELLER and RESPONSE models."""
import os, sys, time

os.environ['SPELLER_API_KEY'] = 'AIzaSyDGeBK4e5H2cLkS3VFEFW6AWHeFKyATjU0'
os.environ['SPELLER_API_BASE_URL'] = 'https://generativelanguage.googleapis.com/v1beta/openai/'
os.environ['SPELLER_MODEL'] = 'gemma-3-4b-it'
os.environ['RESPONSE_API_KEY'] = 'AIzaSyDGeBK4e5H2cLkS3VFEFW6AWHeFKyATjU0'
os.environ['RESPONSE_API_BASE_URL'] = 'https://generativelanguage.googleapis.com/v1beta/openai/'
os.environ['RESPONSE_MODEL'] = 'gemini-2.5-flash'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Zeina_Branch'))
sys.path.insert(0, os.path.dirname(__file__))

from _client import get_client, get_response

# Test RESPONSE model
print("--- Testing RESPONSE model (gemini-2.5-flash) ---")
client = get_client.__wrapped__("RESPONSE")
t0 = time.time()
resp = get_response(
    client,
    "You are a helpful assistant for a BCI speller user. Be concise.",
    'User said: "I NEED WATER"\nPlease respond naturally and helpfully.',
    llm_type="RESPONSE",
)
t1 = time.time()
print(f"Response ({(t1-t0)*1000:.0f}ms): {resp}")

# Test SPELLER model
print("\n--- Testing SPELLER model (gemma-3-4b-it) ---")
client2 = get_client.__wrapped__("SPELLER")
t0 = time.time()
resp2 = get_response(
    client2,
    'You are a word completion assistant. Given a prefix, return the 3 most likely English words. Return ONLY valid JSON: {"predictions": [{"word": "word1", "prob": 0.8}, ...]}',
    "prefix: HE",
    llm_type="SPELLER",
)
t1 = time.time()
print(f"Response ({(t1-t0)*1000:.0f}ms): {resp2}")
