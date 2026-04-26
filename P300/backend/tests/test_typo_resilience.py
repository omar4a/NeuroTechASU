import os
import sys
import asyncio
from dotenv import load_dotenv

# Inject Zeina_Branch into path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.append(os.path.join(root_dir, "Zeina_Branch"))

import _client

async def test_typo():
    print("\n--- Tier 6: Typo Resilience Test ---")
    
    context = "Neuroscience"
    # User meant 'BRAIN' but typed 'BRX'
    partial = "BRX"
    
    print(f"Context: {context} | Partial (with typo): {partial}")
    
    system_prompt = (
        f"You are a predictive BCI typing assistant. Context: '{context}'. "
        "The user is typing one letter at a time via a BCI keyboard. BCI decoding often has errors. "
        "If the current partial word has a typo, suggest the most likely intended corrections. "
        "Output exactly 5 single-word completion suggestions as a comma-separated list in 'word:prob' format."
    )
    
    try:
        client = _client.get_client("SPELLER")
        model = _client._get_model_for_type("SPELLER")
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": f"Complete: {partial}"}],
            temperature=0.0
        )
        print(f"  Results for '{partial}': {response.choices[0].message.content}")
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    os.chdir(root_dir)
    asyncio.run(test_typo())
