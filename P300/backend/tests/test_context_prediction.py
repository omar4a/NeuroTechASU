import os
import sys
import asyncio
from dotenv import load_dotenv

# Inject Zeina_Branch into path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.append(os.path.join(root_dir, "Zeina_Branch"))

import _client

async def test_context():
    print("\n--- Tier 5: Context-Aware Prediction Test ---")
    
    contexts = ["BCI", "Recipes", "Football"]
    partial = "CO"
    
    for ctx in contexts:
        print(f"\nTesting Context: {ctx}")
        system_prompt = (
            f"You are a predictive BCI typing assistant. Context: '{ctx}'. "
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
    asyncio.run(test_context())
