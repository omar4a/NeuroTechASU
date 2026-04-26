import os
import sys
import asyncio
import numpy as np

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root_dir, "Zeina_Branch"))

import _client

async def run_calibration():
    print("=== Advanced LLM Calibration Test ===")
    
    # We test cases where there are MANY equally likely completions
    # A perfectly calibrated model should output ~0.2 for the top 5
    scenarios = [
        {"context": "Names", "history": "My name is", "prefix": "MA"}, 
        # MA -> Matt, Mark, Mary, Max, Mason, etc.
        {"context": "Colors", "history": "My favorite color is", "prefix": "B"},
        # B -> Blue, Black, Brown, Beige, Burgundy
    ]
    
    client = _client.get_client("SPELLER")
    model = _client._get_model_for_type("SPELLER")
    
    for i, s in enumerate(scenarios):
        print(f"\nScenario {i+1}: Context='{s['context']}', Prefix='{s['prefix']}'")
        
        system_prompt = (
            f"You are a predictive BCI typing assistant. Context: '{s['context']}'. "
            f"Current sentence: '{s['history']}'. Current word fragment: '{s['prefix']}'. "
            "Output exactly 5 single-word completion suggestions as a comma-separated list in 'word:prob' format. "
            "The probabilities MUST be between 0.0 and 1.0. The sum of the 5 probabilities should reflect your total confidence."
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": f"Complete: {s['prefix']}"}],
            temperature=0.0
        )
        
        content = response.choices[0].message.content
        pairs = [p.strip().split(":") for p in content.split(",")]
        probs = []
        words = []
        for p in pairs:
            if len(p) >= 2:
                words.append(p[0].strip())
                probs.append(float(p[1].strip()))
                
        print(f"  Distribution: {list(zip(words, probs))}")
        
        # Check if the model is artificially spiking
        # If it gives 0.8 to one name and 0.05 to the rest, it is poorly calibrated (overconfident)
        if len(probs) > 0 and probs[0] > 0.6:
            print(f"  [POOR CALIBRATION DETECTED] The model is artificially overconfident in '{words[0]}'. True probability should be closer to 0.2")
        else:
            print("  [GOOD CALIBRATION] The model correctly distributed probabilities for ambiguous inputs.")

if __name__ == "__main__":
    asyncio.run(run_calibration())
