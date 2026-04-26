import os
import sys
import asyncio
import json
import numpy as np

# Inject Zeina_Branch into path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root_dir, "Zeina_Branch"))

import _client

async def run_evaluation():
    print("=== Rigorous Evaluation of Self-Reported Probabilities ===")
    
    scenarios = [
        # 1. Unambiguous prefix
        {"context": "Animals", "history": "I saw a huge", "prefix": "ELEPH", "expected": "ELEPHANT"},
        # 2. Highly ambiguous prefix
        {"context": "General", "history": "I want to", "prefix": "T", "expected": "THE/TO/THAT"},
        # 3. Typo in prefix (meant BREAD, typed BRX)
        {"context": "Food", "history": "I ate some", "prefix": "BRX", "expected": "BREAD"},
        # 4. Out of context word
        {"context": "Space", "history": "The rocket", "prefix": "L", "expected": "LAUNCH"}
    ]
    
    client = _client.get_client("SPELLER")
    model = _client._get_model_for_type("SPELLER")
    
    results = []
    
    for i, s in enumerate(scenarios):
        print(f"\nScenario {i+1}: Context='{s['context']}', Prefix='{s['prefix']}'")
        
        system_prompt = (
            f"You are a predictive BCI typing assistant. Context: '{s['context']}'. "
            f"Current sentence: '{s['history']}'. Current word fragment: '{s['prefix']}'. "
            "Output exactly 5 single-word completion suggestions as a comma-separated list in 'word:prob' format. "
            "The probabilities MUST be between 0.0 and 1.0. The sum of the 5 probabilities should reflect your total confidence."
        )
        
        # Run 5 iterations per scenario to check consistency/determinism at Temp=0
        for iteration in range(3):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": f"Complete: {s['prefix']}"}],
                temperature=0.0
            )
            
            content = response.choices[0].message.content
            
            # Parse
            pairs = [p.strip().split(":") for p in content.split(",")]
            probs = []
            words = []
            try:
                for p in pairs:
                    if len(p) >= 2:
                        words.append(p[0].strip())
                        probs.append(float(p[1].strip()))
            except Exception as e:
                print(f"  [Iter {iteration}] Parse Error: {e} | Raw: {content}")
                continue
                
            total_mass = sum(probs)
            max_prob = max(probs) if probs else 0
            
            print(f"  [Iter {iteration}] Top: {words[0] if words else 'N/A'} ({probs[0] if probs else 0}), Total Mass: {total_mass:.2f}, Distribution: {probs}")
            
            results.append({
                "scenario": i,
                "iter": iteration,
                "words": words,
                "probs": probs,
                "total_mass": total_mass,
                "max_prob": max_prob
            })

    # Mathematical Soundness Analysis
    print("\n--- Statistical Analysis ---")
    masses = [r['total_mass'] for r in results]
    print(f"Average Total Probability Mass: {np.mean(masses):.2f} (Should theoretically be <= 1.0)")
    print(f"Max Probability Mass Output: {np.max(masses):.2f}")
    
    overconfident_cases = sum(1 for r in results if r['total_mass'] > 1.01)
    if overconfident_cases > 0:
        print(f"WARNING: Model generated probabilities summing to > 1.0 in {overconfident_cases}/{len(results)} tests. (Mathematically invalid)")
    
    # Check Variance (At Temp 0, it should be identical every time)
    scen_0_probs = [r['probs'] for r in results if r['scenario'] == 0]
    if len(scen_0_probs) > 1:
        # Pad to equal length before comparing
        max_len = max(len(p) for p in scen_0_probs)
        arr0 = np.pad(scen_0_probs[0], (0, max_len - len(scen_0_probs[0])))
        arr1 = np.pad(scen_0_probs[1], (0, max_len - len(scen_0_probs[1])))
        diff = np.max(np.abs(arr0 - arr1))
        if diff > 0:
            print(f"WARNING: Temperature 0 generated different probabilities (Max diff: {diff}). Non-deterministic math.")
        else:
            print("INFO: Responses were deterministic at Temp 0.")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
