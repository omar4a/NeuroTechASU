import os
import sys
import math
import asyncio
from dotenv import load_dotenv

# Inject Zeina_Branch into path to use _client.py
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.append(os.path.join(root_dir, "Zeina_Branch"))

import _client

async def test_predict_words():
    print("\n--- Tier 2: LLM API Stress Test (Self-Reported Probabilities Fallback) ---")
    
    # Mock context and history
    context = "BCI"
    history = "We are building a"
    partial_word = "BR"
    
    # We explicitly ask for probabilities since Groq doesn't support logprobs yet.
    system_prompt = (
        f"You are a predictive BCI typing assistant. The overarching context is '{context}'. "
        f"The user has already typed: '{history}'. They are currently spelling: '{partial_word}'. "
        "Output exactly 5 single-word predictions that complete the current partial word. "
        "For each word, provide your estimate of the probability (0.0 to 1.0) that it is the intended word. "
        "Output format: word1:prob1, word2:prob2, word3:prob3, word4:prob4, word5:prob5"
    )
    
    user_query = f"Partial word: {partial_word}"
    
    try:
        client = _client.get_client("SPELLER")
        model = _client._get_model_for_type("SPELLER")
        
        print(f"Using model: {model}")
        
        # Request
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.0 # Strict results
        )
        
        content = response.choices[0].message.content
        print(f"Raw Response: {content}")
        
        # Parse self-reported probabilities
        # Format expected: word:prob, word:prob...
        try:
            pairs = [p.strip().split(":") for p in content.split(",")]
            words = [p[0] for p in pairs]
            probs = [float(p[1]) for p in pairs]
            cumulative_prob = sum(probs)
            
            print(f"Parsed Predictions: {words}")
            print(f"Self-Reported Probabilities: {probs}")
            print(f"Cumulative Confidence: {cumulative_prob:.4f}")
            
            if cumulative_prob > 0.8:
                print("SUCCESS: High confidence threshold met.")
            else:
                print("INFO: Confidence threshold not met.")
                
        except Exception as e:
            print(f"WARNING: Failed to parse response format: {e}")
            
    except Exception as e:
        print(f"TEST FAILED: {e}")

if __name__ == "__main__":
    os.chdir(root_dir)
    asyncio.run(test_predict_words())
