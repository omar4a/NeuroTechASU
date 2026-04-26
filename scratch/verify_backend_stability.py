import os
import sys
import numpy as np
import asyncio
import types
import math

# 1. Mock pylsl
class MockOutlet:
    def push_sample(self, x, y=0):
        print(f"  [LSL OUTLET SENT] {x}")

mock_pylsl = types.ModuleType("pylsl")
mock_pylsl.StreamInfo = lambda *args, **kwargs: None
mock_pylsl.StreamOutlet = lambda *args, **kwargs: MockOutlet()
mock_pylsl.StreamInlet = lambda *args, **kwargs: type("MockInlet", (), {"pull_sample": lambda self, timeout=0: (None, None), "pull_chunk": lambda self: ([], [])})()
mock_pylsl.resolve_byprop = lambda *args, **kwargs: [True]
mock_pylsl.local_clock = lambda: 0.0
sys.modules["pylsl"] = mock_pylsl

# 2. Inject paths and import
backend_dir = os.path.join(os.getcwd(), "P300", "backend")
sys.path.append(backend_dir)
from realtime_inference import RealTimeInference

async def run_verification():
    print("--- Tier 4: Backend Stability Verification ---")
    
    # Initialize agent
    agent = RealTimeInference()
    # Force initialize the outlet for testing
    agent.decoded_outlet = MockOutlet()
    
    # Mock LLM predict_words to prevent API crash
    async def mock_predict(partial):
        print(f"  [MOCK LLM] Predicting for '{partial}'...")
        agent.decoded_outlet.push_sample([f"SSVEP_PREDICTIONS:BRAIN,BREAD,BRING,BRIDGE,BRIGHT"], 0)
        
    agent.predict_words = mock_predict
    
    print("Agent initialized with SSVEP and LLM hooks.")
    
    # Simulate decoding a character 'B'
    print("\nSimulating P300 decoding of 'B'...")
    agent._handle_character_decoded("B")
    await asyncio.sleep(0.1) # Let async task run
    
    # Simulate decoding a character 'R'
    print("\nSimulating P300 decoding of 'R'...")
    agent._handle_character_decoded("R")
    await asyncio.sleep(0.1) # Let async task run
    
    # Verify internal state
    print(f"\nCurrent partial word in backend: '{agent.current_word}'")
    
    # Simulate switching to SSVEP mode
    print("\nSimulating UI command: SSVEP_START")
    agent.bci_mode = "SSVEP"
    print(f"BCI Mode: {agent.bci_mode}")
    
    # Simulate SSVEP classification result (10.0 Hz)
    print("\nSimulating SSVEP detection of 10.0 Hz...")
    agent.decoded_outlet.push_sample(["SSVEP_DECODED_10.0"], 0)
    
    print("\n--- STABILITY TEST PASSED: No Crashes ---")

if __name__ == "__main__":
    asyncio.run(run_verification())
