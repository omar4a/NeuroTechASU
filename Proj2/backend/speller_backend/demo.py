"""Demo-day constants. Specialized for BR41N.IO 2026 / NeurotechX ASU.

These are NOT generic defaults — they tune the speller and the ChatGPT reply
path for the specific 5-minute live demo on 25-26 April 2026. Swap at your
own risk; the LLM prompts are shaped to feel natural for the staged subject.
"""
from __future__ import annotations

# The single context string fed into task5_speller_api.predict_words for every
# prediction during the demo. This is what biases "HE" toward "HELLO" and
# "FR" toward "FROM"/"FRIEND" in the way the narrative wants.
DEMO_SPELLER_CONTEXT: str = (
    "The writer is a live demo subject at the BR41N.IO 2026 hackathon, "
    "wearing a g.tec Unicorn 8-channel EEG headset at Ain Shams University. "
    "They are spelling short, warm messages to ChatGPT through a brain-computer "
    "interface in front of judges. Predictions should favour natural, common "
    "English words a presenter would actually say on stage — greetings, "
    "introductions, gratitude, questions about AI, and simple self-expression."
)

# The persona ChatGPT adopts when it replies on stage.
DEMO_CHAT_SYSTEM_PROMPT: str = (
    "You are ChatGPT, talking live at the BR41N.IO 2026 hackathon with someone "
    "typing through a g.tec Unicorn EEG speller at Ain Shams University. "
    "Every word they type costs mental effort — be warm, concise, and "
    "genuinely interested. Reply in one or two short sentences. Never over-"
    "explain. You are aware that the NeurotechX ASU software team built this "
    "Brain↔ChatGPT pipeline and that you are the first ChatGPT the user is "
    "reaching with their mind. Match the subject's register; if they say "
    "HELLO, greet them back; if they ask a question, answer in plain English."
)

# Typewriter pacing on word-commit. Milliseconds between `type_char`
# emissions once a word is SSVEP-selected. 60ms ≈ human reading speed and
# feels like thought materialising; set to 0 to disable.
TYPEWRITER_INTERVAL_MS: int = 60

# How long we wait for a ChatGPT reply before giving up and showing a
# graceful fallback. Keep generous — the demo should survive a slow call.
CHAT_TIMEOUT_SECONDS: float = 12.0

# Time between backend startup and the pre-warm call to the LLM. Keep tiny —
# the goal is just to let the server finish binding its sockets first.
PREWARM_DELAY_SECONDS: float = 0.25
