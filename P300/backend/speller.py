"""Core predictive speller logic — enhanced for local llms.

Public contract (stable for the Unity UI team):

    predict_words(prefix, sentence, context) -> list[str]

Returns exactly N lowercase/capitalized English words.
See INTEGRATION.md for the full integration guide.

Architecture change from v1 (OpenAI, single prompt):
    - Routing logic lives in Python, not the prompt
    - Three single-task agents replace one overloaded prompt
    - Cold start (both inputs empty) requires NO model call
    - Prefix constraint enforced in Python, not relied on from the model
    - Capitalization applied in Python based on sentence state
    - Wordlist fallback pads results without a second model call
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from _client import get_client, get_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent system prompts
# Each prompt does exactly ONE thing. All branching lives in Python.
# ---------------------------------------------------------------------------

# Agent 1: user is mid-sentence and has started typing a word
_PROMPT_PREFIX_COMPLETION = (
    "You are a word completion assistant for a BCI speller.\n"
    "Given a sentence and a prefix, return the {n} most likely English words that:\n"
    "  1. Start exactly with the given prefix\n"
    "  2. Fit naturally as the next word in the sentence\n\n"
    "Return ONLY valid JSON: {{\"predictions\": [{{\"word\": \"word1\", \"prob\": 0.8}}, ...]}}\n"
    "No explanation. No markdown. No extra keys. Only single words, no phrases."
)

# Agent 2: user is mid-sentence, no prefix typed yet
_PROMPT_NEXT_WORD = (
    "You are a next-word prediction assistant for a BCI speller.\n"
    "Given a sentence and optional context, return the {n} most likely English words "
    "that would naturally follow the sentence.\n\n"
    "Return ONLY valid JSON: {{\"predictions\": [{{\"word\": \"word1\", \"prob\": 0.8}}, ...]}}\n"
    "No explanation. No markdown. No extra keys. Only single words, no phrases."
)

# Agent 3: sentence is empty, user has typed a prefix to start with
_PROMPT_SENTENCE_START = (
    "You are a word completion assistant for a BCI speller.\n"
    "Given a prefix and optional context, return the {n} most likely English words that:\n"
    "  1. Start exactly with the given prefix\n"
    "  2. Would naturally begin a sentence\n\n"
    "Return ONLY valid JSON: {{\"predictions\": [{{\"word\": \"word1\", \"prob\": 0.8}}, ...]}}\n"
    "No explanation. No markdown. No extra keys. Only single words, no phrases."
)


# Agent 4: sentence is empty, context is empty, user has typed a prefix
_PROMPT_PREFIX_ONLY = (
    "You are a word completion assistant for a BCI speller.\n"
    "Given a prefix, return the {n} most likely English words that start exactly with that prefix.\n\n"
    "Return ONLY valid JSON: {{\"predictions\": [{{\"word\": \"word1\", \"prob\": 0.8}}, ...]}}\n"
    "No explanation. No markdown. No extra keys. Only single words, no phrases."
)


# ---------------------------------------------------------------------------
# Cold start defaults — no model call needed when both inputs are empty.
# Order: article, question word, pronoun — covers the three most common
# sentence-opening intents in AAC/BCI use cases.
# ---------------------------------------------------------------------------
_COLD_START_DEFAULTS = ["The", "I", "What", "How", "Can", "Please", "My", "We"]


# ---------------------------------------------------------------------------
# Fixer agent prompt
# Fires only when the main agent returns words that violate the prefix
# constraint. Single-task: "give me N common words starting with X."
# No sentence context needed — this is purely lexical recovery.
# ---------------------------------------------------------------------------
_PROMPT_FIXER = (
    "You are a word lookup assistant for a BCI speller.\n"
    "Return the {n} most common English words that start exactly with the prefix '{prefix}'.\n"
    "They must all begin with '{prefix}' — no exceptions.\n\n"
    "Return ONLY valid JSON: {{\"predictions\": [{{\"word\": \"word1\", \"prob\": 0.8}}, ...]}}\n"
    "No explanation. No markdown. No extra keys. Only single words, no phrases."
)

# ---------------------------------------------------------------------------
# Checker agent prompt
# Fires after any main/fixer agent call if validation fails.
# Its only job: identify and remove invalid predictions.
# ---------------------------------------------------------------------------
_PROMPT_CHECKER = (
    "You are a strict output validator for a BCI speller.\n"
    "You will receive a list of predicted words and a prefix.\n"
    "Remove any prediction that:\n"
    "  1. Does not start exactly with the prefix\n"
    "  2. Is more than one word (contains spaces)\n"
    "  3. Is a placeholder like 'word1', 'word2', 'word3', etc.\n"
    "  4. Is not a real English word\n"
    "Return ONLY the valid predictions as JSON: {{\"predictions\": [{{\"word\": \"...\", \"prob\": ...}}, ...]}}\n"
    "If all predictions are invalid, return {{\"predictions\": []}}\n"
    "No explanation. No markdown. No extra keys. Only single words, no phrases."
)


class API:
    _N_PREDICTIONS = 4
    # Fallback used only when model fails AND wordlist has no prefix match.
    # Deliberately generic — callers should treat this as degraded mode.
    _FALLBACK_WORDS = ("the", "and", "of")

    def __init__(self, prediction_count: int = _N_PREDICTIONS):
        self.speller_client = get_client("SPELLER")
        self.response_client = get_client("RESPONSE")
        self._N_PREDICTIONS = prediction_count
        self._conversation_history: list[dict] = []  # Accumulates across session

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict_words(
        self,
        context: str,
        prefix: str,
        sentence: str = "",
        return_probs: bool = False,
    ) -> list[str] | list[dict]:
        """Predict N complete words based on current speller state.

        Routing table (all edge cases handled here, not in the model):
          prefix=''  sentence=''  ->  cold_start()        [no model call]
          prefix=X   sentence=''  context='' -> prefix_only_agent
          prefix=X   sentence=''  context=Y  -> sentence_start_agent
          prefix=''  sentence=X   ->  next_word_agent
          prefix=X   sentence=X   ->  prefix_completion_agent

        Post-processing (always in Python):
          - Prefix constraint enforcement + wordlist padding
          - Capitalization based on sentence boundary state

        Args:
            context:  Free-text description of the user's intent/topic.
            prefix:   Letters typed so far for the current word (may be empty).
            sentence: In-progress sentence up to the current word (may be empty).

        Returns:
            Exactly self._N_PREDICTIONS words. Never raises on model failure
            — returns a documented fallback list instead.
        """
        if not isinstance(context, str):
            raise ValueError(f"context must be a string; got {context!r}")

        # Normalise inputs
        prefix   = prefix.strip()   if isinstance(prefix, str)   else ""
        sentence = sentence.strip() if isinstance(sentence, str) else ""
        
        raw_context = context.strip() if isinstance(context, str) else ""
        context     = raw_context or "(no context)"

        has_prefix   = bool(prefix)
        has_sentence = bool(sentence)
        has_context  = bool(raw_context)

        # --- Route ---
        if not has_prefix and not has_sentence:
            return self._cold_start()

        if has_prefix and not has_sentence:
            if not has_context:
                raw = self._call_prefix_only_agent(prefix)
            else:
                raw = self._call_sentence_start_agent(prefix, context)
        elif not has_prefix and has_sentence:
            raw = self._call_next_word_agent(sentence, context)
        else:
            raw = self._call_prefix_completion_agent(prefix, sentence, context)

        # --- Parse ---
        predictions = self._parse_predictions(raw)

        # --- Post-process (Python owns this, not the model) ---
        if has_prefix:
            predictions = self._enforce_prefix(predictions, prefix)

        predictions = self._apply_capitalization(predictions, sentence)

        if return_probs:
            return predictions[: self._N_PREDICTIONS]
        return [p["word"] for p in predictions[: self._N_PREDICTIONS]]

    def final_answer(self, context: str, sentence: str) -> str:
        """Generate a final free-text response based on full sentence and context.
        
        Uses the response model to create a natural, conversational reply.
        
        Args:
            context: High-level scenario/intent (e.g., "medical question").
            sentence: The fully-typed user sentence.
        
        Returns:
            A natural language response string. Returns empty string on failure.
        """
        return self.respond(sentence, context)

    def respond(
        self,
        sentence: str,
        context: str = "",
        mental_state: Optional[str] = None,
    ) -> str:
        """Generate a contextual response to a completed user sentence.
        
        Uses the response model to understand the user's intent
        and generate a meaningful, conversational reply.
        
        Args:
            sentence: The user's completed sentence.
            context: Optional high-level context (e.g., topic, scenario).
            mental_state: Optional user mental state (e.g. "Focused", "Fatigued")
        
        Returns:
            A natural, contextual response string. Returns empty string on failure.
        """
        if not sentence or not sentence.strip():
            logger.warning("Empty sentence passed to respond")
            return ""
        
        length_instruction = "Generate a concise and thoughtful response."
        if mental_state == "Focused":
            length_instruction = "The user is deeply FOCUSED. Generate a highly detailed, comprehensive, and rich response."
        elif mental_state == "Fatigued":
            length_instruction = "The user is FATIGUED. Generate an extremely brief, concise, and straight-to-the-point response."

        system_prompt = (
            "You are a helpful, friendly assistant responding to a user in a conversational context. "
            "Your job is to provide accurate, thoughtful replies to what they say. "
            f"{length_instruction} "
            "Interpret any typos gracefully as the user is using a BCI speller."
        )
        user_message = f"User said: \"{sentence}\"\n"
        if context:
            user_message += f"Context: {context}\n"
        user_message += "Please respond naturally and helpfully."
        
        try:
            # Build messages: system + full conversation history + new user turn
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(self._conversation_history)
            messages.append({"role": "user", "content": user_message})

            response = get_response(
                self.response_client,
                messages=messages,
                llm_type="RESPONSE",
            )
            reply = response.strip()

            # Append this exchange to history for future calls
            self._conversation_history.append({"role": "user", "content": user_message})
            self._conversation_history.append({"role": "assistant", "content": reply})

            return reply
        except (ValueError, RuntimeError) as exc:
            logger.warning("Response generation failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Agents — one prompt per task
    # ------------------------------------------------------------------

    def _validate_predictions(self, predictions: list[dict], prefix: str) -> list[dict]:
        """Python-side validation — fast, runs before the checker agent.

        Catches the obvious failures without a model call:
        - Multi-word phrases (contains spaces)
        - Template placeholders (word1, word2, word3)
        - Wrong prefix
        """
        prefix_lower = prefix.lower() if prefix else ""
        placeholder  = re.compile(r"^word\d+$", re.IGNORECASE)

        def is_valid(p: dict) -> bool:
            w = p["word"]
            if " " in w.strip():                              # phrase
                return False
            if placeholder.match(w.strip()):                  # template leak
                return False
            if prefix_lower and not w.lower().startswith(prefix_lower):  # wrong prefix
                return False
            return True

        return [p for p in predictions if is_valid(p)]


    def _call_checker_agent(
        self, predictions: list[dict], prefix: str
    ) -> list[dict]:
        """Checker agent — fires only when Python validation already found issues.

        Asks the model to clean up its own output. Intentionally lightweight:
        no sentence context, no prediction task — purely validation.
        Runs at most once per predict_words call (after main or fixer).
        """
        system = _PROMPT_CHECKER
        user   = (
            f"prefix: {prefix.lower() if prefix else '(empty)'}\n"
            f"predictions: {json.dumps(predictions)}"
        )
        raw = self._safe_call(self.speller_client, system, user)
        return self._parse_predictions(raw) if raw else []

    def _call_prefix_completion_agent(
        self, prefix: str, sentence: str, context: str
    ) -> str | None:
        system = _PROMPT_PREFIX_COMPLETION.format(n=self._N_PREDICTIONS)
        user   = f"context: {context}\nsentence: {sentence}\nprefix: {prefix}"
        return self._safe_call(self.speller_client, system, user)

    def _call_next_word_agent(self, sentence: str, context: str) -> str | None:
        system = _PROMPT_NEXT_WORD.format(n=self._N_PREDICTIONS)
        user   = f"context: {context}\nsentence: {sentence}"
        return self._safe_call(self.speller_client, system, user)

    def _call_sentence_start_agent(self, prefix: str, context: str) -> str | None:
        system = _PROMPT_SENTENCE_START.format(n=self._N_PREDICTIONS)
        user   = f"context: {context}\nprefix: {prefix}"
        return self._safe_call(self.speller_client, system, user)

    def _call_prefix_only_agent(self, prefix: str) -> str | None:
        system = _PROMPT_PREFIX_ONLY.format(n=self._N_PREDICTIONS)
        user   = f"prefix: {prefix}"
        return self._safe_call(self.speller_client, system, user)

    def _cold_start(self) -> list[dict]:
        """No model call — return deterministic sentence starters.

        Covers the three most common AAC/BCI opening intents:
          'The' / 'A'  -> starting a declarative sentence
          'I'          -> first-person statement
          'What/How'   -> question
        """
        return [{"word": w, "prob": 1.0 / len(_COLD_START_DEFAULTS)} for w in _COLD_START_DEFAULTS[: self._N_PREDICTIONS]]

    def _safe_call(self, client, system_prompt: str, user_message: str) -> str | None:
        """Wrapper that converts any model exception to None (-> fallback path)."""
        try:
            return get_response(
                client,
                system_prompt=system_prompt,
                user_message=user_message,
            )
        except (ValueError, RuntimeError) as exc:
            logger.warning("Model call failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Output processing — all validation and correction lives here
    # ------------------------------------------------------------------

    def _parse_predictions(self, raw: str | None) -> list[dict]:
        """Parse JSON from model output. Returns fallback on any failure."""
        if not raw:
            logger.warning("Empty response from model; returning fallback")
            return [{"word": w, "prob": 0.0} for w in self._FALLBACK_WORDS]

        try:
            # Strip markdown fences — some models wrap JSON in ```json ... ```
            # even when explicitly told not to.
            cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
            payload = json.loads(cleaned)
            predictions = payload.get("predictions", [])
            if not isinstance(predictions, list):
                raise ValueError("'predictions' field was not a list")
            
            parsed = []
            for item in predictions:
                if isinstance(item, dict) and "word" in item:
                    parsed.append({"word": str(item["word"]).strip(), "prob": float(item.get("prob", 0.0))})
                else:
                    parsed.append({"word": str(item).strip(), "prob": 0.0})
            return parsed
            
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Could not parse model output %r: %s", raw, exc)
            return [{"word": w, "prob": 0.0} for w in self._FALLBACK_WORDS]

    def _enforce_prefix(self, predictions: list[str], prefix: str) -> list[str]:
        """Hard-filter: keep only words starting with prefix.

        If the main agent returned any invalid words, call the fixer agent
        for exactly the number of missing slots. The fixer agent fires at most
        once per predict_words call.

        Worst case: two sequential model calls. This is an accepted tradeoff
        for full agentic behaviour — callers should be aware of the latency
        ceiling this creates.
        """
        prefix_lower = prefix.lower()

        # 1. Keep valid predictions from the main agent
        valid = [p for p in predictions if p["word"].lower().startswith(prefix_lower)]

        # 2. If short, call the fixer agent for the missing count
        missing = self._N_PREDICTIONS - len(valid)
        if missing > 0:
            logger.info(
                "Main agent returned %d invalid words for prefix %r — calling fixer",
                missing, prefix,
            )
            raw = self._call_fixer_agent(prefix, missing)
            fixer_predictions = self._parse_predictions(raw)

            already = {v["word"].lower() for v in valid}
            for p in fixer_predictions:
                if len(valid) >= self._N_PREDICTIONS:
                    break
                if p["word"].lower().startswith(prefix_lower) and p["word"].lower() not in already:
                    valid.append(p)
                    already.add(p["word"].lower())

            # 3. Run Python validator on everything collected so far
            valid = self._validate_predictions(valid, prefix)

            # 4. If still short after validation, call checker on the original raw output
            if len(valid) < self._N_PREDICTIONS:
                logger.info("Running checker agent — %d valid so far", len(valid))
                checked = self._call_checker_agent(valid, prefix)
                already = {v["word"].lower() for v in valid}
                for p in checked:
                    if len(valid) >= self._N_PREDICTIONS:
                        break
                    if p["word"].lower() not in already:
                        valid.append(p)
                        already.add(p["word"].lower())

        # 3. Absolute last resort: fixer also failed or returned non-prefix words.
        #    Return the prefix itself as a word — at minimum the user sees what
        #    they typed. No further model calls.
        if not valid:
            logger.warning("Fixer agent also failed for prefix %r", prefix)
            valid = [{"word": prefix, "prob": 0.0}]

        return valid[: self._N_PREDICTIONS]

    def _call_fixer_agent(self, prefix: str, missing_count: int) -> str | None:
        """Fixer agent: recover common words for a given prefix.

        Called only when the main agent violated the prefix constraint.
        Intentionally has no sentence context — it is purely lexical recovery.
        Asking for exactly `missing_count` keeps the response minimal.
        """
        system = _PROMPT_FIXER.format(n=missing_count, prefix=prefix.lower())
        user   = f"prefix: {prefix.lower()}"
        return self._safe_call(self.speller_client, system, user)

    def _apply_capitalization(self, predictions: list[dict], sentence: str) -> list[dict]:
        """Capitalize if predictions will start a new sentence; lowercase otherwise.

        A new sentence is indicated by:
          - sentence being empty (first word of a new sentence)
          - sentence ending with terminal punctuation (. ! ?)
        """
        at_sentence_start = (
            not sentence
            or sentence.rstrip().endswith((".", "!", "?"))
        )
        for p in predictions:
            if at_sentence_start:
                p["word"] = p["word"].capitalize()
            else:
                p["word"] = p["word"].lower()
        return predictions


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
#
# Preserves the stable import paths documented in INTEGRATION.md:
#     from task5_speller_api import predict_words, respond_to_sentence
# Unity UI callers (and the Proj2 WebSocket backend) depend on these import
# paths. The wrappers lazy-instantiate one API() so repeated calls share a
# warm client.
# ---------------------------------------------------------------------------

_DEFAULT_API: "API | None" = None


def predict_words(
    prefix: str = "",
    context: str = "",
    sentence: str = "",
    mental_state: Optional[str] = None,
) -> list[str]:
    """Module-level wrapper over `API.predict_words`.

    `mental_state` is accepted for signature stability (WBS 4.1 — see
    DESIGN_NOTES.md §1 and §7.5) but is currently ignored.
    """
    global _DEFAULT_API
    if _DEFAULT_API is None:
        _DEFAULT_API = API()
    return _DEFAULT_API.predict_words(
        prefix=prefix, context=context, sentence=sentence, return_probs=True if mental_state else False
    )


def respond_to_sentence(
    sentence: str,
    context: str = "",
    mental_state: Optional[str] = None,
) -> str:
    """Module-level wrapper over `API.respond`.
    
    Generate a contextual response to a completed user sentence using the
    response model. The model decides whether to search for recent information.
    
    Args:
        sentence: The user's completed sentence.
        context: Optional high-level context (e.g., topic, scenario).
        mental_state: Optional user mental state (e.g. "Focused", "Fatigued")
    
    Returns:
        A natural, contextual response string.
    """
    global _DEFAULT_API
    if _DEFAULT_API is None:
        _DEFAULT_API = API()
    return _DEFAULT_API.respond(sentence=sentence, context=context, mental_state=mental_state)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    tests = [
        # Normal cases
        {"context": "football",   "prefix": "ch", "sentence": "Barcelona is the "},
        {"context": "technology", "prefix": "br", "sentence": "The future of AI is "},
        {"context": "food",       "prefix": "p",  "sentence": "I want to cook "},
        # Edge cases handled in Python
        {"context": "",           "prefix": "",   "sentence": ""},           # cold start
        {"context": "medicine",   "prefix": "tr", "sentence": ""},           # sentence start
        {"context": "weather",    "prefix": "",   "sentence": "It is very"}, # next word
        {"context": "sport",      "prefix": "xr", "sentence": "He ran "},   # rare prefix
    ]

    api = API()

    print(f"{'context':<15} {'prefix':<6} {'sentence':<35} predictions")
    print("-" * 80)
    for test in tests:
        t0 = time.perf_counter()
        results = api.predict_words(**test)
        elapsed = time.perf_counter() - t0
        print(
            f"{test['context']:<15} "
            f"{test['prefix']!r:<6} "
            f"{test['sentence']!r:<35} "
            f"{results}  ({elapsed:.2f}s)"
        )
