# trial & error

Running journal of design decisions we're testing before demo day. Each entry
is a hypothesis → what we tried → what we'll verify on hardware. Add to the
top; leave old entries in place.

---

## 2026-04-20 · Continuation mode without a 4th SSVEP frequency

### The flow we want

```
P300 → LLM → SSVEP → [LLM → SSVEP] → [LLM → SSVEP] …
         ^                                         |
         |          if SSVEP can't lock            |
         +—— fallback to P300 for fresh prefix ——+
```

**First word:** subject enters 1–2 letters via P300, the LLM returns 3 completions, the subject picks one with SSVEP. This is the classic speller beat.

**Every subsequent word:** skip P300 entirely. Call `predict_words(prefix="", sentence=<so-far>, context=<demo persona>)` — Mohamed's prompt already handles empty-prefix as "next word from context + sentence". Three continuation candidates flicker at 10/12/15 Hz; subject looks at the one they want.

**If the subject doesn't want any of the three continuations**, they look *away* (at the grid, at the typed text, anywhere that's not a flickering box). The SSVEP classifier fails to lock within a timeout and the system falls back to P300 for a fresh prefix.

No 4th SSVEP target, no 4-class retraining, no fourth flickering box on the UI.

### Why pure timeout alone DOESN'T work (the non-obvious part)

CCA/FBCCA is an `argmax` operation: given a window, it always returns one of the target frequencies. It doesn't have a "no decision" output. So if the subject looks at a wall, the classifier still reports some frequency every second — noise-dominated but non-null.

Worse, looking at the upstream `SSVEP Protocol/ssvep_realtime.py` (Omar's, commit 5a11c01): `LSLStreamer.current_display_pred` is set once a majority vote reaches count > 2, **and is never cleared**. After lock-in, every subsequent window fires the same `callback("prediction", …)` — stale value keeps reappearing.

So if we naively add `await ssvep.next_prediction(timeout=5.0)` and wait for a None, we'll almost never see a None on live hardware — the queue will keep receiving stale predictions from the stuck classifier state.

### What actually separates "picked" from "didn't pick"

Three real signals, combined:

1. **Top-correlation threshold.**
   `classify_cca()` already returns `(pred, corrs)` — we take `top = max(corrs)`. When the subject attends a target, `top` is roughly 0.4–0.6 for CCA (higher for FBCCA). When they look away, all three correlations hover around 0.15–0.25 and `top` stays below ~0.3. **Reject below-threshold picks before emitting.**

2. **Margin check.**
   `margin = top - second_max(corrs)`. When the subject clearly fixates one target, the margin is big. When they're not, all three cluster and margin is tiny. **Require margin > ~0.05.**

3. **Per-cycle classifier reset.**
   After a word commits, **clear** `LSLStreamer.history` and reset `current_display_pred = "--"`. Without this, the stuck-lock bug above pollutes the next cycle.

### Changes required

**Upstream** (Omar's `SSVEP Protocol/ssvep_realtime.py`):

- Add `LSLStreamer.reset()` that empties `history` and sets `current_display_pred = "--"`.
- Richer callback payload. Today: `callback("prediction", freq)`. Proposed: `callback("prediction", {"freq": freq, "top_corr": float, "margin": float})`.
- Before firing the callback, apply threshold + margin checks (or let the consumer do the filtering — simpler upstream).
- ~15 lines.

**Backend** (`Proj2/backend/speller_backend/ssvep_consumer.py`):

- `SSVEPConsumer` protocol gains `async reset()`.
- `RealSSVEPConsumer.reset()`: drain the asyncio queue **and** call `self._streamer.reset()`.
- `MockSSVEPConsumer.reset()`: no-op (the mock queue is the test script).
- Optionally filter by threshold in `RealSSVEPConsumer._callback` before putting on the queue.

**Session** (`Proj2/backend/speller_backend/session.py`):

- New flow: after `_consume_ssvep` commits a word, branch on `sentence` / flag:
  - continuation path: call a new `_transition_to_continuation()` that calls `predict_words(prefix="", sentence=…)` directly and starts SSVEP (no `start_flashing`).
  - prefix path: existing `_transition_to_prefix()` (unchanged).
- On `next_prediction()` returning `None` or unmapped freq → fall back to `_transition_to_prefix()`.
- Call `ssvep.reset()` at each transition between cycles.

### Calibration step (must happen on demo subject, per-subject thresholds)

During the 2-min baseline session:

- 30 s per target, subject instructed to attend that frequency (three targets × 30 s = 90 s).
- 30 s with subject looking at a non-flickering central fixation cross (no attention block).
- Compute `top_corr` distribution: attended vs unattended, per target.
- Set threshold = 10th percentile of attended values (or midway between attended-mean and unattended-mean, whichever is lower).

From Ali's 2026-04-20 tryout (`Ali_SSVEP_Tryout_results.txt`): SNR at Oz was 12.27 (10 Hz), 5.76 (12 Hz), 5.61 (15 Hz). Translating to CCA `top_corr` expectations (different metric but correlated): 10 Hz should comfortably exceed 0.4, 12/15 Hz might hover closer to 0.3. **Thresholds may need to be per-target, not a single global value.**

### Open questions to resolve on hardware tomorrow

1. Does the upstream `current_display_pred` stuck-state actually manifest? (It does in code — but real-world callback cadence may mask it.)
2. Where does the threshold sit for this subject on 12 Hz and 15 Hz specifically? Might be uncomfortably close to the unattended floor.
3. Is 5 s the right SSVEP window? Shorter = faster fallback, more misses. Longer = slower, fewer false negatives.
4. Does FBCCA's harmonic stacking push `top_corr` high enough that threshold gating is trivial? If yes, this whole discussion simplifies.
5. Harmonic note for later: 10 Hz × 3 = 30 Hz = 15 Hz × 2. Shared harmonics already degrade 15 Hz SNR in FBCCA; check if it's a problem for threshold tuning.

### What we test first: LLM continuation quality

Before touching hardware we can isolate and test **just the continuation prompt**. The session already supports empty-prefix + non-empty-sentence calls to `predict_words` — Mohamed's prompt handles that branch explicitly.

Test protocol (mocked SSVEP, live Groq):

- Seed sentences of length 1, 2, 3 words ending with likely demo opener words ("hello", "welcome", "my", "how", "thank").
- Call `predict_words(prefix="", sentence=<seed>, context=DEMO_SPELLER_CONTEXT)`.
- For each seed, record the 3 continuations and ask: does top-1 feel like a natural demo-stage continuation?
- Iterate on `DEMO_SPELLER_CONTEXT` and/or the speller_api system prompt until top-1 is usable ≥80% of the time across the seeds we'll actually use on stage.

If LLM continuation quality is good: Option A (continuation chain) is the hero path, P300 fallback is the safety net.

If LLM continuation quality is middling: fall back to Option B more often; rehearse with shorter continuations only.

### Not doing (explicitly rejected)

- **4th SSVEP frequency for "retry".** Nearest safe candidates (8.57 Hz, 17 Hz) need per-subject SNR validation; 4-class CCA/FBCCA is harder than 3-class; the UI grows a 4th flickering box that competes for attention. Rejected in favour of the threshold + reset + timeout combo above.
- **EEG-driven attention gating.** Computing a separate attention score from occipital alpha is interesting for Scenario B but is scope for after the demo.
- **Parallel P300 + SSVEP paradigms.** Running both simultaneously muddies both classifiers visually. Stick to time-sliced modes.
