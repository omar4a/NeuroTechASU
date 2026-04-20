# Proj2 speller backend

WebSocket + HTTP server that bridges the Proj2 BCI speller UI (`Proj2/ui/app.js`) to the LLM word-prediction backend (`task5_speller_api`) and Omar's real-time SSVEP classifier (`SSVEP Protocol/ssvep_realtime.py`).

## What it does

```
Browser UI (Proj2/ui)         speller-backend (this package)           Groq / LLM
     |                               |                                      |
     |--- ws connect, init --------->|                                      |
     |<-- start_flashing -------------|                                     |
     |                               |   (P300 classifier POSTs decoded    |
     |                               |    letters to http://.../p300/char) |
     |<-- type_char (H) -------------|                                      |
     |<-- type_char (E) -------------|                                      |
     |                               |--- predict_words(prefix=HE) ------->|
     |                               |<------- ["hello","hope","help"] ----|
     |<-- stop_flashing -------------|                                      |
     |<-- update_predictions --------|                                      |
     |<-- start_ssvep ---------------|                                      |
     |                               |   (SSVEP classifier consumes LSL    |
     |                               |    and emits 10.0 | 12.0 | 15.0 Hz) |
     |<-- stop_ssvep ----------------|                                      |
     |<-- backspace, type_char(h) ...|                                      |
     |--- (loop for next word) ----->|                                      |
```

## Install (editable, for development)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e /path/to/Project\ 1/speller_api   # the task5_speller_api package
pip install -e .[dev]                             # this backend + tests
```

For real hardware:

```bash
pip install -e .[real-ssvep]
```

## Run

```bash
export OPENAI_API_KEY=...                    # Groq by default; see speller_api/.env.example
export OPENAI_API_BASE_URL=https://api.groq.com/openai/v1
export OPENAI_MODEL=llama-3.3-70b-versatile
speller-backend
# or: python -m speller_backend
```

### Config (env vars)

| Var | Default | Meaning |
|---|---|---|
| `SPELLER_WS_HOST`      | `0.0.0.0` | WebSocket bind host |
| `SPELLER_WS_PORT`      | `8765`    | WebSocket port (frontend default) |
| `SPELLER_HTTP_PORT`    | `8766`    | HTTP port for P300 char injection |
| `SPELLER_PREFIX_LENGTH`| `2`       | Auto-commit prefix after N P300 chars |
| `SPELLER_SSVEP_MODE`   | `mock`    | `mock` (preprogrammed 10/12/15) or `real` (Unicorn LSL) |
| `SPELLER_CONTINUATION_MODE` | `true` | `true`: after first word, skip P300 and offer 3 LLM continuations (Option A). `false`: enter fresh prefix via P300 for every word (Option B). See `trial&error.md`. |
| `SPELLER_LOG_LEVEL`    | `INFO`    | |

## Inject a character manually (no P300 needed)

```bash
curl -X POST http://localhost:8766/p300/char \
     -H 'Content-Type: application/json' \
     -d '{"char":"H"}'
```

When the P300 classifier lands, it hits the same endpoint.

## Tests

```bash
pytest
```

All tests mock the WebSocket, speller_api, and SSVEP consumer — no network, no hardware.

## Protocol

See `speller_backend/protocol.py`. Two frontend events (`init`, `flash`) and seven backend commands (`start_flashing`, `stop_flashing`, `start_ssvep`, `stop_ssvep`, `type_char`, `update_predictions`, `backspace`). `backspace` is a new command added by this backend; frontend support lands when Zeina wires it into `app.js`.
