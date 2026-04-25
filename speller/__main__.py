"""CLI entry point — lets teammates smoke-test without writing Python.

Example:

    python -m task5_speller_api he --context "writing an email to my professor"
    → ["hello", "hope", "help"]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import sys
import time

from .speller import API, predict_words, respond_to_sentence


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_word(sentence: str, word: str) -> str:
    sentence = sentence.strip()
    if not sentence:
        return word
    return f"{sentence} {word}"


def _display_sentence(sentence: str, prefix: str) -> str:
    sentence = sentence.strip()
    if sentence and prefix:
        return f"{sentence} {prefix}"
    return sentence or prefix


def _render_simulator(
    topic: str,
    sentence: str,
    prefix: str,
    predictions: list[str],
    latency_ms: float,
) -> None:
    if os.name == "nt":
        os.system("cls")

    print("BCI Speller Real-Time Simulator")
    print("=" * 38)
    print(f"Topic: {topic}")
    print(f"Sentence: {_display_sentence(sentence, prefix) or '(empty)'}")
    print(f"Current prefix: {prefix or '(empty)'}")
    print(f"Prediction latency: {latency_ms:.1f} ms")
    print()
    print("Recommendations")
    for idx, word in enumerate(predictions, start=1):
        print(f"  {idx}. {word}")
    print()
    print("Controls")
    print("  - Type letters to update predictions in real time")
    print("  - Press 1 / 2 / 3 to select a recommendation")
    print("  - Backspace to remove one typed letter")
    print("  - Space to commit typed prefix as a manual word")
    print("  - Enter to finish and save JSON report")


def _default_report_path() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("logs", f"speller_simulation_{stamp}.json")


def _run_simulator(output_path: str | None) -> int:
    if os.name != "nt":
        print("error: simulator currently supports Windows consoles only", file=sys.stderr)
        return 2

    import msvcrt

    topic = input("Enter topic/context for this test session: ").strip()
    if not topic:
        print("error: topic cannot be empty", file=sys.stderr)
        return 2

    api = API()
    sentence = ""
    prefix = ""
    events: list[dict[str, object]] = []

    session_started_at = _utc_now_iso()
    report_path = output_path or _default_report_path()

    def refresh() -> tuple[list[str], float]:
        t0 = time.perf_counter()
        predictions = api.predict_words(context=topic, prefix=prefix, sentence=sentence)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return predictions, elapsed_ms

    try:
        predictions, latency_ms = refresh()
        _render_simulator(topic, sentence, prefix, predictions, latency_ms)

        while True:
            ch = msvcrt.getwch()

            if ch == "\x03":
                raise KeyboardInterrupt

            if ch in ("\r", "\n"):
                break

            if ch in ("\x00", "\xe0"):
                _ = msvcrt.getwch()
                continue

            if ch == "\x08":
                prefix = prefix[:-1]
                predictions, latency_ms = refresh()
                _render_simulator(topic, sentence, prefix, predictions, latency_ms)
                continue

            if ch == " ":
                if prefix:
                    selected_word = prefix.lower()
                    sentence_before = sentence
                    sentence = _append_word(sentence, selected_word)
                    events.append(
                        {
                            "timestamp": _utc_now_iso(),
                            "selection_mode": "manual",
                            "sentence_before": sentence_before,
                            "prefix_at_selection": prefix,
                            "letters_typed": len(prefix),
                            "shown_predictions": list(predictions),
                            "selected_index": None,
                            "selected_word": selected_word,
                            "prediction_latency_ms": round(latency_ms, 3),
                        }
                    )
                    prefix = ""
                predictions, latency_ms = refresh()
                _render_simulator(topic, sentence, prefix, predictions, latency_ms)
                continue

            if ch in ("1", "2", "3", "4"):
                idx = int(ch) - 1
                if idx < len(predictions):
                    selected_word = predictions[idx]
                    sentence_before = sentence
                    sentence = _append_word(sentence, selected_word)
                    events.append(
                        {
                            "timestamp": _utc_now_iso(),
                            "selection_mode": "recommendation",
                            "sentence_before": sentence_before,
                            "prefix_at_selection": prefix,
                            "letters_typed": len(prefix),
                            "shown_predictions": list(predictions),
                            "selected_index": idx + 1,
                            "selected_word": selected_word,
                            "prediction_latency_ms": round(latency_ms, 3),
                        }
                    )
                    prefix = ""
                    predictions, latency_ms = refresh()
                    _render_simulator(topic, sentence, prefix, predictions, latency_ms)
                continue

            if ch.isalpha() or ch in ("'", "-"):
                prefix += ch.lower()
                predictions, latency_ms = refresh()
                _render_simulator(topic, sentence, prefix, predictions, latency_ms)
                continue

    except KeyboardInterrupt:
        print("\nSession interrupted. Saving results so far...")

    session_ended_at = _utc_now_iso()
    
    response = ""
    if sentence:
        try:
            response = api.respond(sentence, context=topic)
        except Exception as e:
            print(f"warning: failed to get response: {e}", file=sys.stderr)
    
    report = {
        "topic": topic,
        "started_at": session_started_at,
        "ended_at": session_ended_at,
        "event_count": len(events),
        "events": events,
        "final_sentence": sentence,
        "response": response,
    }

    parent = os.path.dirname(report_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if response:
        print("\nModel Response:")
        print("-" * 40)
        print(response)
        print("-" * 40)
    print(f"Saved simulation report to: {report_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m task5_speller_api",
        description="Predict words from a BCI prefix, or run an interactive simulator.",
    )
    parser.add_argument(
        "prefix",
        nargs="?",
        default="",
        help="letters committed so far (usually 1-3)",
    )
    parser.add_argument(
        "--context",
        default="",
        help="Optional scenario context, e.g. 'ordering food at a restaurant'",
    )
    parser.add_argument(
        "--sentence",
        default="",
        help="Optional sentence being composed up to the current word",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run real-time simulator and save a JSON quality report",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON path for --simulate mode",
    )
    parser.add_argument(
        "--respond",
        default="",
        help="Generate a contextual response to the given sentence",
    )
    args = parser.parse_args(argv)

    if args.simulate:
        return _run_simulator(args.output or None)

    if args.respond:
        try:
            response = respond_to_sentence(args.respond, context=args.context)
            print(response)
            return 0
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 3

    if not args.prefix:
        parser.error("prefix is required unless --simulate is used")

    try:
        words = predict_words(
            prefix=args.prefix, context=args.context, sentence=args.sentence
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    print(json.dumps(words))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
