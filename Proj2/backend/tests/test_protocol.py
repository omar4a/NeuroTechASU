"""Protocol tests: command builders validate + event parser is tolerant."""
from __future__ import annotations

import pytest

from speller_backend import protocol


class TestCommandBuilders:
    def test_start_stop_flashing(self):
        assert protocol.start_flashing() == {"command": "start_flashing"}
        assert protocol.stop_flashing() == {"command": "stop_flashing"}

    def test_start_stop_ssvep(self):
        assert protocol.start_ssvep() == {"command": "start_ssvep"}
        assert protocol.stop_ssvep() == {"command": "stop_ssvep"}

    def test_type_char_ok(self):
        assert protocol.type_char("H") == {"command": "type_char", "char": "H"}

    @pytest.mark.parametrize("bad", ["", "HI", 42, None])
    def test_type_char_rejects_non_single_char(self, bad):
        with pytest.raises(ValueError):
            protocol.type_char(bad)  # type: ignore[arg-type]

    def test_update_predictions_ok(self):
        msg = protocol.update_predictions(["hello", "hope", "help"])
        assert msg == {"command": "update_predictions", "words": ["hello", "hope", "help"]}

    @pytest.mark.parametrize("bad", [[], ["a"], ["a", "b"], ["a", "b", "c", "d"], None])
    def test_update_predictions_rejects_wrong_shape(self, bad):
        with pytest.raises(ValueError):
            protocol.update_predictions(bad)  # type: ignore[arg-type]

    def test_update_predictions_rejects_non_strings(self):
        with pytest.raises(ValueError):
            protocol.update_predictions(["a", 1, "c"])  # type: ignore[list-item]

    def test_backspace_ok(self):
        assert protocol.backspace(2) == {"command": "backspace", "count": 2}
        assert protocol.backspace(0) == {"command": "backspace", "count": 0}

    @pytest.mark.parametrize("bad", [-1, 1.5, "2", None])
    def test_backspace_rejects_bad_counts(self, bad):
        with pytest.raises(ValueError):
            protocol.backspace(bad)  # type: ignore[arg-type]

    def test_chatgpt_reply_ok(self):
        msg = protocol.chatgpt_reply("Hello back!")
        assert msg == {"command": "chatgpt_reply", "text": "Hello back!"}

    def test_chatgpt_reply_accepts_empty_string(self):
        assert protocol.chatgpt_reply("") == {"command": "chatgpt_reply", "text": ""}

    @pytest.mark.parametrize("bad", [None, 42, ["a"], {"text": "x"}])
    def test_chatgpt_reply_rejects_non_string(self, bad):
        with pytest.raises(ValueError):
            protocol.chatgpt_reply(bad)  # type: ignore[arg-type]


class TestParseEvent:
    def test_init(self):
        ev = protocol.parse_event({"event": "init", "context": "chat", "timestamp": 1234.5})
        assert isinstance(ev, protocol.InitEvent)
        assert ev.context == "chat"
        assert ev.timestamp == 1234.5

    def test_flash(self):
        ev = protocol.parse_event({"event": "flash", "target": "row_2", "timestamp": 99.0})
        assert isinstance(ev, protocol.FlashEvent)
        assert ev.target == "row_2"

    def test_unknown_event_returns_none(self):
        assert protocol.parse_event({"event": "whatever"}) is None

    def test_non_dict_returns_none(self):
        assert protocol.parse_event("not a dict") is None  # type: ignore[arg-type]

    def test_missing_timestamp_defaults_to_zero(self):
        ev = protocol.parse_event({"event": "init", "context": "chat"})
        assert ev.timestamp == 0.0

    def test_tolerates_null_timestamp(self):
        ev = protocol.parse_event({"event": "init", "context": "chat", "timestamp": None})
        assert ev.timestamp == 0.0
