from __future__ import annotations

import argparse
import math
import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from typing import Iterable

from pylsl import StreamInlet, StreamInfo, proc_clocksync, proc_dejitter, resolve_byprop

from quality_engine import QualityEngine


DEFAULT_CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
RAW_STREAM_NAME = "UnicornRecorderRawDataLSLStream"
ELECTRODE_LAYOUT = {
    "Fz": (0.50, 0.18),
    "C3": (0.28, 0.38),
    "Cz": (0.50, 0.36),
    "C4": (0.72, 0.38),
    "Pz": (0.50, 0.58),
    "PO7": (0.32, 0.76),
    "Oz": (0.50, 0.82),
    "PO8": (0.68, 0.76),
}
PALETTE = {
    "background": "#07131b",
    "head": "#142733",
    "head_outline": "#294453",
    "text": "#e6f3f7",
    "muted": "#8aa5b1",
    "good_outer": "#0e5032",
    "good_inner": "#31d986",
    "bad_outer": "#5a1919",
    "bad_inner": "#ff5c5c",
    "idle_outer": "#2b3a45",
    "idle_inner": "#8596a1",
}


@dataclass
class ChannelState:
    label: str
    good: bool | None
    display_uv: float | None


class RollingQualityBuffer:
    def __init__(self, channel_count: int, window_samples: int, sampling_rate: float) -> None:
        self.engine = QualityEngine(sampling_rate=sampling_rate)
        self._buffers = [deque(maxlen=window_samples) for _ in range(channel_count)]
        self._lock = threading.Lock()
        self._last_sample_time = 0.0

    def extend(self, sample: Iterable[float]) -> None:
        with self._lock:
            for buffer, value in zip(self._buffers, sample):
                buffer.append(float(value))
            self._last_sample_time = time.monotonic()

    def snapshot(self, labels: list[str]) -> list[ChannelState]:
        with self._lock:
            stale = (time.monotonic() - self._last_sample_time) > 1.5
            snapshots = [(label, list(buffer)) for label, buffer in zip(labels, self._buffers)]

        states: list[ChannelState] = []
        for label, raw_window in snapshots:
            if stale or len(raw_window) < 10:
                states.append(ChannelState(label=label, good=None, display_uv=None))
                continue

            metrics = self.engine.process_window(label, raw_window)
            states.append(
                ChannelState(
                    label=label,
                    good=bool(metrics["is_good"]),
                    display_uv=float(metrics["display_uv"]),
                )
            )

        return states


class LSLReader(threading.Thread):
    def __init__(
        self,
        inlet: StreamInlet,
        buffer: RollingQualityBuffer,
        channel_indices: list[int],
    ) -> None:
        super().__init__(daemon=True)
        self._inlet = inlet
        self._buffer = buffer
        self._channel_indices = channel_indices
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            samples, _ = self._inlet.pull_chunk(timeout=0.2, max_samples=32)
            for sample in samples:
                self._buffer.extend(float(sample[index]) for index in self._channel_indices)


class BrainQualityMonitor:
    def __init__(self, root: tk.Tk, labels: list[str], buffer: RollingQualityBuffer, stream_name: str) -> None:
        self._root = root
        self._labels = labels
        self._buffer = buffer
        self._dot_items: dict[str, tuple[int, int, int]] = {}
        self._value_labels: dict[str, int] = {}
        self._status_var = tk.StringVar(value=f"Connected to LSL stream: {stream_name}")
        self._quality_var = tk.StringVar(value="Triple-check quality: Vpp 0.5-500 uV, std < 50 uV")

        self._root.title("Unicorn Signal Quality Monitor")
        self._root.configure(bg=PALETTE["background"])
        self._root.minsize(760, 760)

        self._canvas = tk.Canvas(
            self._root,
            width=760,
            height=700,
            bg=PALETTE["background"],
            highlightthickness=0,
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)

        footer = tk.Frame(self._root, bg=PALETTE["background"])
        footer.pack(fill=tk.X, padx=18, pady=(0, 18))

        tk.Label(
            footer,
            textvariable=self._status_var,
            bg=PALETTE["background"],
            fg=PALETTE["text"],
            anchor="w",
            font=("Segoe UI", 10, "bold"),
        ).pack(fill=tk.X)
        tk.Label(
            footer,
            textvariable=self._quality_var,
            bg=PALETTE["background"],
            fg=PALETTE["muted"],
            anchor="w",
            font=("Segoe UI", 10),
        ).pack(fill=tk.X, pady=(4, 0))

        self._draw_figure()
        self._schedule_refresh()

    def _draw_figure(self) -> None:
        width = 760
        height = 700
        cx = width / 2
        cy = height / 2 + 10
        radius = 255

        self._canvas.create_oval(
            cx - radius,
            cy - radius,
            cx + radius,
            cy + radius,
            fill=PALETTE["head"],
            outline=PALETTE["head_outline"],
            width=3,
        )
        self._canvas.create_polygon(
            cx - 24,
            cy - radius - 3,
            cx,
            cy - radius - 36,
            cx + 24,
            cy - radius - 3,
            fill=PALETTE["head"],
            outline=PALETTE["head_outline"],
            width=3,
        )
        self._canvas.create_arc(
            cx - radius - 34,
            cy - 92,
            cx - radius + 22,
            cy + 92,
            start=270,
            extent=180,
            style=tk.ARC,
            outline=PALETTE["head_outline"],
            width=3,
        )
        self._canvas.create_arc(
            cx + radius - 22,
            cy - 92,
            cx + radius + 34,
            cy + 92,
            start=90,
            extent=180,
            style=tk.ARC,
            outline=PALETTE["head_outline"],
            width=3,
        )

        for label in self._labels:
            nx, ny = ELECTRODE_LAYOUT[label]
            x = cx - radius + (2 * radius * nx)
            y = cy - radius + (2 * radius * ny)
            outer = self._canvas.create_oval(x - 28, y - 28, x + 28, y + 28, fill=PALETTE["idle_outer"], outline="")
            inner = self._canvas.create_oval(x - 16, y - 16, x + 16, y + 16, fill=PALETTE["idle_inner"], outline="")
            text = self._canvas.create_text(
                x,
                y - 42,
                text=label,
                fill=PALETTE["text"],
                font=("Segoe UI", 11, "bold"),
            )
            value = self._canvas.create_text(
                x,
                y + 42,
                text="waiting",
                fill=PALETTE["muted"],
                font=("Consolas", 9),
            )
            self._dot_items[label] = (outer, inner, text)
            self._value_labels[label] = value

    def _schedule_refresh(self) -> None:
        self._refresh()
        self._root.after(100, self._schedule_refresh)

    def _refresh(self) -> None:
        for state in self._buffer.snapshot(self._labels):
            outer, inner, _ = self._dot_items[state.label]
            if state.good is None:
                outer_color = PALETTE["idle_outer"]
                inner_color = PALETTE["idle_inner"]
                value_text = "waiting"
            elif state.good:
                outer_color = PALETTE["good_outer"]
                inner_color = PALETTE["good_inner"]
                value_text = f"{state.display_uv:5.1f} uV"
            else:
                outer_color = PALETTE["bad_outer"]
                inner_color = PALETTE["bad_inner"]
                value_text = f"{state.display_uv:5.1f} uV"

            self._canvas.itemconfigure(outer, fill=outer_color)
            self._canvas.itemconfigure(inner, fill=inner_color)
            self._canvas.itemconfigure(self._value_labels[state.label], text=value_text)


def channel_labels_from_stream(info: StreamInfo) -> list[str]:
    labels: list[str] = []
    channels = info.desc().child("channels")
    if channels.empty():
        return labels

    channel = channels.child("channel")
    while not channel.empty():
        label = channel.child_value("label")
        if label:
            labels.append(label.strip())
        channel = channel.next_sibling("channel")
    return labels


def select_stream(timeout_seconds: float) -> StreamInfo:
    candidates = resolve_byprop("name", RAW_STREAM_NAME, timeout=timeout_seconds)
    for info in candidates:
        if info.channel_count() < len(DEFAULT_CHANNELS):
            continue
        if info.type() != "EEG":
            continue
        return info

    raise RuntimeError(
        f"LSL stream '{RAW_STREAM_NAME}' was not found with a compatible EEG configuration. "
        "Make sure Unicorn Recorder is actively streaming the raw outlet."
    )


def prime_stream(
    inlet: StreamInlet,
    channel_indices: list[int],
    timeout_seconds: float,
) -> list[float]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        chunk, _ = inlet.pull_chunk(timeout=0.5, max_samples=16)
        if not chunk:
            continue
        sample = chunk[0]
        return [float(sample[index]) for index in channel_indices]

    raise RuntimeError(
        "Connected to UnicornRecorderRawDataLSLStream but did not receive any samples. "
        "Check that the raw LSL stream is enabled and actively transmitting in Unicorn Recorder."
    )


def select_channel_indices(stream_labels: list[str]) -> tuple[list[int], list[str]]:
    normalized = {label.upper(): index for index, label in enumerate(stream_labels)}
    if all(label.upper() in normalized for label in DEFAULT_CHANNELS):
        indices = [normalized[label.upper()] for label in DEFAULT_CHANNELS]
        return indices, DEFAULT_CHANNELS

    return list(range(len(DEFAULT_CHANNELS))), DEFAULT_CHANNELS


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live Unicorn signal quality monitor over LSL.")
    parser.add_argument("--search-timeout", type=float, default=12.0, help="Seconds to wait for an LSL stream.")
    parser.add_argument(
        "--first-sample-timeout",
        type=float,
        default=3.0,
        help="Seconds to wait for the first sample after connecting to the LSL stream.",
    )
    parser.add_argument("--window-seconds", type=float, default=1.0, help="Rolling window size in seconds.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    stream_info = select_stream(args.search_timeout)
    stream_labels = channel_labels_from_stream(stream_info)
    channel_indices, display_labels = select_channel_indices(stream_labels)

    sample_rate = stream_info.nominal_srate()
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        sample_rate = 250.0

    window_samples = max(1, round(sample_rate * args.window_seconds))
    inlet = StreamInlet(stream_info, processing_flags=proc_clocksync | proc_dejitter)
    buffer = RollingQualityBuffer(len(display_labels), window_samples, sample_rate)
    buffer.extend(prime_stream(inlet, channel_indices, args.first_sample_timeout))
    reader = LSLReader(
        inlet=inlet,
        buffer=buffer,
        channel_indices=channel_indices,
    )
    reader.start()

    root = tk.Tk()
    stream_name = f"{stream_info.name()} | {stream_info.channel_count()} ch @ {sample_rate:.0f} Hz"
    app = BrainQualityMonitor(root=root, labels=display_labels, buffer=buffer, stream_name=stream_name)
    root.protocol("WM_DELETE_WINDOW", root.quit)

    try:
        root.mainloop()
    finally:
        reader.stop()
        inlet.close_stream()


if __name__ == "__main__":
    main()
