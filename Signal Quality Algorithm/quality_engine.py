from __future__ import annotations

import numpy as np


class QualityEngine:
    def __init__(self, sampling_rate: float = 250.0) -> None:
        self.fs = sampling_rate
        self.scale_factor = 4_500_000.0 / 8_388_608.0
        self.railing_threshold_uv = 180_000.0

    def process_window(self, raw_counts: list[float] | np.ndarray) -> dict[str, float | bool]:
        """
        Input: 1D array of raw ADC counts for a single channel.
        Output: quality metrics and a boolean 'is_good'.
        """
        data_uv = np.asarray(raw_counts, dtype=float) * self.scale_factor
        offset_uv = float(np.mean(data_uv))
        detrended = data_uv - offset_uv

        v_pp = float(np.ptp(detrended))
        std_dev = float(np.std(detrended))

        is_connected = 0.5 < v_pp < 500.0
        is_clean = std_dev < 50.0
        is_railed = abs(offset_uv) > self.railing_threshold_uv

        return {
            "is_good": bool(is_connected and is_clean and not is_railed),
            "v_pp": v_pp,
            "std_dev": std_dev,
            "offset_uv": offset_uv,
            "is_connected": bool(is_connected),
            "is_clean": bool(is_clean),
            "is_railed": bool(is_railed),
        }
