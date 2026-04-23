"""
Real-Time P300 Inference Engine.

Trains both xDAWN+LDA and Riemannian MDM pipelines on stored training data,
then runs a live decoding loop that:
  1. Buffers EEG data from a Unicorn headset via LSL
  2. Receives flash markers from the PsychoPy UI
  3. Epochs & preprocesses flash-locked data
  4. Accumulates Bayesian evidence across flashes to identify the attended character
  5. Pushes decoded characters back to the UI via LSL + logs to file

Key improvements over previous version:
  - Uses collections.deque for O(1) buffer management instead of list slicing
  - Pre-computed filter coefficients via shared signal_processing module
  - Proper MDM probability estimation via softmax over negative distances
  - Ensemble mode that averages both pipeline probabilities
  - Artifact rejection on individual epochs
"""

import os
import sys

# PsychoPy Runner isolates the environment, stripping standard user site-packages.
# We append the standard Python 3.10 user site (where pyriemann/mne/sklearn live)
# to sys.path. We APPEND so that PsychoPy's own libraries take priority.
_user_site = os.path.expanduser(r"~\AppData\Roaming\Python\Python310\site-packages")
if os.path.isdir(_user_site) and _user_site not in sys.path:
    sys.path.append(_user_site)

import numpy as np
import pylsl
import asyncio
from collections import deque

from pyriemann.spatialfilters import Xdawn
from pyriemann.estimation import XdawnCovariances
from pyriemann.classification import MDM
from mne.decoding import Vectorizer
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from signal_processing import (
    FS, EPOCH_LEN, SAMPLES_PER_EPOCH, BASELINE_SAMPLES,
    MATRIX_CHARS, ARTIFACT_THRESHOLD_UV,
    apply_preprocessing, reject_artifacts, extract_epoch
)

# Maximum EEG buffer size: 60 seconds of data
MAX_BUFFER_SAMPLES = FS * 60


class RealTimeInference:
    def __init__(self):
        # EEG ring buffer — deque with bounded maxlen for O(1) append + auto-eviction
        self.eeg_data = deque(maxlen=MAX_BUFFER_SAMPLES)
        self.eeg_times = deque(maxlen=MAX_BUFFER_SAMPLES)
        
        self.current_trial_flashes = []
        self.is_running = True
        
        # Incremental decoding state — avoids re-processing all flashes on every EVALUATE
        self._processed_flash_idx = 0
        self._accumulated_log_probs = {c: -np.log(len(MATRIX_CHARS)) for c in MATRIX_CHARS}
        
        # Algorithm selection: "xdawn_lda", "riemann_mdm", or "ensemble"
        self.active_algo = "ensemble"
        self.ensemble_weight_lda = 0.5  # Weight for LDA in ensemble (MDM gets 1 - this)
        self.model_lda = None
        self.model_mdm = None
        # Separate xDAWN covariance transformer + MDM classifier for manual predict_proba
        self._mdm_cov_transformer = None
        self._mdm_classifier = None
        
        # Create LSL outlet for decoded characters
        info_dec = pylsl.StreamInfo('Speller_Decoded', 'Markers', 1, 0, 'string', 'bci_decoder_123')
        self.decoded_outlet = pylsl.StreamOutlet(info_dec)
        print("LSL Decoded Outlet created.")
        
        # Hard logging fallback — writes decoded characters to file even if LSL drops
        self.log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spelled_text.txt")
        with open(self.log_file, "w") as f:
            f.write("")  # Clear or create file on exact launch
        
    def load_and_train_model(self):
        """Load training data and fit both classification pipelines."""
        print("Loading training data to train the live classifier...")
        output_dir = os.path.dirname(os.path.abspath(__file__))
        x_path = os.path.join(output_dir, "X_train.npy")
        y_path = os.path.join(output_dir, "y_train.npy")
        
        if not os.path.exists(x_path):
            raise FileNotFoundError("X_train.npy required to train the real-time model. Do data collection first.")
            
        X = np.load(x_path)
        y = np.load(y_path)
        
        if len(X) == 0 or len(X.shape) < 3:
            print("\n" + "!" * 60)
            print("FATAL ERROR: Your X_train.npy dataset is completely empty!")
            print("Because your PsychoPy UI crashed on your earlier attempts, no data was successfully collected.")
            print("-> You MUST boot up PsychoPy WITHOUT 'Inference Mode' first to train the algorithms, then you can use Freestyle!")
            print("!" * 60 + "\n")
            os.system('pause' if os.name == 'nt' else "read -p 'Press Enter to continue...'")
            import sys; sys.exit(1)
        
        # X_train is now fully preprocessed by data_collection.py
        
        print("Training xDAWN + LDA Pipeline...")
        self.model_lda = make_pipeline(
            Xdawn(nfilter=2),
            Vectorizer(),
            LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.9)
        )
        self.model_lda.fit(X, y)
        
        print("Training Riemannian MDM Pipeline...")
        # Build MDM pipeline with separate references for manual probability extraction
        self._mdm_cov_transformer = XdawnCovariances(nfilter=2, estimator="oas")
        self._mdm_classifier = MDM()
        
        X_cov = self._mdm_cov_transformer.fit_transform(X, y)
        self._mdm_classifier.fit(X_cov, y)
        
        # Also build a combined pipeline for convenience
        self.model_mdm = make_pipeline(
            XdawnCovariances(nfilter=2, estimator="oas"),
            MDM()
        )
        self.model_mdm.fit(X, y)
        print("Models Trained Successfully.")

    def _mdm_predict_proba(self, X_test):
        """
        Compute proper probability estimates from MDM using softmax over negative distances.
        
        MDM classifies by measuring Riemannian geodesic distance to each class centroid.
        Since it doesn't natively produce probabilities, we convert distances to probabilities
        using softmax: P(class_k) = exp(-d_k) / sum(exp(-d_j) for all j)
        
        Parameters
        ----------
        X_test : np.ndarray, shape (epochs, channels, samples)
            Test epochs in raw EEG format.
        
        Returns
        -------
        np.ndarray, shape (epochs, 2) : Probability estimates [P(non-target), P(target)]
        """
        # Transform to covariance space
        X_cov = self._mdm_cov_transformer.transform(X_test)
        
        # Get distances to each class centroid — shape (n_samples, n_classes)
        # MDM stores centroids in self.covmeans_ after fitting
        from pyriemann.utils.distance import distance
        
        n_samples = X_cov.shape[0]
        n_classes = len(self._mdm_classifier.classes_)
        distances = np.zeros((n_samples, n_classes))
        
        for j, centroid in enumerate(self._mdm_classifier.covmeans_):
            for i in range(n_samples):
                distances[i, j] = distance(X_cov[i], centroid, metric=self._mdm_classifier.metric)
        
        # Convert distances to probabilities via softmax over negative distances
        neg_distances = -distances
        # Numerical stability: subtract max per row
        neg_distances -= neg_distances.max(axis=1, keepdims=True)
        exp_neg_d = np.exp(neg_distances)
        probs = exp_neg_d / exp_neg_d.sum(axis=1, keepdims=True)
        
        return probs

    async def lsl_worker(self, inlet):
        """Continuously buffer EEG data from the Unicorn headset."""
        print("Connected to Unicorn stream. Listening for data...")
        while self.is_running:
            chunk, timestamps = inlet.pull_chunk()
            if timestamps:
                # deque handles eviction automatically — no manual slicing needed
                self.eeg_data.extend(chunk)
                self.eeg_times.extend(timestamps)
                    
            await asyncio.sleep(0.01)

    async def marker_worker(self, marker_inlet):
        """Process flash markers from the PsychoPy UI."""
        print("Connected to Marker stream. Receiving markers...")
        while self.is_running:
            marker, timestamp = marker_inlet.pull_sample(timeout=0.0)
            if timestamp is not None and marker is not None:
                m_str = marker[0]
                if m_str == "SESSION_START":
                    self.current_trial_flashes = []
                    self._reset_trial_state()
                    print("Trial sequence initiating...")
                elif m_str.startswith("FLASH_GROUP_"):
                    group = list(m_str.replace("FLASH_GROUP_", ""))
                    self.current_trial_flashes.append((timestamp, group))
                elif m_str == "EVALUATE":
                    with open(self.log_file, "a") as f: f.write(f"\n[DEBUG] EVALUATE called. Buffer holds {len(self.eeg_data)} EEG samples.\n")
                    predicted_char, hit_threshold = self.decode_trial(check_threshold=True)
                    if hit_threshold and predicted_char:
                        print(f"*** DYNAMIC STOP *** PREDICTED: {predicted_char} (Confidence > 0.95)")
                        self.decoded_outlet.push_sample([f"DECODED_{predicted_char}"], pylsl.local_clock())
                        
                        # HARD LOG TO FILE
                        with open(self.log_file, "a") as f:
                            f.write(predicted_char)
                            
                        self.current_trial_flashes = []
                        self._reset_trial_state()
                elif m_str == "TRIAL_END":
                    print("Trial ended. Decoding final fallback signal...")
                    with open(self.log_file, "a") as f: f.write(f"\n[DEBUG] TRIAL_END received! Flashes logged: {len(self.current_trial_flashes)}\n")
                    await asyncio.sleep(0.5)  # Wait half a second for final epochs 
                    
                    predicted_char, _ = self.decode_trial(check_threshold=False)
                    with open(self.log_file, "a") as f: f.write(f"\n[DEBUG] FALLBACK PREDICTION PROCESSED! Char: {predicted_char}\n")
                    
                    if predicted_char:
                        print(f"---------> FINAL PREDICTED: {predicted_char} <---------")
                        self.decoded_outlet.push_sample([f"DECODED_{predicted_char}"], pylsl.local_clock())
                        
                        # HARD LOG TO FILE
                        with open(self.log_file, "a") as f:
                            f.write(predicted_char)
                    else:
                        print("Failed to decode trial.")
                    self.current_trial_flashes = []
                    self._reset_trial_state()
            await asyncio.sleep(0.01)

    async def main_loop(self):
        """Resolve LSL streams and run the main async event loop."""
        print("Resolving LSL Streams...")
        loop = asyncio.get_running_loop()
        unicorn_streams = await loop.run_in_executor(None, pylsl.resolve_byprop, 'name', 'UnicornRecorderLSLStream', 1, 10.0)
        marker_streams = await loop.run_in_executor(None, pylsl.resolve_byprop, 'name', 'Speller_Markers', 1, 10.0)
        
        if not unicorn_streams:
            print("ERROR: Could not find Unicorn stream within 10 seconds.")
            return
            
        if not marker_streams:
            print("ERROR: Could not find Speller_Markers stream in 10 seconds.")
            print("Make sure psychopy_speller.py is running.")
            return
            
        inlet = pylsl.StreamInlet(unicorn_streams[0])
        marker_inlet = pylsl.StreamInlet(marker_streams[0])
        
        self.is_running = True
        print("\n[READY] Engine active. Decoding dynamic markers...")
        
        await asyncio.gather(
            self.lsl_worker(inlet),
            self.marker_worker(marker_inlet)
        )

    def _reset_trial_state(self):
        """Reset incremental decoding state for a new trial."""
        self._processed_flash_idx = 0
        self._accumulated_log_probs = {c: -np.log(len(MATRIX_CHARS)) for c in MATRIX_CHARS}

    def _evaluate_accumulated(self, check_threshold):
        """
        Convert accumulated log-probabilities to normalized probabilities
        and return the best character.
        
        Parameters
        ----------
        check_threshold : bool
            If True, only return a character if confidence >= 0.95.
        
        Returns
        -------
        tuple : (predicted_char: str or None, hit_threshold: bool)
        """
        max_log_p = max(self._accumulated_log_probs.values())
        safe_probs = {c: np.exp(val - max_log_p) for c, val in self._accumulated_log_probs.items()}
        
        total_p = sum(safe_probs.values())
        if total_p == 0:
            return None, False
            
        for c in safe_probs:
            safe_probs[c] /= total_p
        
        best_char = max(safe_probs.items(), key=lambda x: x[1])
        
        if check_threshold:
            if best_char[1] >= 0.95:
                return best_char[0], True
            return None, False
            
        return best_char[0], True

    def decode_trial(self, check_threshold=False):
        """
        Decode the current trial using INCREMENTAL Bayesian probability accumulation.
        
        Only processes NEW flashes since the last EVALUATE call, and filters only
        the relevant time window instead of the entire 60-second ring buffer.
        Log-probabilities are accumulated across calls within a trial.
        
        Parameters
        ----------
        check_threshold : bool
            If True, only return a character if confidence >= 0.95 (dynamic stopping).
            If False, always return the best-guess character (fallback mode).
        
        Returns
        -------
        tuple : (predicted_char: str or None, hit_threshold: bool)
        """
        if not self.current_trial_flashes:
            return None, False
            
        if not self.eeg_data:
            print("ERROR: No EEG data in buffer! Is the Unicorn Suite actively broadcasting?")
            return None, False
        
        # --- BUG FIX #1: Only process NEW flashes since last EVALUATE ---
        new_flashes = self.current_trial_flashes[self._processed_flash_idx:]
        
        if not new_flashes:
            # No new flashes — evaluate using already-accumulated probabilities
            if any(v != -np.log(len(MATRIX_CHARS)) for v in self._accumulated_log_probs.values()):
                return self._evaluate_accumulated(check_threshold)
            return None, False
        
        # Convert deque to numpy arrays
        time_arr = np.array(self.eeg_times)
        data_arr = np.array(self.eeg_data)
        
        if data_arr.ndim < 2:
            print("ERROR: EEG data array is malformed or empty.")
            return None, False
            
        data_arr = data_arr[:, :8]  # Keep only 8 EEG channels
        
        # --- BUG FIX #2: Filter only the relevant time window ---
        # Compute the index range needed for unprocessed flashes
        earliest_flash_time = new_flashes[0][0]
        latest_flash_time = new_flashes[-1][0]
        
        idx_start = np.searchsorted(time_arr, earliest_flash_time)
        idx_end = np.searchsorted(time_arr, latest_flash_time)
        
        # Expand window to include baseline before earliest and full epoch after latest
        window_start = max(0, idx_start - BASELINE_SAMPLES - 10)
        window_end = min(len(data_arr), idx_end + SAMPLES_PER_EPOCH + 10)
        
        # Add extra padding on both sides for filtfilt edge stability
        # filtfilt default padlen = 3 * max(len(a), len(b)) - 1 ≈ 26 for our filters
        # We use 100 samples (~400ms) for generous guard band
        filter_pad = 100
        filter_start = max(0, window_start - filter_pad)
        filter_end = min(len(data_arr), window_end + filter_pad)
        
        # Extract and filter only the needed window
        data_window = data_arr[filter_start:filter_end]
        time_window = time_arr[filter_start:filter_end]
        
        if len(data_window) < 50:  # Minimum viable length for filtfilt
            print("WARNING: Insufficient data in window for filtering.")
            return None, False
        
        data_filtered = apply_preprocessing(data_window)
        
        # Epoch only the new flashes
        X_test = []
        groups = []
        rejected_count = 0
        evicted_count = 0
        
        for f_time, group in new_flashes:
            # --- BUG FIX #4: Skip flashes whose data has been evicted ---
            if len(time_window) == 0 or f_time < time_window[0]:
                evicted_count += 1
                continue
            
            epoch = extract_epoch(data_filtered, time_window, f_time, apply_baseline=True)
            
            if epoch is None:
                continue
            
            # Artifact rejection — skip epochs with extreme amplitudes
            if not reject_artifacts(epoch, ARTIFACT_THRESHOLD_UV):
                rejected_count += 1
                continue
                
            X_test.append(epoch)
            groups.append(group)
        
        # Mark all current flashes as processed regardless of outcome
        self._processed_flash_idx = len(self.current_trial_flashes)
        
        if evicted_count > 0:
            with open(self.log_file, "a") as f: f.write(f"\n[DEBUG] Buffer eviction: {evicted_count} flashes skipped (data evicted from ring buffer).\n")
                
        if not X_test:
            # No new valid epochs, but we may have accumulated probabilities from prior calls
            if any(v != -np.log(len(MATRIX_CHARS)) for v in self._accumulated_log_probs.values()):
                with open(self.log_file, "a") as f: f.write(f"\n[DEBUG] No new valid epochs in this batch, using accumulated probabilities from prior {self._processed_flash_idx} flashes.\n")
                return self._evaluate_accumulated(check_threshold)
            print("No valid epochs found in buffer!")
            with open(self.log_file, "a") as f: f.write(f"\n[DEBUG] decode_trial() failed! X_test is entirely empty! None of the {len(new_flashes)} new flashes mapped to EEG buffer (Length: {len(time_arr)}). First EEG Time: {time_arr[0] if len(time_arr) > 0 else 'N/A'}, First Flash Time: {new_flashes[0][0]}\n")
            return None, False
        
        if rejected_count > 0:
            with open(self.log_file, "a") as f: f.write(f"\n[DEBUG] Artifact rejection: {rejected_count}/{rejected_count + len(X_test)} epochs rejected.\n")
            
        with open(self.log_file, "a") as f: f.write(f"\n[DEBUG] Incremental epoching: {len(X_test)} new epochs (total processed: {self._processed_flash_idx}).\n")
        
        X_test = np.array(X_test)
        
        # Classify only the NEW epochs
        if self.active_algo == "ensemble":
            y_probs_lda = self.model_lda.predict_proba(X_test)[:, 1]
            y_probs_mdm = self._mdm_predict_proba(X_test)[:, 1]
            y_probs = (self.ensemble_weight_lda * y_probs_lda + 
                      (1 - self.ensemble_weight_lda) * y_probs_mdm)
        elif self.active_algo == "riemann_mdm":
            y_probs = self._mdm_predict_proba(X_test)[:, 1]
        else:  # xdawn_lda (default fallback)
            y_probs = self.model_lda.predict_proba(X_test)[:, 1]
        
        # Accumulate log-probabilities INCREMENTALLY into persistent state
        for i, group in enumerate(groups):
            p_target = np.clip(y_probs[i], 1e-7, 1.0 - 1e-7)
            
            for c in MATRIX_CHARS:
                if c in group:
                    self._accumulated_log_probs[c] += np.log(p_target)
                else:
                    self._accumulated_log_probs[c] += np.log(1.0 - p_target)
        
        return self._evaluate_accumulated(check_threshold)


async def main():
    agent = RealTimeInference()
    agent.load_and_train_model()
    await agent.main_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\nCRITICAL ENGINE FAILURE: {e}")
        os.system('pause' if os.name == 'nt' else "read -p 'Press Enter to close...'")
