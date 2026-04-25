"""
Real-Time P300 Inference Engine.

This is the "decoder." It takes the models we trained and uses them to 
guess which letter you are looking at in real-time.

How it works:
1. Training: It loads your saved data and trains two AI models (LDA and MDM).
2. Listening: It buffers EEG data and flash markers from LSL.
3. Bayesian Accumulation: This is the secret sauce. For every flash, it calculates
   the probability that it was a "target." It keeps adding these probabilities 
   up until one letter stands out with > 95% confidence.
4. Output: Once it's sure, it sends the letter back to the UI.
"""

import os
import sys
import numpy as np
import pylsl
import asyncio
from collections import deque

# PsychoPy Runner isolates the environment, stripping standard user site-packages.
# We append the standard Python 3.10 user site (where pyriemann/mne/sklearn live)
# to sys.path. We APPEND so that PsychoPy's own libraries take priority.
_user_site = os.path.expanduser(r"~\AppData\Roaming\Python\Python310\site-packages")
if os.path.isdir(_user_site) and _user_site not in sys.path:
    sys.path.append(_user_site)

# Import SSVEP Classifier
ssvep_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "SSVEP Protocol"))
if ssvep_path not in sys.path:
    sys.path.append(ssvep_path)
from ssvep_realtime import SSVEPClassifier

# Standard scientific BCI libraries
from pyriemann.spatialfilters import Xdawn
from pyriemann.estimation import XdawnCovariances
from pyriemann.classification import MDM
from mne.decoding import Vectorizer
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from signal_processing import (
    FS, EPOCH_LEN, SAMPLES_PER_EPOCH, BASELINE_SAMPLES,
    MATRIX_CHARS, ARTIFACT_THRESHOLD_UV, ACTIVE_CHANNEL_INDICES,
    TRAINING_DATA_DIR,
    apply_preprocessing, reject_artifacts, extract_epoch
)

# 120 seconds of EEG data is kept in memory. Old data is automatically deleted.
MAX_BUFFER_SAMPLES = FS * 120


class RealTimeInference:
    def __init__(self):
        # Hybrid BCI States
        self.bci_mode = "P300" # Toggles between P300 and SSVEP
        self.ssvep_targets = [10.0, 12.0, 15.0, 8.57, 7.5] # 5 distinct options
        self.ssvep_classifier = SSVEPClassifier(targets=self.ssvep_targets)
        import collections
        self.ssvep_buffer = collections.deque(maxlen=int(FS * 1.0)) # 1-second rolling window
        self.ssvep_history = collections.deque(maxlen=4)
        self.roi_indices = [5, 6, 7] # PO7, Oz, PO8 indices from ACTIVE_CHANNEL_INDICES
        # Ring buffers: Deque is like a pipe; when you push data in one end, 
        # old data falls out the other end once it's full.
        self.eeg_data = deque(maxlen=MAX_BUFFER_SAMPLES)
        self.eeg_times = deque(maxlen=MAX_BUFFER_SAMPLES)
        
        self.current_trial_flashes = []
        self.is_running = True
        
        # Synthetic clock tracking
        self.global_sample_count = 0
        self.global_t_anchor = None
        
        # Incremental decoding state: remembers how many flashes we've already
        # processed so we don't repeat work.
        self._processed_flash_idx = 0
        # Count of flashes that actually contributed to the accumulator
        # (extracted an epoch AND passed artifact rejection). Used as the
        # confidence denominator so artifacts don't suppress dynamic stop.
        self._n_decoder_updates = 0

        # This dictionary stores our "confidence" for every letter (A-Z, 1-9).
        # We start with every letter having an equal score of 0.
        self._accumulated_scores = {c: 0.0 for c in MATRIX_CHARS}
        
        # Recording arrays for post-hoc diagnosis
        self.recorded_eeg_data = []
        self.recorded_eeg_times = []
        self.recorded_flashes = []
        self.recorded_predictions = []
        
        # Ensemble weight for LDA vs MDM. Measured: MDM via softmax-over-
        # negative-Riemannian-distance is essentially uncalibrated noise
        # (target p mean 0.513, non-target 0.484, std ~0.02), so mixing it
        # with LDA at 0.5 halves LDA's effective signal. Keep at 1.0 until
        # MDM is replaced with a proper calibration (Platt / isotonic).
        self.active_algo = "ensemble"
        self.ensemble_weight_lda = 1.0
        
        self.model_lda = None
        self.model_mdm = None
        self._mdm_cov_transformer = None
        self._mdm_classifier = None
        
        # LSL Outlet: How we talk back to the PsychoPy UI.
        # We don't create this until the model is trained.
        self.decoded_outlet = None
        
        # Log file for debugging
        self.log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spelled_text.txt")
        
    def load_and_train_model(self):
        """
        Loads the training data and prepares the AI models.
        """
        print("Loading training data to train the live classifier...")
        x_path = os.path.join(TRAINING_DATA_DIR, "X_train.npy")
        y_path = os.path.join(TRAINING_DATA_DIR, "y_train.npy")
        
        if not os.path.exists(x_path):
            raise FileNotFoundError("X_train.npy not found. You must train first!")
            
        X = np.load(x_path) # Brain waves
        y = np.load(y_path) # 1 if Target, 0 if Not
        
        # Filter to use only our 8 EEG channels
        X = X[:, ACTIVE_CHANNEL_INDICES, :]
        
        # Pipeline 1: xDAWN + LDA
        # xDAWN is a "spatial filter" that ignores noise and focuses on the P300.
        # LDA is a classic classifier that draws a line between Target and Non-Target.
        print("Training xDAWN + LDA Pipeline...")
        self.model_lda = make_pipeline(
            Xdawn(nfilter=3),
            Vectorizer(),
            LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.7)
        )
        self.model_lda.fit(X, y)
        
        # Pipeline 2: Riemannian MDM
        # This is high-level math that looks at the "geometry" of brain signals.
        print("Training Riemannian MDM Pipeline...")
        self._mdm_cov_transformer = XdawnCovariances(nfilter=3, estimator="oas")
        self._mdm_classifier = MDM()
        X_cov = self._mdm_cov_transformer.fit_transform(X, y)
        self._mdm_classifier.fit(X_cov, y)
        
        print("Models Trained Successfully.")
        
        # Now publish the "Ready" signal for the UI
        info_dec = pylsl.StreamInfo('Speller_Decoded', 'Markers', 1, 0, 'string', 'bci_decoder_123')
        self.decoded_outlet = pylsl.StreamOutlet(info_dec)

    def _mdm_predict_proba(self, X_test):
        """
        Calculates probabilities from the MDM model.
        MDM normally just gives a "Yes/No", but we need a "How sure are you?" percentage.
        """
        X_cov = self._mdm_cov_transformer.transform(X_test)
        from pyriemann.utils.distance import distance
        
        n_samples = X_cov.shape[0]
        n_classes = len(self._mdm_classifier.classes_)
        distances = np.zeros((n_samples, n_classes))
        
        # Measure how "far" the current brain wave is from the "Target" average.
        for j, centroid in enumerate(self._mdm_classifier.covmeans_):
            for i in range(n_samples):
                distances[i, j] = distance(X_cov[i], centroid, metric=self._mdm_classifier.metric)
        
        # Convert distance to probability (closer = higher probability)
        neg_distances = -distances
        neg_distances -= neg_distances.max(axis=1, keepdims=True)
        exp_neg_d = np.exp(neg_distances)
        probs = exp_neg_d / exp_neg_d.sum(axis=1, keepdims=True)
        return probs

    async def lsl_worker(self, inlet):
        """Continuously pulls EEG data into our 120-second ring buffer."""
        print("Synchronizing LSL clocks... (3 seconds)")
        await asyncio.sleep(3.0)
        print("[READY] Clock sync complete. Pulling data.")
        
        while self.is_running:
            chunk, timestamps = inlet.pull_chunk()
            if timestamps:
                timestamps = np.array(timestamps)
                n_samples = len(timestamps)
                
                # 1. Calculate ideal time offsets from the start of the session
                chunk_ideal_offsets = (self.global_sample_count + np.arange(n_samples)) / FS
                
                # 2. Find the anchor (minimum delay) for this chunk
                chunk_anchor = np.min(timestamps - chunk_ideal_offsets)
                
                # 3. Update the global true hardware clock anchor
                if self.global_t_anchor is None:
                    self.global_t_anchor = chunk_anchor
                else:
                    self.global_t_anchor = min(self.global_t_anchor, chunk_anchor)
                    
                # 4. Generate perfectly uniform timestamps using the true anchor
                synthetic_timestamps = self.global_t_anchor + chunk_ideal_offsets
                
                self.global_sample_count += n_samples
                self.eeg_data.extend(chunk)
                self.eeg_times.extend(synthetic_timestamps)
                self.recorded_eeg_data.extend(chunk)
                self.recorded_eeg_times.extend(synthetic_timestamps)
                
                if self.bci_mode == "SSVEP":
                    for sample in chunk:
                        roi_data = [sample[idx] for idx in self.roi_indices]
                        self.ssvep_buffer.append(roi_data)
                        
                        if len(self.ssvep_buffer) == self.ssvep_buffer.maxlen:
                            X = np.array(self.ssvep_buffer).T
                            self.ssvep_buffer.clear()
                            
                            pred, _ = self.ssvep_classifier.classify_fbcca(X)
                            self.ssvep_history.append(pred)
                            
                            if len(self.ssvep_history) == self.ssvep_history.maxlen:
                                import collections
                                counter = collections.Counter(self.ssvep_history)
                                most_common_pred, count = counter.most_common(1)[0]
                                if count > 2:
                                    print(f"*** SSVEP STOP *** FOUND FREQ: {most_common_pred}")
                                    self.decoded_outlet.push_sample([f"SSVEP_DECODED_{most_common_pred}"], pylsl.local_clock())
                                    self.ssvep_history.clear()
                                    self.bci_mode = "P300" # Auto-reset
            await asyncio.sleep(0.01)

    async def marker_worker(self, marker_inlet):
        """
        Listens to the UI and decides when to try and decode a letter.
        """
        while self.is_running:
            marker, timestamp = marker_inlet.pull_sample(timeout=0.0)
            if timestamp is not None and marker is not None:
                m_str = marker[0]
                
                if m_str == "SESSION_START":
                    self._reset_trial_state()
                    print("New letter trial starting...")
                
                elif m_str.startswith("FLASH_GROUP_"):
                    # The UI tells us which letters just flashed
                    group = list(m_str.replace("FLASH_GROUP_", ""))
                    self.current_trial_flashes.append((timestamp, group))
                    self.recorded_flashes.append({"time": timestamp, "group": group})
                
                elif m_str == "SSVEP_START":
                    print("Switching engine to SSVEP Decoding...")
                    self.bci_mode = "SSVEP"
                    self.ssvep_buffer.clear()
                    
                elif m_str == "SSVEP_STOP":
                    print("Switching engine to P300 Decoding...")
                    self.bci_mode = "P300"
                
                elif m_str == "EVALUATE":
                    # The UI wants to know if we've found the letter yet.
                    # Guard: Don't perform inference until model is trained
                    if self.model_lda is not None:
                        predicted_char, hit_threshold = self.decode_trial(check_threshold=True)
                        if hit_threshold and predicted_char:
                            print(f"*** DYNAMIC STOP *** FOUND: {predicted_char}")
                            self.decoded_outlet.push_sample([f"DECODED_{predicted_char}"], pylsl.local_clock())
                            self.recorded_predictions.append({"time": pylsl.local_clock(), "predicted": predicted_char, "type": "dynamic_stop"})
                            self._reset_trial_state()
                
                elif m_str == "TRIAL_END":
                    # Flashing is over! We MUST pick a letter now.
                    # But we must wait for the last flashes to actually arrive in the EEG buffer!
                    # (With 220ms offset + 800ms epoch, the last flash needs 1.02s of data).
                    print("TRIAL_END received. Waiting for last brainwaves to arrive...")
                    
                    predicted_char = None
                    # Wait up to 3 seconds for the EEG buffer to catch up
                    for _ in range(300): 
                        # Guard: Don't decode until model is trained
                        if self.model_lda is None:
                            await asyncio.sleep(0.01)
                            continue

                        predicted_char, _ = self.decode_trial(check_threshold=False)
                        # Check if we've processed all flashes in this trial
                        if self._processed_flash_idx >= len(self.current_trial_flashes):
                            break
                        await asyncio.sleep(0.01)
                    
                    if predicted_char:
                        print(f"---------> FINAL GUESS: {predicted_char} <---------")
                        self.decoded_outlet.push_sample([f"DECODED_{predicted_char}"], pylsl.local_clock())
                        self.recorded_predictions.append({"time": pylsl.local_clock(), "predicted": predicted_char, "type": "fallback"})
                    else:
                        print("ERROR: Failed to decode letter after 3s timeout.")
                        
                    self._reset_trial_state()
                    
                elif m_str == "SESSION_END":
                    print("\nSaving inference recording for post-hoc diagnosis...")
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    rec_dir = os.path.join(base_dir, "inference_recording")
                    os.makedirs(rec_dir, exist_ok=True)
                    np.save(os.path.join(rec_dir, "eeg_continuous.npy"), np.array(self.recorded_eeg_data))
                    np.save(os.path.join(rec_dir, "eeg_timestamps.npy"), np.array(self.recorded_eeg_times))
                    np.save(os.path.join(rec_dir, "flash_events.npy"), np.array(self.recorded_flashes, dtype=object))
                    np.save(os.path.join(rec_dir, "predictions.npy"), np.array(self.recorded_predictions, dtype=object))
                    print("Inference recording saved to 'inference_recording/' directory.\n")
                    self._reset_trial_state()
                    
            await asyncio.sleep(0.01)

    def decode_trial(self, check_threshold=False):
        """
        The Brain of the Brain-Computer Interface.
        Uses Bayesian math to combine all flashes into a single guess.
        """
        if not self.current_trial_flashes: return None, False
            
        # 1. Grab only the NEW flashes since the last time we checked.
        new_flashes = self.current_trial_flashes[self._processed_flash_idx:]
        if not new_flashes: return self._evaluate_accumulated(check_threshold)

        time_arr = np.array(self.eeg_times)
        data_arr = np.array(self.eeg_data)[:, :8]

        # Safety: deques are extended (data, then times) in lsl_worker with
        # no await between, so lengths should match. Clip just in case a
        # race or partial extension ever slips through.
        n_common = min(len(time_arr), len(data_arr))
        time_arr = time_arr[:n_common]
        data_arr = data_arr[:n_common]

        # 2. Filter the entire buffer to avoid filtfilt edge transients destroying the P300
        data_filtered = apply_preprocessing(data_arr)

        X_test = []
        groups = []
        last_processed_idx = self._processed_flash_idx
        n_stale = 0  # flashes older than the buffer — permanently undecodable
        epoch_wall_spans = []  # for diagnostic on non-uniform LSL delivery

        # 3. For every flash, try to extract its 800ms brain wave
        for f_time, group in new_flashes:
            # Stale: flash timestamp has fallen off the back of the ring
            # buffer (can happen if decode_trial was blocked for > buffer
            # duration). Advance past it so we don't stall forever.
            if len(time_arr) and f_time < time_arr[0]:
                last_processed_idx += 1
                n_stale += 1
                continue

            epoch = extract_epoch(data_filtered, time_arr, f_time, apply_baseline=True)

            if epoch is None:
                # If data hasn't arrived in LSL yet, we stop and wait for it.
                break

            last_processed_idx += 1

            # Diagnostic: sample count-to-wall-time sanity. Unicorn Recorder's
            # LSL timestamps reflect delivery time (bursty), so the 200-sample
            # epoch window sometimes spans 0.5-1.1s instead of 0.8s, degrading
            # classifier input. See investigation in repo issues.
            idx = np.searchsorted(time_arr, f_time)
            if 0 <= idx < len(time_arr) - SAMPLES_PER_EPOCH:
                epoch_wall_spans.append(
                    float(time_arr[idx + SAMPLES_PER_EPOCH - 1] - time_arr[idx])
                )

            # Skip if user blinked
            if not reject_artifacts(epoch, ARTIFACT_THRESHOLD_UV):
                continue

            X_test.append(epoch[ACTIVE_CHANNEL_INDICES, :])
            groups.append(group)
            # Only count flashes that actually feed the accumulator so the
            # confidence denominator is correct.
            self._n_decoder_updates += 1

        if n_stale:
            print(f"[WARN] decode_trial dropped {n_stale} stale flash(es) "
                  f"older than buffer start.")
        if epoch_wall_spans:
            spans = np.asarray(epoch_wall_spans)
            bad = int((np.abs(spans - EPOCH_LEN) > 0.15).sum())
            if bad:
                print(f"[WARN] {bad}/{len(spans)} epochs span "
                      f"wall-time outside 0.65-0.95s "
                      f"(min={spans.min():.3f}s, max={spans.max():.3f}s). "
                      f"LSL EEG stream is bursty — classifier input is "
                      f"misaligned with training data.")
        
        self._processed_flash_idx = last_processed_idx
        if not X_test: return self._evaluate_accumulated(check_threshold)
        
        # 4. Ask the AI: "How much P300 signal is in these windows?"
        X_test = np.array(X_test)
        y_probs_lda = self.model_lda.predict_proba(X_test)[:, 1]

        # Skip MDM when LDA is doing all the work — saves a full
        # Riemannian-distance pass per EVALUATE.
        if self.ensemble_weight_lda >= 1.0:
            y_probs = y_probs_lda
        else:
            y_probs_mdm = self._mdm_predict_proba(X_test)[:, 1]
            y_probs = (self.ensemble_weight_lda * y_probs_lda
                       + (1 - self.ensemble_weight_lda) * y_probs_mdm)
        
        # 5. Robust Additive Update: For every character, update its score.
        # Instead of using brittle log-probabilities, we simply add the raw probabilities.
        # This prevents a single overconfident false-negative from destroying a character's score.
        for i, group in enumerate(groups):
            p_target = np.clip(y_probs[i], 0.0, 1.0)
            for c in MATRIX_CHARS:
                if c in group:
                    self._accumulated_scores[c] += p_target
                else:
                    self._accumulated_scores[c] += (1.0 - p_target)
        
        return self._evaluate_accumulated(check_threshold)

    def _evaluate_accumulated(self, check_threshold):
        """Check if any letter has a decisively high score."""
        if not self.current_trial_flashes:
            return None, False

        scores_arr = np.fromiter(self._accumulated_scores.values(), dtype=np.float64)

        # Guard: if no flash ever produced a valid epoch, the accumulator is
        # still at the uniform prior. Returning max(dict) here is a silent
        # bias toward MATRIX_CHARS[0] ('A'). Surface the failure instead so
        # the UI can prompt a repeat or the session can be flagged.
        if np.ptp(scores_arr) < 1e-9:
            return None, False

        # Use successfully-updated-flash count as the denominator, not
        # _processed_flash_idx — the latter counts artifact-rejected flashes
        # that did NOT contribute to the accumulator, artificially suppressing
        # confidence in high-artifact sessions.
        flashes_processed = self._n_decoder_updates
        if flashes_processed == 0:
            return None, False

        best_char = max(self._accumulated_scores.items(), key=lambda kv: kv[1])[0]

        # If the trial ended naturally, return the best guess so far.
        if not check_threshold:
            return best_char, False

        # Dynamic stop is disabled until a proper posterior is in place.
        #
        # The prior formula
        #
        #     confidence = (max_score - mean_score) / (flashes * k)
        #
        # is dimensionally (signal / flashes) — both numerator and
        # denominator scale linearly with flash count, so the ratio is
        # essentially constant after the first block, independent of how
        # much evidence has accumulated. That caused early firing on a
        # still-noisy accumulator and committed the wrong letter (target
        # LMFAO → HAMSE on the Apr-24 session; raising the threshold
        # from 0.85 to 0.99 didn't fix it because the ratio is nearly
        # constant with N). A correct dynamic stop needs a posterior
        # whose concentration grows with N — a log-prob Bayesian
        # accumulator with bounded p, or a rank-stability heuristic
        # across consecutive EVALUATE calls. Until that is rebuilt,
        # rely on TRIAL_END's fallback path, which uses the full
        # args.blocks × events-per-block budget before committing
        # (≥ 95 % accuracy at 6 blocks in simulation).
        return None, False

    def _reset_trial_state(self):
        """Clear memory for a new letter."""
        self._processed_flash_idx = 0
        self._n_decoder_updates = 0
        self.current_trial_flashes = []
        self._accumulated_scores = {c: 0.0 for c in MATRIX_CHARS}

    async def main_loop(self):
        print("Resolving LSL Streams...")
        loop = asyncio.get_running_loop()
        u_streams = await loop.run_in_executor(None, pylsl.resolve_byprop, 'name', 'UnicornRecorderLSLStream', 1, 10.0)
        m_streams = await loop.run_in_executor(None, pylsl.resolve_byprop, 'name', 'Speller_Markers', 1, 10.0)
        
        if not u_streams:
            print("\n" + "!" * 60)
            print("FATAL ERROR: Could not find UnicornRecorderLSLStream!")
            print("Did you forget to hit START in the Unicorn Recorder?")
            print("!" * 60 + "\n")
            return
            
        if not m_streams:
            print("\nFATAL ERROR: Could not find Speller_Markers stream!\n")
            return
        
        # Use stable processing flags
        p_flags = pylsl.proc_dejitter | pylsl.proc_threadsafe
        inlet = pylsl.StreamInlet(u_streams[0], max_buflen=360, processing_flags=p_flags)
        marker_inlet = pylsl.StreamInlet(m_streams[0], max_buflen=120, processing_flags=p_flags)

        # Start listening immediately in the background
        # This ensures we capture SESSION_START even if we are still training.
        lsl_task = asyncio.create_task(self.lsl_worker(inlet))
        marker_task = asyncio.create_task(self.marker_worker(marker_inlet))
        
        # Train the model in the background while the workers buffer data
        print("Training models in background...")
        await loop.run_in_executor(None, self.load_and_train_model)
        
        print("\n[READY] Engine active and models trained. Decoding dynamic markers...")
        await asyncio.gather(lsl_task, marker_task)

async def main():
    agent = RealTimeInference()
    # main_loop now handles model training internally
    await agent.main_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\nCRITICAL ENGINE FAILURE: {e}")
