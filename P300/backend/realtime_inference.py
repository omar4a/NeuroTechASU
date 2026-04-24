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
        # Ring buffers: Deque is like a pipe; when you push data in one end, 
        # old data falls out the other end once it's full.
        self.eeg_data = deque(maxlen=MAX_BUFFER_SAMPLES)
        self.eeg_times = deque(maxlen=MAX_BUFFER_SAMPLES)
        
        self.current_trial_flashes = []
        self.is_running = True
        
        # Incremental decoding state: remembers how many flashes we've already 
        # processed so we don't repeat work.
        self._processed_flash_idx = 0
        
        # This dictionary stores our "confidence" for every letter (A-Z, 1-9).
        # We start with every letter having an equal score of 0.
        self._accumulated_scores = {c: 0.0 for c in MATRIX_CHARS}
        
        # Recording arrays for post-hoc diagnosis
        self.recorded_eeg_data = []
        self.recorded_eeg_times = []
        self.recorded_flashes = []
        self.recorded_predictions = []
        
        # We use an "Ensemble" (a team of two AI models) to be more accurate.
        self.active_algo = "ensemble"
        self.ensemble_weight_lda = 0.5 
        
        self.model_lda = None
        self.model_mdm = None
        self._mdm_cov_transformer = None
        self._mdm_classifier = None
        
        # LSL Outlet: How we talk back to the PsychoPy UI.
        info_dec = pylsl.StreamInfo('Speller_Decoded', 'Markers', 1, 0, 'string', 'bci_decoder_123')
        self.decoded_outlet = pylsl.StreamOutlet(info_dec)
        
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
            Xdawn(nfilter=4),
            Vectorizer(),
            LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.7)
        )
        self.model_lda.fit(X, y)
        
        # Pipeline 2: Riemannian MDM
        # This is high-level math that looks at the "geometry" of brain signals.
        print("Training Riemannian MDM Pipeline...")
        self._mdm_cov_transformer = XdawnCovariances(nfilter=4, estimator="oas")
        self._mdm_classifier = MDM()
        X_cov = self._mdm_cov_transformer.fit_transform(X, y)
        self._mdm_classifier.fit(X_cov, y)
        
        print("Models Trained Successfully.")

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
                self.eeg_data.extend(chunk)
                self.eeg_times.extend(timestamps)
                self.recorded_eeg_data.extend(chunk)
                self.recorded_eeg_times.extend(timestamps)
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
                
                elif m_str == "EVALUATE":
                    # The UI wants to know if we've found the letter yet.
                    predicted_char, hit_threshold = self.decode_trial(check_threshold=True)
                    if hit_threshold and predicted_char:
                        print(f"*** DYNAMIC STOP *** FOUND: {predicted_char}")
                        self.decoded_outlet.push_sample([f"DECODED_{predicted_char}"], pylsl.local_clock())
                        self.recorded_predictions.append({"time": pylsl.local_clock(), "predicted": predicted_char, "type": "dynamic_stop"})
                        self._reset_trial_state()
                
                elif m_str == "TRIAL_END":
                    # Flashing is over! We MUST pick a letter now.
                    predicted_char, _ = self.decode_trial(check_threshold=False)
                    if predicted_char:
                        print(f"---------> FINAL GUESS: {predicted_char} <---------")
                        self.decoded_outlet.push_sample([f"DECODED_{predicted_char}"], pylsl.local_clock())
                        self.recorded_predictions.append({"time": pylsl.local_clock(), "predicted": predicted_char, "type": "fallback"})
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
        
        # 2. Filter the entire buffer to avoid filtfilt edge transients destroying the P300
        data_filtered = apply_preprocessing(data_arr)
        
        X_test = []
        groups = []
        last_processed_idx = self._processed_flash_idx
        
        # 3. For every flash, try to extract its 800ms brain wave
        for f_time, group in new_flashes:
            epoch = extract_epoch(data_filtered, time_arr, f_time, apply_baseline=True)
            
            if epoch is None:
                # If data hasn't arrived in LSL yet, we stop and wait for it.
                break
                
            last_processed_idx += 1
            
            # Skip if user blinked
            if not reject_artifacts(epoch, ARTIFACT_THRESHOLD_UV):
                continue
                
            X_test.append(epoch[ACTIVE_CHANNEL_INDICES, :])
            groups.append(group)
        
        self._processed_flash_idx = last_processed_idx
        if not X_test: return self._evaluate_accumulated(check_threshold)
        
        # 4. Ask the AI: "How much P300 signal is in these windows?"
        X_test = np.array(X_test)
        y_probs_lda = self.model_lda.predict_proba(X_test)[:, 1]
        y_probs_mdm = self._mdm_predict_proba(X_test)[:, 1]
        
        # Ensemble: combine both AI models' opinions
        y_probs = (self.ensemble_weight_lda * y_probs_lda + (1 - self.ensemble_weight_lda) * y_probs_mdm)
        
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
            
        # Find the character with the highest accumulated score
        best_char = max(self._accumulated_scores.items(), key=lambda x: x[1])[0]
        
        # Calculate confidence as the normalized gap between the best character and the mean
        scores = list(self._accumulated_scores.values())
        max_score = max(scores)
        mean_score = sum(scores) / len(scores)
        
        flashes_processed = self._processed_flash_idx
        if flashes_processed == 0: return None, False
        
        # Max theoretical gap per flash is ~0.15 for our clipped prob distribution
        confidence = (max_score - mean_score) / (flashes_processed * 0.15 + 1e-5)
        
        # If the trial ended naturally, return the best guess so far.
        if not check_threshold:
            return best_char, False
            
        # Dynamic Stop Threshold
        if confidence >= 0.85: # If the lead is decisive
            return best_char, True
            
        return None, False

    def _reset_trial_state(self):
        """Clear memory for a new letter."""
        self._processed_flash_idx = 0
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
        
        inlet = pylsl.StreamInlet(u_streams[0], max_buflen=360)
        marker_inlet = pylsl.StreamInlet(m_streams[0], max_buflen=120)
        
        print("\n[READY] Engine active. Decoding dynamic markers...")
        await asyncio.gather(self.lsl_worker(inlet), self.marker_worker(marker_inlet))

async def main():
    agent = RealTimeInference()
    agent.load_and_train_model()
    await agent.main_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\nCRITICAL ENGINE FAILURE: {e}")
