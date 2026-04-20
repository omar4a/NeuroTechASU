"""
SSVEP Pilot Screening Script
Data Source: g.tec Unicorn Hybrid Black (250 Hz)
Channel Array: ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
Region of Interest (ROI): ['PO7', 'Oz', 'PO8']
"""

import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt

def compute_snr(psds, noise_n_neighbor_freqs=3, noise_skip_neighbor_freqs=1):
    """
    Calculate Local SNR based on neighboring frequency bins to isolate neural responses.
    
    Parameters:
    - psds: ndarray, shape (..., n_freqs)
        Power Spectral Density values (from epochs).
    - noise_n_neighbor_freqs: int
        Number of neighboring frequencies to average for baseline noise estimation.
    - noise_skip_neighbor_freqs: int
        Number of adjacent frequencies to skip to prevent spectral leakage from the 
        target bin into the noise estimate.
        
    Returns:
    - snr: ndarray, same shape as psds
        Calculated Signal-to-Noise Ratio reflecting pure response amplitude over surrounding floor.
    """
    n_freqs = psds.shape[-1]
    snr = np.zeros_like(psds)
    
    for i in range(n_freqs):
        # Calculate array indices for the left and right neighbor brackets
        left_idx = np.arange(i - noise_skip_neighbor_freqs - noise_n_neighbor_freqs,
                             i - noise_skip_neighbor_freqs)
        right_idx = np.arange(i + 1 + noise_skip_neighbor_freqs,
                              i + 1 + noise_skip_neighbor_freqs + noise_n_neighbor_freqs)
        
        # Filter out bounds that overshoot the frequency array
        left_idx = left_idx[(left_idx >= 0) & (left_idx < n_freqs)]
        right_idx = right_idx[(right_idx >= 0) & (right_idx < n_freqs)]
        
        neighbors = np.concatenate((left_idx, right_idx))
        
        # Calculate noise baseline as the average of neighbors, then compute SNR relative to it
        if len(neighbors) > 0:
            noise_floor_power = np.mean(psds[..., neighbors], axis=-1)
            # Avoid division by zero on artificially clean signals
            noise_floor_power = np.where(noise_floor_power == 0, np.finfo(float).eps, noise_floor_power)
            
            snr[..., i] = psds[..., i] / noise_floor_power
        else:
            snr[..., i] = 1.0  # Safe default if no valid neighbors exist (at frequency edges)
            
    return snr


def generate_overlapping_events(start_sec, end_sec, sfreq, epoch_dur=4.0, overlap=2.0, event_id=1, margin_sec=1.0):
    """Generate dummy event structure for overlapping window segmentation within a target block."""
    # Apply margin to avoid transition noise
    actual_start = start_sec + margin_sec
    actual_end = end_sec - margin_sec
    step = epoch_dur - overlap
    
    # Establish starts sequentially
    starts = np.arange(actual_start, actual_end - epoch_dur + step, step)
    # Exclude any starts that would push the epoch boundary past the block end
    starts = starts[starts + epoch_dur <= actual_end]
    
    # Format into MNE event spec: (sample_index, previous_value_dummy, event_trigger_id)
    events = np.zeros((len(starts), 3), dtype=int)
    events[:, 0] = (starts * sfreq).astype(int)
    events[:, 2] = event_id
    
    return events


def main():
    # --- 1. CONFIGURATION & INGESTION --- 
    sfreq = 250.0 
    ch_names = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
    ch_types = ['eeg'] * len(ch_names)
    roi_channels = ['PO7', 'Oz', 'PO8']
    
    file_path = 'Ali_SSVEP_Tryout.csv'
    
    try:
        # Load the CSV utilizing pandas.
        df = pd.read_csv(file_path)
        
        # Read matching columns if they exist natively, or default to the first 8 columns.
        if all(ch in df.columns for ch in ch_names):
            data = df[ch_names].values.T
        else:
            print(f"Warning: Columns in '{file_path}' don't directly match expected names. Truncating to first 8 columns.")
            data = df.iloc[:, :8].values.T
            
    except FileNotFoundError:
        print(f"[{file_path}] not found. Generating dummy synthetic candidate data for standard demonstration...")
        
        # Fallback sequence: Synthetically engineer 88.5 seconds with embedded pseudo-SSVEP spikes
        n_samples = int(sfreq * 88.5)
        t = np.arange(n_samples) / sfreq
        data = np.random.randn(len(ch_names), n_samples) * 5e-6 # General background noise floor
        
        # Inject our SSVEP responses primarily inside occipital channels (Region of Interest)
        roi_indices = [ch_names.index(ch) for ch in roi_channels]
        
        # 10 Hz target stimulation (0 - 20 sec)
        for idx in roi_indices:
            data[idx, :int(20.0*sfreq)] += 15e-6 * np.sin(2 * np.pi * 10 * t[:int(20.0*sfreq)])
        # 12 Hz target stimulation (33.5 - 53.5 sec)
        for idx in roi_indices:
            data[idx, int(33.5*sfreq):int(53.5*sfreq)] += 15e-6 * np.sin(2 * np.pi * 12 * t[int(33.5*sfreq):int(53.5*sfreq)])
        # 15 Hz target stimulation (68.5 - 88.5 sec)
        for idx in roi_indices:
            data[idx, int(68.5*sfreq):] += 15e-6 * np.sin(2 * np.pi * 15 * t[int(68.5*sfreq):])

    # Transform numerical matrix to an instantiated MNE Raw space
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info, verbose=False)
    
    # Embed standard 10-20 montage matrix to allow later topographic plotting
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    
    
    # --- 2. SIGNAL PROCESSING PIPELINE --- 
    
    # High-pass filter at 0.1 Hz strictly to strip hardware DC drift. 
    # Notice: NO Common Average Reference (CAR) applied as requested by specifications.
    print("Applying 0.1 Hz High-Pass Filter against DC Offset...")
    raw.filter(l_freq=0.1, h_freq=None, fir_design='firwin', verbose=False)
    
    # Slicing logic: The file houses 88.5 seconds.
    # 0-20s: 10Hz | 20-33.5s: baseline | 33.5-53.5s: 12Hz | 53.5-68.5s: baseline | 68.5-88.5s: 15Hz
    # We formulate distinct overlapping 4-second epochs bounding exactly within these blocks.
    # We use a margin parameter (1.0 second) to exclude the imperfect transition edges.
    print("Isolating epochs with 4.0s windows, 2.0s overlap, and 1.0s edge margins...")
    ev_10hz = generate_overlapping_events(0.0,  20.0, sfreq, epoch_dur=4.0, overlap=2.0, event_id=10, margin_sec=1.0)
    ev_12hz = generate_overlapping_events(33.5, 53.5, sfreq, epoch_dur=4.0, overlap=2.0, event_id=12, margin_sec=1.0)
    ev_15hz = generate_overlapping_events(68.5, 88.5, sfreq, epoch_dur=4.0, overlap=2.0, event_id=15, margin_sec=1.0)
    
    events = np.vstack([ev_10hz, ev_12hz, ev_15hz])
    event_dict = {'10Hz': 10, '12Hz': 12, '15Hz': 15}
    
    epochs = mne.Epochs(raw, events, event_dict, tmin=0, tmax=4.0 - (1/sfreq), 
                        baseline=None, preload=True, verbose=False)


    # --- 3. FREQUENCY ANALYSIS (PSD & SNR) --- 
    
    print("Executing Welch method for Power Spectral Density evaluation...")
    # Leveraging n_per_seg=int(sfreq*4) to achieve exact 0.25 frequency resolution bins
    psd_spectrum = epochs.compute_psd(method='welch', fmin=1.0, fmax=30.0, 
                                      n_per_seg=int(sfreq*4), verbose=False)
    
    psds = psd_spectrum.get_data() # Yields (epochs, channels, freqs)
    freqs = psd_spectrum.freqs
    
    print("Determining custom local SNR isolating parameters...")
    snrs = compute_snr(psds, noise_n_neighbor_freqs=3, noise_skip_neighbor_freqs=1)


    # --- 4. CONSOLE LOGGING & CAPTURE --- 
    
    # Quick helper rounding index retriever
    def get_freq_idx(target_f):
        return np.argmin(np.abs(freqs - target_f))
        
    idx_10 = get_freq_idx(10.0)
    idx_12 = get_freq_idx(12.0)
    idx_15 = get_freq_idx(15.0)
    
    # Calculate dimensional average over respective epochs
    snr_10hz_block = snrs[epochs.events[:, 2] == 10].mean(axis=0) # shape: (channels, freqs)
    snr_12hz_block = snrs[epochs.events[:, 2] == 12].mean(axis=0)
    snr_15hz_block = snrs[epochs.events[:, 2] == 15].mean(axis=0)
    
    oz_idx = ch_names.index('Oz')
    
    cand_name = file_path.replace('.csv', '')
    res_file = f"{cand_name}_results.txt"
    report = (
        f"\n" + "="*45 + "\n"
        f"    *** SSVEP PILOT SCREENING RESULTS ***\n"
        f"    Candidate: {cand_name}\n"
        f"=============================================\n"
        f"Target 10 Hz SNR at ROI (Oz): {snr_10hz_block[oz_idx, idx_10]:.2f}\n"
        f"Target 12 Hz SNR at ROI (Oz): {snr_12hz_block[oz_idx, idx_12]:.2f}\n"
        f"Target 15 Hz SNR at ROI (Oz): {snr_15hz_block[oz_idx, idx_15]:.2f}\n"
        f"=============================================\n"
    )
    print(report)
    with open(res_file, 'w') as f:
        f.write(report)


    # --- 5. TOPOGRAPHICAL GRAPHIC GENERATION --- 
    
    print("Rendering Spatial Topographies...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Set a combined colorbar limit to ensure relative coloring makes visual sense across charts
    vmax = max(snr_10hz_block[:, idx_10].max(), snr_12hz_block[:, idx_12].max(), snr_15hz_block[:, idx_15].max())
    vmin = min(1.0, snr_10hz_block[:, idx_10].min(), snr_12hz_block[:, idx_12].min(), snr_15hz_block[:, idx_15].min())

    mne.viz.plot_topomap(snr_10hz_block[:, idx_10], epochs.info, axes=axes[0], show=False, vlim=(vmin, vmax), cmap='Spectral_r')
    axes[0].set_title('Evoked SNR at 10 Hz Target')
    
    mne.viz.plot_topomap(snr_12hz_block[:, idx_12], epochs.info, axes=axes[1], show=False, vlim=(vmin, vmax), cmap='Spectral_r')
    axes[1].set_title('Evoked SNR at 12 Hz Target')
    
    mne.viz.plot_topomap(snr_15hz_block[:, idx_15], epochs.info, axes=axes[2], show=False, vlim=(vmin, vmax), cmap='Spectral_r')
    axes[2].set_title('Evoked SNR at 15 Hz Target')
    
    plt.suptitle("Candidate Pilot Amplitude Distributions (SSVEP Local SNR)", fontsize=16, y=1.05)
    
    # Save the output visualization dynamically per candidate
    plot_path = f'{cand_name}_aptitude_topoplot.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Generated successfully. Topoplot graphic saved natively into '{plot_path}'.\n")
    

if __name__ == "__main__":
    main()
