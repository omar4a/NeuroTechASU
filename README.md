# NeuroTech Ain Shams University

**The first NeuroTechX student chapter in Africa & MENA**, founded by Omar AbdAlAal at Ain Shams University in 2025 (50+ members). The chapter hosted the local hub of the **g.tec BCI Spring School & BR41N.IO hackathon (2026)**, runs EEG workshops and paper reviews, and builds real BCIs on the 8-channel **g.tec Unicorn Hybrid Black**.

This repo is the chapter's shared lab bench. The polished, standalone versions live in their own repos.

## Projects

### P300 speller (`P300/`) → cleaned up in [p300-bci-speller](https://github.com/omar4a/p300-bci-speller)
The full development history of the real-time brain-typing speller: data collection, offline evaluation and diagnostics.
- `realtime_inference.py`, `signal_processing.py`: LSL decoder with a synthetic sample clock, ASR and Bayesian dynamic stopping.
- `bci_classifiers.py`, `eegnet_classifier.py`, `evaluate_pipelines.py`: xDAWN+LDA, Riemannian MDM and EEGNet, compared offline.
- `calibrate_epoch_timing.py`: sweeps the post-flash window (ROC-AUC vs. offset). `calibrate_bayesian_kde.py`: fits the score distributions used for evidence accumulation.
- `simulate_realtime.py`: replays recorded sessions through the live decoder for reproducible debugging. `diagnose_*.py` + `diagnostics/`: per-session ERP plots.
- `tests/test_p300_paradigm.py`: paradigm/stimulus tests.

### Signal-quality monitor (`Signal Quality Algorithm/`)
A live Tkinter head map for the raw Unicorn LSL stream. Each electrode is green or red, based on peak-to-peak range, variance and ADC-rail checks over a 1-second window. This was the prototype of the 5-check contact-quality engine that shipped in the [MindMetric app](https://github.com/omar4a/mindmetric-eeg-app).

### SSVEP (`SSVEP Protocol/`, `SSVEP Tryouts/`)
- `ssvep_realtime.py`: real-time CCA decoding of 10 / 12 / 15 Hz targets with harmonics.
- `ssvep_screening.py`, `ssvep_experiment.py`: pilot screening that measures each candidate's SNR per frequency at Oz (e.g. 12.3 at 10 Hz) and draws an aptitude topoplot.

### Research notes (`docs/research/`)
Literature reviews on P300 speller performance and SSVEP algorithms, plus the hackathon project specifications.

## Tech
`Python` · `pylsl` · `pyRiemann` · `MNE` · `scikit-learn` · `PyTorch (EEGNet)` · `PsychoPy` · `Tkinter` · `g.tec Unicorn`
