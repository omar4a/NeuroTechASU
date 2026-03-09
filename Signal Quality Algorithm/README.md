# Signal Quality Algorithm

Live development monitor for Unicorn Hybrid Black signal quality over LSL.

## What this does

- Connects to the raw Unicorn LSL EEG stream named `UnicornRecorderRawDataLSLStream`.
- Displays a simple top-down head figure with 8 electrodes.
- Colors each electrode green or red based on a separate signal-quality engine.

## Quality engine

The quality logic lives in `quality_engine.py` and currently runs a triple-check on the last 1 second of raw data for each channel:

- raw counts are scaled to microvolts using `4500000 / 8388608`
- the 1-second window is detrended by subtracting its mean
- `Vpp` must be between `0.5` and `500.0` uV
- `std_dev` must be below `50.0` uV
- channels with absolute offset above `180000` uV are marked bad as a rail check

The UI now displays `Vpp` under each channel, and the processor can be swapped independently of the Tkinter and LSL code.

Raw samples are scaled before display using:

`4500000 / 8388608`

## Expected channel layout

The monitor assumes the first 8 EEG channels map to:

`Fz, C3, Cz, C4, Pz, PO7, Oz, PO8`

## Run

1. Open Unicorn Recorder.
2. Connect the device.
3. Enable LSL output and start the stream.
4. Create the virtual environment:

   `py -3.13 -m venv .venv`

5. Install Python dependencies:

   `.venv\Scripts\python.exe -m pip install -r "Signal Quality Algorithm\\requirements.txt"`

6. Start the monitor:

   `.venv\Scripts\python.exe "Signal Quality Algorithm\\brain_quality_monitor.py"`

## Notes

- If no LSL stream is found, make sure Recorder is not only connected but actively streaming over LSL.
- The script connects only to `UnicornRecorderRawDataLSLStream` and fails fast if that outlet is missing or not producing samples.
- Raw-count scaling and signal-quality decisions are isolated in `quality_engine.py`.
- The monitor currently uses the first 8 channels when Recorder does not publish channel labels.
