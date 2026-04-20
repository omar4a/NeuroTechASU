"""
PsychoPy SSVEP Tryouts Experiment
Provides visually stable SSVEP stimuli (10Hz, 12Hz, 15Hz) displayed concurrently.
"""

from psychopy import visual, core, event
import numpy as np

def main():
    print("Initializing PsychoPy Window. This may take a moment...")
    win = visual.Window(
        size=(1280, 720),
        fullscr=False,
        monitor='testMonitor',
        units='pix',
        color=[-1, -1, -1],  
        waitBlanking=True
    )
    
    fps = win.getActualFrameRate()
    if fps is not None:
        print(f"\n[HARDWARE] Detected screen refresh rate: {fps:.2f} Hz")
    else:
        print("\n[HARDWARE] Warning: Could not detect frame rate. Ensure your OS is set to 60Hz.")

    # 3. Setup Core Stimuli
    frequencies = [10, 12, 15]
    boxes = []
    bci_texts = []
    
    box_size = 350
    spacing = 400  # Margin/spacing between boxes
    
    for i, freq in enumerate(frequencies):
        bx = (i - 1) * spacing
        b = visual.Rect(
            win=win, width=box_size, height=box_size, pos=(bx, 0),
            fillColor=[1, 1, 1], lineColor=[1, 1, 1]
        )
        boxes.append(b)
        
        t = visual.TextStim(
            win=win, text=f"{freq} Hz", color=[-1, -1, -1], 
            pos=(bx, 0), height=80, bold=True
        )
        bci_texts.append(t)
    
    photodiode_patch = visual.Rect(
        win=win, width=80, height=80, pos=(-600, 320), fillColor=[1, 1, 1], lineColor=[1, 1, 1]
    )

    status_text = visual.TextStim(
        win=win,
        text="All frequencies flashing. Look at a target and check the Real-Time Classifier.",
        pos=(0, 320),
        color=[1, 1, 1],
        height=25
    )

    phase_clock = core.Clock()

    print("\n[RUNNING] Starting SSVEP visualizer loop...")
    print("Press ESC inside the window to safely quit.")

    while True:
        t = phase_clock.getTime()
        
        keys = event.getKeys()
        if 'escape' in keys or 'q' in keys:
            print("Exiting Experiment...")
            break
            
        # Draw static elements
        status_text.draw()

        # Draw flashing boxes
        for i, freq in enumerate(frequencies):
            is_on = np.sin(2 * np.pi * freq * t) >= 0
            if is_on:
                boxes[i].draw()
                bci_texts[i].draw()
                
        # Photodiode patch locked to 10 Hz for hardware sync checks
        if np.sin(2 * np.pi * 10 * t) >= 0:
            photodiode_patch.draw()   
            
        win.flip()

    win.close()
    core.quit()

if __name__ == "__main__":
    main()
