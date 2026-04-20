"""
PsychoPy SSVEP Tryouts Experiment
Provides visually stable SSVEP stimuli (10Hz, 12Hz, 15Hz) with high contrast.
Features clickable GUI buttons to live-switch the flickering frequency.
"""

from psychopy import visual, core, event
import numpy as np

def main():
    print("Initializing PsychoPy Window. This may take a moment...")
    # 1. Setup Window
    # We use a pure black background to ensure the maximum possible contrast when the white box appears.
    # waitBlanking=True is CRITICAL. It ensures frames are swapped exactly during the monitor's hardware 
    # V-Sync, eliminating screen tearing and ensuring precise stimulus durations.
    win = visual.Window(
        size=(1280, 720),
        fullscr=False,
        monitor='testMonitor',
        units='pix',
        color=[-1, -1, -1],  # True Black (-1 to 1 scale in PsychoPy)
        waitBlanking=True
    )
    
    # 2. Hardware Verification Logs
    # This evaluates the true refresh rate so we can confirm if 10/12/15 Hz will divide cleanly.
    fps = win.getActualFrameRate()
    if fps is not None:
        print(f"\n[HARDWARE] Detected screen refresh rate: {fps:.2f} Hz")
        if abs(fps - 60.0) < 1.0 or abs(fps - 120.0) < 1.0:
            print("[HARDWARE] GREAT: Your monitor is ~60Hz or ~120Hz. 10Hz, 12Hz, and 15Hz are perfectly supported.")
        else:
            print("[HARDWARE] WARNING: Your monitor might not be running at 60Hz/120Hz.")
            print("The code will dynamically compensate using elapsed absolute time to keep the mathematical frequency exact, but you may see slight micro-frame jitter.")
    else:
        print("\n[HARDWARE] Warning: Could not detect frame rate. Ensure your OS is set to 60Hz.")

    # 3. Setup Core Stimuli
    # "High Contrast" -> Box is White [1,1,1] and Text is Black [-1,-1,-1]
    box = visual.Rect(
        win=win,
        width=500,
        height=500,
        fillColor=[1, 1, 1],  
        lineColor=[1, 1, 1]
    )
    
    bci_text = visual.TextStim(
        win=win,
        text="BCI",
        color=[-1, -1, -1],   # Black text inside the white box
        height=150,
        bold=True
    )
    
    # We add a small "Photodiode Patch" in the top left corner.
    # To be "absolute absolute sure" about flicker, hardware guys tape a light sensor here.
    photodiode_patch = visual.Rect(
        win=win, width=80, height=80, pos=(-600, 320), fillColor=[1, 1, 1], lineColor=[1, 1, 1]
    )

    # 4. Setup GUI Buttons
    frequencies = [10, 12, 15]
    current_freq = 10
    
    button_shapes = []
    button_labels = []
    
    btn_y = -300
    btn_w = 200
    btn_h = 60
    spacing = 250
    
    for i, freq in enumerate(frequencies):
        btn_x = (i - 1) * spacing # Centers the 3 buttons
        
        btn = visual.Rect(
            win=win,
            width=btn_w,
            height=btn_h,
            pos=(btn_x, btn_y),
            fillColor=[-0.5, -0.5, -0.5], # Base Grey
            lineColor=[1, 1, 1]
        )
        button_shapes.append(btn)
        
        lbl = visual.TextStim(
            win=win,
            text=f"{freq} Hz",
            pos=(btn_x, btn_y),
            color=[1, 1, 1],
            height=30,
            bold=True
        )
        button_labels.append(lbl)

    status_text = visual.TextStim(
        win=win,
        text=f"Current: {current_freq} Hz. Click buttons to switch or press 'esc' to exit.",
        pos=(0, 320),
        color=[1, 1, 1],
        height=25
    )

    # 5. Input Devices & Clocks
    mouse = event.Mouse(win=win)
    phase_clock = core.Clock()

    print("\n[RUNNING] Starting SSVEP visualizer loop...")
    print("Press ESC inside the window to safely quit.")

    # 6. Main Experiment Loop
    while True:
        # GET TIMING: Extract absolute time spent since clock initialization.
        # This is the secret to guaranteed frequencies: rather than counting frames 
        # (which suffers if a frame drops), we lock the phase natively to physical time.
        t = phase_clock.getTime()
        
        # SQUARE WAVE LOGIC: A sine wave > 0 gives exactly a 50% duty cycle square wave mathematically locked to 't'.
        is_on = np.sin(2 * np.pi * current_freq * t) >= 0
        
        # Handle Mouse Interaction
        if any(mouse.getPressed()):
            for i, btn in enumerate(button_shapes):
                if btn.contains(mouse):
                    if current_freq != frequencies[i]:
                        current_freq = frequencies[i]
                        status_text.text = f"Current: {current_freq} Hz. Click buttons to switch or press 'esc' to exit."
                        print(f"-> Switched SSVEP frequency to {current_freq} Hz")
        
        # Handle Keyboard
        keys = event.getKeys()
        if 'escape' in keys or 'q' in keys:
            print("Exiting Experiment...")
            break
            
        # --- DRAW LAYER 1: The UI ---
        status_text.draw()
        for i, (btn, lbl) in enumerate(zip(button_shapes, button_labels)):
            # Turn the selected button Green, leave others gray
            if current_freq == frequencies[i]:
                btn.fillColor = [0.2, 0.8, 0.2] 
            else:
                btn.fillColor = [-0.5, -0.5, -0.5]
            
            btn.draw()
            lbl.draw()

        # --- DRAW LAYER 2: The SSVEP Flicker ---
        # Instead of dynamically changing opacities (which strains the GPU and creates muddy gradients),
        # we only call .draw() when the system is in the ON phase. 
        # This achieves a pure high-contrast binary toggle.
        if is_on:
            box.draw()
            bci_text.draw()
            photodiode_patch.draw()   
            
        # Send everything to the GPU and block execution UNTIL the screen's hardware VSYNC.
        win.flip()

    win.close()
    core.quit()

if __name__ == "__main__":
    main()
