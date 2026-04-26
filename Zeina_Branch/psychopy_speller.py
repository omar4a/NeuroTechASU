import argparse
import random
import types
import numpy as np
import re

# Apply NumPy 2.0 backward compatibility monkey-patch for PsychoPy 2023
if not hasattr(np, 'alltrue'): np.alltrue = np.all
if not hasattr(np, 'sometrue'): np.sometrue = np.any
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'object'): np.object = object

from psychopy import visual, core, event
import pylsl

import sys, os
# speller_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "speller")
sys.path.insert(0, "c:\\Users\\pc\\Downloads\\NeuroTechASU-Sandbox\\")
from speller import predict_words, respond_to_sentence

matrixChars = [
    ['A', 'B', 'C', 'D', 'E', 'F'],
    ['G', 'H', 'I', 'J', 'K', 'L'],
    ['M', 'N', 'O', 'P', 'Q', 'R'],
    ['S', 'T', 'U', 'V', 'W', 'X'],
    ['Y', 'Z', '1', '2', '3', '4'],
    ['5', '6', '7', '8', '9', '_']
]

DIM_COLOR = '#262626' # Dimmed white
FLASH_COLOR = '#00FF00'
FIXATION_COLOR = '#FF0000'
BG_COLOR = '#121212'
READY_COLOR = '#555555' # Faint white — used between letters so user can find next target

def get_char_pos(target_char):
    for r in range(6):
        for c in range(6):
            if matrixChars[r][c] == target_char:
                return r, c
    return None

def generate_flash_sequence(mode, target_char=None):
    """Generate a shuffled list of flash-group IDs for one block.

    mode == 1: Row-Column Paradigm (RCP).
        12 events per block: 6 rows (r0-r5) + 6 cols (c0-c5).
        Each char is in exactly 1 row + 1 col -> 2-membership, target
        rate 2/12 ≈ 16.7 %.

    mode == 2: Checkerboard-style Paradigm (CBP).
        12 events per block: 6 stride-diagonal groups (g0-g5) + 6
        rows (r0-r5). The stride groups come from ``(r*2 + k) % 6``,
        which scatters the 6 flashed cells across rows and 3 distinct
        columns — the spatial-adjacency-breaking property that
        motivates CBP over RCP. Each char at (r, c) is in exactly
        1 stride group (k = (c - 2r) mod 6) + 1 row (r), giving
        2-membership with unique (stride, row) signature per char.
        Target rate 2/12 ≈ 16.7 %.

        Historical note:
          - The originally-shipped CBP used only the 6 stride groups
            (g0-g5), giving each char membership 1 → information-
            theoretically incapable of disambiguating 1-of-36 →
            accumulator plateaued at ~14 % accuracy.
          - An intermediate fix used stride+cols, but chars at
            (r, c) and (r+3, c) shared the same (stride, col)
            signature, capping accuracy at ~50 %.
          - Current design uses stride+rows: all 36 signatures
            unique (verified in tests), measured ≥ 93 % accuracy at
            6 blocks on the held-out LDA p-distribution.
    """
    target_pos = get_char_pos(target_char) if target_char else None

    if mode == 1:
        items = ['r0', 'r1', 'r2', 'r3', 'r4', 'r5',
                 'c0', 'c1', 'c2', 'c3', 'c4', 'c5']
    elif mode == 2:
        items = ['g0', 'g1', 'g2', 'g3', 'g4', 'g5',
                 'r0', 'r1', 'r2', 'r3', 'r4', 'r5']
    else:
        raise ValueError(f"unknown mode={mode!r}; expected 1 (RCP) or 2 (CBP)")

    valid = False
    seq = []
    attempts = 0
    while not valid and attempts < 1000:
        attempts += 1
        seq = items[:]
        random.shuffle(seq)
        valid = True
        for i in range(len(seq) - 1):
            curr = seq[i]
            nxt = seq[i+1]

            # Constraint 1: Adjacency avoid — same axis, adjacent index.
            if curr[0] == nxt[0]:
                if abs(int(curr[1]) - int(nxt[1])) == 1:
                    valid = False
                    break

            # Constraint 2: don't flash two target-containing groups back-to-back.
            if target_pos:
                r, c = target_pos
                # Target lives in row r, col c, and in stride group k where
                # (r*2 + k) % 6 == c  ->  k = (c - 2*r) mod 6.
                target_stride_k = (c - 2 * r) % 6
                target_labels = {f'r{r}', f'c{c}', f'g{target_stride_k}'}
                if curr in target_labels and nxt in target_labels:
                    valid = False
                    break
    return seq

def handle_prediction_screen(win, current_word, current_sentence, fps, outlet, dec_inlet):
    # 1. Fetch predictions safely
    try:
        predictions = predict_words(prefix=current_word, context="", sentence="")
    except Exception:
        predictions = []

    # 2. STRICT SANITIZATION: Only allow clean, short, printable words
    VALID_WORD_RE = re.compile(r'^[A-Za-z0-9\-\']{1,18}$')
    clean_preds = []
    for w in predictions:
        w_str = str(w).strip()
        if VALID_WORD_RE.match(w_str) and w_str.lower() != current_word.lower().strip():
            clean_preds.append(w_str)
            
    # 3. GATE: Skip SSVEP entirely if < 2 meaningful suggestions
    if len(clean_preds) < 2:
        return current_word, current_sentence  # P300 continues uninterrupted

    # 4. Prepare exactly 4 UI slots (pad with safe placeholder)
    options = clean_preds[:4]
    while len(options) < 4:
        options.append("...")
        
    options.append("<-")  # Backspace option

    # SSVEP Frequencies mapped to 60Hz frame intervals
    frame_intervals = [6, 5, 4, 7, 8]
    targets = [10.0, 12.0, 15.0, 8.57, 7.5]
    
    # Setup UI elements
    stims = []
    x_pos = [-0.6, -0.3, 0.0, 0.3, 0.6]
    for i, text in enumerate(options):
        stim = visual.TextStim(win, text=text, pos=(x_pos[i], 0), color='#FFFFFF', height=0.1)
        stims.append(stim)
        
    timer_text = visual.TextStim(win, text=" ", pos=(0, -0.5), color='#FF0000', height=0.08)

    # Static delay before SSVEP flashes start
    timer_text.setText("Find your target... ")
    for _ in range(int(1.5 * fps)):
        for stim in stims:
            stim.draw()
        timer_text.draw()
        win.flip()
        
    if outlet:
        import pylsl
        outlet.push_sample(["SSVEP_START"], pylsl.local_clock())

    # 10s timeout loop
    timeout_frames = int(10.0 * fps)
    selected = None

    for frame in range(timeout_frames):
        time_left = 10 - (frame / fps)
        timer_text.setText(f"{time_left:.1f}s ")
        
        for i, stim in enumerate(stims):
            if (frame // (frame_intervals[i] // 2)) % 2 == 0:
                stim.color = '#FFFFFF'
            else:
                 stim.color = '#262626'
            stim.draw()
            
        timer_text.draw()
        win.flip()
        
        # Check backend for SSVEP Decoded Marker
        if dec_inlet:
            marker, _ = dec_inlet.pull_sample(timeout=0.0)
            if marker and marker[0].startswith("SSVEP_DECODED_"):
                try:
                    freq = float(marker[0].replace("SSVEP_DECODED_", ""))
                    if freq in targets:
                        idx = targets.index(freq)
                        selected = options[idx]
                        break
                except ValueError:
                    pass
        
        # Mock SSVEP Selection using keys 1-5 for testing
        keys = event.getKeys(keyList=['1', '2', '3', '4', '5', 'escape'])
        if 'escape' in keys:
            if outlet:
                import pylsl
                outlet.push_sample(["SSVEP_STOP"], pylsl.local_clock())
            core.quit()
        if keys:
            idx = int(keys[0]) - 1
            selected = options[idx]
            break
            
    if outlet:
        import pylsl
        outlet.push_sample(["SSVEP_STOP"], pylsl.local_clock())
        
    # Process selection
    if selected == "<-":
        current_word = current_word[:-1]
    elif selected and selected != "...":
        current_sentence += selected + " "
        current_word = " "
    elif selected == "...":
        current_sentence += current_word + " "
        current_word = " "
        
    return current_word, current_sentence

def display_response_screen(win, sentence, fps):
    response = respond_to_sentence(sentence, "")
    
    resp_text = visual.TextStim(win, text=response, pos=(0, 0), color='#FFFFFF', height=0.08, wrapWidth=1.5)
    inst_text = visual.TextStim(win, text="(Press ESC to exit)", pos=(0, -0.8), color='#555555', height=0.05)
    
    while True:
        resp_text.draw()
        inst_text.draw()
        win.flip()
        if event.getKeys(keyList=['escape']):
            break


def main():
    from psychopy import gui
    
    # Define default settings strictly as strings to prevent PsychoPy's auto-caster from crashing when you backspace box contents!
    exp_info = {
        'Mode (1:RCP, 2:CBP)': '1',
        'Target Word': 'NEUROTECH_BRAIN',
        'Inference Mode': False,
        'Inference Trials': '5',
        'Blocks (Flashes)': '12',
        'Monitor FPS': '60'
    }
    
    # Present a clean graphical dialog window natively within PsychoPy
    dlg = gui.DlgFromDict(dictionary=exp_info, sortKeys=False, title="P300 Speller Config")
    if not dlg.OK:
        core.quit() # User pressed cancel
        
    args = types.SimpleNamespace()
    
    try:
        args.mode = int(exp_info['Mode (1:RCP, 2:CBP)']) if exp_info['Mode (1:RCP, 2:CBP)'] else 1
        args.word = exp_info['Target Word']
        args.inference = exp_info['Inference Mode']
        args.trials = int(exp_info['Inference Trials']) if exp_info['Inference Trials'] else 5
        args.blocks = int(exp_info['Blocks (Flashes)']) if exp_info['Blocks (Flashes)'] else 10
        args.fps = int(exp_info['Monitor FPS']) if exp_info['Monitor FPS'] else 60
    except ValueError:
        dlg2 = gui.Dlg(title="Invalid Input")
        dlg2.addText("You typed letters into a box that requires numbers!")
        dlg2.show()
        core.quit()

    # Create the LSL Marker Outlet
    print("Initializing LSL Marker Outlet...")
    info = pylsl.StreamInfo('Speller_Markers', 'Markers', 1, 0, 'string', 'psychopy_v1')
    outlet = pylsl.StreamOutlet(info)

    # Initialize Window explicitly requesting FullScreen and locking V-Sync
    # This must happen BEFORE subprocess creation, otherwise Pyglet loses foreground focus 
    # and gets permanently stuck calculating the monitor refresh rate!
    win = visual.Window(size=[1280, 720], fullscr=True, allowGUI=False, 
                        color=BG_COLOR, units='norm', waitBlanking=True, checkTiming=False)

    # --- AUTO-LAUNCH BACKEND ---
    import subprocess
    import sys
    import os
    
    ui_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(os.path.dirname(ui_dir), "backend")
    backend_script = os.path.join(backend_dir, "realtime_inference.py" if args.inference else "data_collection.py")
    
    # Resolve the real Python interpreter — PsychoPy Runner sets sys.executable
    # to its own launcher (e.g., psychopy.exe), not the actual python.exe.
    # We need the raw interpreter to launch backend scripts as subprocesses.
    _exe_dir = os.path.dirname(sys.executable)
    _python_exe = os.path.join(_exe_dir, "python.exe")
    if not os.path.isfile(_python_exe):
        _python_exe = sys.executable  # Fallback
    
    print(f"Auto-launching Backend: {backend_script}")
    try:
        # Launch backend. Path/dependency fixes are handled inside the backend scripts.
        subprocess.Popen([_python_exe, backend_script])
        
        # Adaptive handshake: poll for backend's LSL stream instead of hardcoded wait.
        # For inference mode, wait until Speller_Decoded stream appears (model trained).
        # For training mode, wait until data_collection.py has resolved its streams.
        if args.inference:
            print("Waiting for backend to train models and publish Speller_Decoded stream...")
            max_wait = 30  # seconds — generous ceiling for model training
            for i in range(max_wait):
                handshake = pylsl.resolve_byprop('name', 'Speller_Decoded', 1, 1.0)
                if handshake:
                    print(f"Backend ready! Handshake completed in {i+1} seconds.")
                    break
                print(f"  Still waiting... ({i+1}/{max_wait}s)")
            else:
                print(f"WARNING: Backend did not publish Speller_Decoded within {max_wait}s. Proceeding anyway.")
        else:
            # Training mode: backend just needs to resolve LSL streams, which is fast
            print("Waiting 3 seconds for data collection backend to initialize...")
            core.wait(3.0)
    except Exception as e:
        print(f"Failed to auto-start backend: {e}")
                        
    # Setup grid spacing
    grid_stims = []
    text_size_base = 0.15
    text_size_pop = 0.25
    x_positions = [-0.6, -0.36, -0.12, 0.12, 0.36, 0.6]
    y_positions = [0.6, 0.36, 0.12, -0.12, -0.36, -0.6] 

    for r in range(6):
        row_stims = []
        for c in range(6):
            stim = visual.TextStim(win, text=matrixChars[r][c], 
                                   pos=(x_positions[c], y_positions[r]), 
                                   color=DIM_COLOR, height=text_size_base, bold=True)
            row_stims.append(stim)
        grid_stims.append(row_stims)

    # UI labels
    instruction = visual.TextStim(win, text="Initializing...", pos=(0, 0.85), color=FLASH_COLOR, height=0.08)
    typed_text = visual.TextStim(win, text="", pos=(0, -0.85), color='#FFFFFF', height=0.1)
    
    # Pre-compute frames for exact 150ms timings (300ms ISI total)
    # 300ms ISI eliminates ERP temporal overlap entirely.
    # The P300 response lasts 200-400ms; at 300ms ISI each ERP fully resolves
    # before the next flash, maximizing single-trial discriminability.
    # E.g., 60fps -> 9 frames is 150ms
    frames_flash_on = int(round((150.0 / 1000.0) * args.fps))
    frames_flash_off = int(round((150.0 / 1000.0) * args.fps))
    if frames_flash_on < 1: frames_flash_on = 1
    if frames_flash_off < 1: frames_flash_off = 1

    def draw_all():
        for row in grid_stims:
            for stim in row:
                stim.draw()
        instruction.draw()
        typed_text.draw()

    def exec_flash(group_id, target_char):
        chars_flashed = []
        stims_to_pop = []
        
        # Determine group
        if group_id.startswith('r'):
            r = int(group_id[1])
            for c in range(6):
                stims_to_pop.append(grid_stims[r][c])
                chars_flashed.append(matrixChars[r][c])
        elif group_id.startswith('c'):
            c = int(group_id[1])
            for r in range(6):
                stims_to_pop.append(grid_stims[r][c])
                chars_flashed.append(matrixChars[r][c])
        elif group_id.startswith('g'):
            # CBP stride-diagonal group: 6 cells scattered across rows,
            # never two adjacent rows in the same column.
            k = int(group_id[1])
            for r in range(6):
                c = (r * 2 + k) % 6
                stims_to_pop.append(grid_stims[r][c])
                chars_flashed.append(matrixChars[r][c])
                
        tag = 1 if target_char and target_char in chars_flashed else 0
        if target_char:
            marker_str = f"FLASH_{tag}_{''.join(chars_flashed)}"
        else:
            marker_str = f"FLASH_GROUP_{''.join(chars_flashed)}"

        # POP ON State
        for stim in stims_to_pop:
            stim.color = FLASH_COLOR
            stim.height = text_size_pop
        
        draw_all()
        # Flip #1 (0ms onset)
        win.flip()
        # Push marker immediately using native LSL clock to guarantee exact Unicorn sync
        outlet.push_sample([marker_str], pylsl.local_clock())
        
        # Hold ON state for duration (already completed 1 frame via the flip above)
        for _ in range(frames_flash_on - 1):
            draw_all()
            win.flip()
            
        # POP OFF State
        for stim in stims_to_pop:
            stim.color = DIM_COLOR
            stim.height = text_size_base
            
        draw_all()
        win.flip()
        
        # Hold OFF state
        for _ in range(frames_flash_off - 1):
            draw_all()
            win.flip()

    current_word = ""
    current_sentence = ""

    if args.inference:
        print("Connecting to Backend Decoded marker stream (waiting for model training)...")
        dec_inlet = None
        # Patient resolution: model training takes ~15s, so we wait up to 40s
        for i in range(40):
            streams = pylsl.resolve_byprop('name', 'Speller_Decoded', 1, 1.0)
            if streams:
                dec_inlet = pylsl.StreamInlet(streams[0])
                print(f"Backend Ready! Connection established after {i+1}s.")
                break
            print(f"  Still waiting for backend model to train... ({i+1}/40s)")
            
        if not dec_inlet:
            from psychopy import gui
            dlg = gui.Dlg(title="FATAL SYSTEM ERROR")
            dlg.addText("Could not resolve Speller_Decoded stream!")
            dlg.addText("Check the backend console — is the model still training?")
            dlg.show()
            core.quit()
        
        for t in range(args.trials):
            outlet.push_sample(["SESSION_START"], pylsl.local_clock())
            instruction.setText("Freestyle - find your target character...")
            
            # 3 second break
            for _ in range(int(3.0 * args.fps)):
                draw_all()
                win.flip()
                
            instruction.setText("Flashing...")
            
            char_found = None
            for b in range(args.blocks):
                if char_found:
                    break
                    
                seq = generate_flash_sequence(args.mode, None)
                for group in seq:
                    exec_flash(group, None)
             
                # --- 1. CHECK FOR MANUAL KEYBOARD INPUT (TESTING) ---
                keys = event.getKeys()
                if keys:
                    key_pressed = keys[0].upper()
                    
                    if key_pressed == 'ESCAPE':
                        outlet.push_sample(["SESSION_END"], pylsl.local_clock())
                        win.close()
                        core.quit()
                    
                    # If the key matches a character in our matrix, accept it instantly
                    valid_chars = [c for row in matrixChars for c in row]
                    if key_pressed in valid_chars:
                        char_found = key_pressed
                        print(f"[MANUAL OVERRIDE] User typed '{char_found}'. Bypassing BCI decoder.")
                        break  # Breaks the flash loop, char_found is set
                
                # --- 2. CHECK FOR BCI DECODER (ONLY IF NO KEY WAS PRESSED) ---
                # This ensures the backend doesn't override your manual test
                if dec_inlet and not char_found:
                    marker, _ = dec_inlet.pull_sample(timeout=0.0)
                    if marker and marker[0].startswith("DECODED_"):
                        char_found = marker[0].replace("DECODED_", "")
                        break
                            
                # Trigger an evaluation after a full block, but don't stop flashing
                if not char_found:
                    outlet.push_sample(["EVALUATE"], pylsl.local_clock())
            
            # If we reached max blocks without a dynamic stop
            if not char_found:
                instruction.setText("Finalizing... Decoding Fallback...")
                draw_all()
                win.flip()
                outlet.push_sample(["TRIAL_END"], pylsl.local_clock())
                
                # Extended wait timeout to guarantee backend catch-up.
                # We wait up to 10 seconds to completely eliminate race conditions,
                # but it will break instantly as soon as the marker arrives.
                for _ in range(int(10.0 * args.fps)):
                    draw_all()
                    win.flip()
                    if event.getKeys(keyList=['escape']):
                        outlet.push_sample(["SESSION_END"], pylsl.local_clock())
                        win.close()
                        core.quit()
                        
                    if dec_inlet:
                        marker, _ = dec_inlet.pull_sample(timeout=0.0)
                        if marker:
                            print(f"[UI] Received Marker from Backend: {marker[0]}")
                            if marker[0].startswith("DECODED_"):
                                char_found = marker[0].replace("DECODED_", "")
                                break
                            
            if char_found:
                current_word += char_found
                
                if len(current_word) >= 2:
                    current_word, current_sentence = handle_prediction_screen(win, current_word, current_sentence, args.fps, outlet, dec_inlet)
                
                typed_text.setText(current_sentence + current_word)
                
                # 3 sec illuminated preparation period between letters.
                # All letters light up in faint white so the user can locate
                # and focus on the next target before flashing resumes.
                instruction.setText("Find the next letter...")
                for row in grid_stims:
                    for stim in row:
                        stim.color = READY_COLOR
                for _ in range(int(3.0 * args.fps)):
                    draw_all()
                    win.flip()
                # Reset all letters back to dim
                for row in grid_stims:
                    for stim in row:
                        stim.color = DIM_COLOR
                        
        # Inference complete, tell backend to save the recording
        outlet.push_sample(["SESSION_END"], pylsl.local_clock())
        
        display_response_screen(win, current_sentence + current_word, args.fps)
                
    else:
        # SUPERVISED TRAINING
        for char in args.word:
            pos = get_char_pos(char)
            if pos is None:
                continue
            r, c = pos
            target_stim = grid_stims[r][c]
            
            instruction.setText(f"Focus on '{char}'...")
            
            target_stim.color = FIXATION_COLOR
            target_stim.height = text_size_pop
            
            # 3 sec Fixation
            for _ in range(int(3.0 * args.fps)):
                draw_all()
                win.flip()
                
            # Erase fixation
            target_stim.color = DIM_COLOR
            target_stim.height = text_size_base
            
            instruction.setText("Flashing...")
            
            for b in range(args.blocks):
                seq = generate_flash_sequence(args.mode, char)
                for group in seq:
                    exec_flash(group, char)
                    
                    if event.getKeys(keyList=['escape']):
                        win.close()
                        core.quit()
                        
            current_word += char
            typed_text.setText(current_word)
            
            # 3 sec illuminated preparation period between letters.
            # All letters light up in faint white so the user can locate
            # and focus on the next target before flashing resumes.
            instruction.setText("Find the next letter...")
            for row in grid_stims:
                for stim in row:
                    stim.color = READY_COLOR
            for _ in range(int(3.0 * args.fps)):
                draw_all()
                win.flip()
            # Reset all letters back to dim
            for row in grid_stims:
                for stim in row:
                    stim.color = DIM_COLOR
            
            # 20 sec extended rest between words (after every space character)
            if char == '_':
                for countdown in range(30, 0, -1):
                    instruction.setText(f"Word complete! REST: {countdown}s")
                    for _ in range(int(1.0 * args.fps)):
                        draw_all()
                        win.flip()
                        if event.getKeys(keyList=['escape']):
                            win.close()
                            core.quit()
        
        outlet.push_sample(["SESSION_END"], pylsl.local_clock())

    instruction.setText("Session Complete. Esc to exit.")
    draw_all()
    win.flip()
    event.waitKeys(keyList=['escape'])
    
    win.close()
    core.quit()


if __name__ == '__main__':
    main()
