import argparse
import random
import types
import numpy as np

# Apply NumPy 2.0 backward compatibility monkey-patch for PsychoPy 2023
if not hasattr(np, 'alltrue'): np.alltrue = np.all
if not hasattr(np, 'sometrue'): np.sometrue = np.any
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'object'): np.object = object

from psychopy import visual, core, event
import pylsl

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
    if mode == 1:
        # RCP Mode
        items = ['r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5']
        target_pos = get_char_pos(target_char) if target_char else None
        
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
                
                # Constraint 1: Adjacency avoid
                if curr[0] == nxt[0]:
                    if abs(int(curr[1]) - int(nxt[1])) == 1:
                        valid = False
                        break
                        
                # Constraint 2: Target consecutive avoid
                if target_pos:
                    is_curr_targ_r = (curr == f'r{target_pos[0]}')
                    is_curr_targ_c = (curr == f'c{target_pos[1]}')
                    is_nxt_targ_r = (nxt == f'r{target_pos[0]}')
                    is_nxt_targ_c = (nxt == f'c{target_pos[1]}')
                    
                    if (is_curr_targ_r and is_nxt_targ_c) or (is_curr_targ_c and is_nxt_targ_r):
                        valid = False
                        break
        return seq
    else:
        # CBP Mode
        groups = [0, 1, 2, 3, 4, 5]
        random.shuffle(groups)
        return [f'g{k}' for k in groups]


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

    current_spelled = ""

    if args.inference:
        # FREESTYLE INFERENCE 
        print("Resolving Speller_Decoded Stream from backend...")
        dec_streams = pylsl.resolve_byprop('name', 'Speller_Decoded', 1, 10.0)
        
        if not dec_streams:
            from psychopy import gui
            dlg = gui.Dlg(title="FATAL SYSTEM ERROR")
            dlg.addText("Could not resolve Speller_Decoded stream!")
            dlg.addText("This means realtime_inference.py crashed in the background.")
            dlg.addText("Did you forget to do Supervised Training first? Check the black console!")
            dlg.show()
            core.quit()
            
        dec_inlet = pylsl.StreamInlet(dec_streams[0])
        
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
                    if event.getKeys(keyList=['escape']):
                        outlet.push_sample(["SESSION_END"], pylsl.local_clock())
                        win.close()
                        core.quit()
                        
                    # Check for Decoding marker mid-flash
                    if dec_inlet:
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
                
                # Extended wait timeout to guarantee backend catch-up
                for _ in range(int(3.0 * args.fps)):
                    draw_all()
                    win.flip()
                    if dec_inlet:
                        marker, _ = dec_inlet.pull_sample(timeout=0.0)
                        if marker:
                            print(f"[UI] Received Marker from Backend: {marker[0]}")
                            if marker[0].startswith("DECODED_"):
                                char_found = marker[0].replace("DECODED_", "")
                                break
                            
            if char_found:
                current_spelled += char_found
                typed_text.setText(current_spelled)
                
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
                        
            current_spelled += char
            typed_text.setText(current_spelled)
            
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
