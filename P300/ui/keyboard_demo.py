"""Keyboard-only demo for Project 2: Connect Your Brain to ChatGPT.

Runs the full P300/SSVEP speller workflow using keyboard input only.
Designed for screen recording demos — no EEG hardware required.

Flow:
  1. Dialog        → EEG Mode / Keyboard Mode selection
  2. CONTEXT_SELECT  → Press 1 (with context) or 2 (no context)
  3. CONTEXT_P300    → Type context word on grid, press _ to submit
  4. MAIN_SPELLER    → Type letters to spell, predictions after 2+ chars
  5. PREDICTION_SELECT → Press 1-4 to pick a word
  6. LLM_RESPONSE    → AI response displayed; press SPACE (SSVEP Continue)
  7. → Back to MAIN_SPELLER for next sentence
"""

import os
import sys
import re
import threading
import time
import math

import numpy as np
if not hasattr(np, 'alltrue'): np.alltrue = np.all
if not hasattr(np, 'sometrue'): np.sometrue = np.any
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'object'): np.object = object

from psychopy import visual, core, event

# ---------------------------------------------------------------------------
# Backend import
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.join(os.path.dirname(_THIS_DIR), "backend")
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

sys.path.insert(0, _BACKEND_DIR)
from speller import predict_words, respond_to_sentence

# ---------------------------------------------------------------------------
# Visual constants — matching P300/ui/psychopy_speller.py palette
# ---------------------------------------------------------------------------
BG_COLOR = '#121212'
DIM_COLOR = '#262626'
FLASH_COLOR = '#00FF00'
FIXATION_COLOR = '#FF0000'
READY_COLOR = '#555555'
ACCENT_CYAN = '#00FFFF'
ACCENT_TEAL = '#00E5FF'
ACCENT_PURPLE = '#7C4DFF'
TEXT_WHITE = '#FFFFFF'
TEXT_DIM = '#888888'
PANEL_BG = '#1A1A2E'
CARD_BG = '#16213E'
CARD_BORDER = '#0F3460'
SUBMIT_GOLD = '#E2B714'

# Grid — exact same as psychopy_speller.py
MATRIX = [
    ['A', 'B', 'C', 'D', 'E', 'F'],
    ['G', 'H', 'I', 'J', 'K', 'L'],
    ['M', 'N', 'O', 'P', 'Q', 'R'],
    ['S', 'T', 'U', 'V', 'W', 'X'],
    ['Y', 'Z', '1', '2', '3', '4'],
    ['5', '6', '7', '8', '9', '_'],
]

VALID_KEYS = set('abcdefghijklmnopqrstuvwxyz')

def get_grid_pos(ch):
    for r in range(6):
        for c in range(6):
            if MATRIX[r][c] == ch:
                return r, c
    return None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    from psychopy import gui

    # ── Launch Dialog ──
    dlg_info = {
        '01. Input Mode': ['Keyboard Mode', 'EEG Mode'],
    }
    dlg = gui.DlgFromDict(dictionary=dlg_info, sortKeys=False,
                           title="NeuroTech ASU — Project 2 Config")
    if not dlg.OK:
        core.quit()

    if dlg_info['01. Input Mode'] == 'EEG Mode':
        err = gui.Dlg(title="Not Available")
        err.addText("EEG Mode requires hardware. Use Keyboard Mode for demo.")
        err.show()
        core.quit()

    # ── Window — same params as psychopy_speller.py ──
    win = visual.Window(size=[1280, 720], fullscr=True, allowGUI=False,
                        color=BG_COLOR, units='norm', waitBlanking=True,
                        checkTiming=False)
    fps = 60

    # ── Build grid stims — EXACT same positions as psychopy_speller.py ──
    grid_bgs = []
    grid_stims = []
    text_size_base = 0.12
    text_size_pop = 0.18
    x_positions = [-0.6, -0.36, -0.12, 0.12, 0.36, 0.6]
    y_positions = [0.5, 0.3, 0.1, -0.1, -0.3, -0.5]

    for r in range(6):
        row_bgs = []
        row_stims = []
        for c in range(6):
            bg = visual.Rect(win, width=0.2, height=0.17,
                             pos=(x_positions[c], y_positions[r]),
                             fillColor='#1a1a2e', lineColor='#2a2a4a',
                             lineWidth=1, opacity=0.6)
            stim = visual.TextStim(win, text=MATRIX[r][c],
                                   pos=(x_positions[c], y_positions[r]),
                                   color=DIM_COLOR, height=text_size_base,
                                   bold=True)
            row_bgs.append(bg)
            row_stims.append(stim)
        grid_bgs.append(row_bgs)
        grid_stims.append(row_stims)

    # ── UI Chrome ──
    title_bar = visual.Rect(win, width=2.2, height=0.08, pos=(0, 0.92),
                            fillColor='#0a0a14', lineColor=None)
    title_text = visual.TextStim(win, text='NEUROTECH ASU — BCI SPELLER',
                                 pos=(0, 0.92), color=ACCENT_CYAN,
                                 height=0.04, bold=True)

    status_text = visual.TextStim(win, text='', pos=(0, 0.84),
                                  color=TEXT_DIM, height=0.03)

    # Typed text display — bottom bar
    typed_panel = visual.Rect(win, width=1.8, height=0.12, pos=(0, -0.75),
                              fillColor=PANEL_BG, lineColor='#2a2a4a',
                              lineWidth=1, opacity=0.8)
    typed_label = visual.TextStim(win, text='Spelled:', pos=(-0.85, -0.72),
                                  color=TEXT_DIM, height=0.025,
                                  anchorHoriz='left')
    typed_display = visual.TextStim(win, text='', pos=(-0.85, -0.78),
                                    color=TEXT_WHITE, height=0.045,
                                    anchorHoriz='left', wrapWidth=1.6,
                                    bold=True)

    # Current word indicator
    word_label = visual.TextStim(win, text='Current:', pos=(0.45, -0.72),
                                 color=TEXT_DIM, height=0.025,
                                 anchorHoriz='left')
    word_display = visual.TextStim(win, text='', pos=(0.55, -0.72),
                                   color=ACCENT_CYAN, height=0.035,
                                   bold=True, anchorHoriz='left')

    # Context display (top right)
    context_display = visual.TextStim(win, text='', pos=(0.65, 0.84),
                                      color=ACCENT_PURPLE, height=0.03,
                                      bold=True)

    # Instruction bar (very bottom)
    instruction = visual.TextStim(win, text='', pos=(0, -0.92),
                                  color=TEXT_DIM, height=0.025)

    # ── Context P300 input elements (pre-created to avoid per-frame alloc) ──
    ctx_input_box = visual.Rect(win, width=0.8, height=0.08, pos=(0, -0.65),
                                fillColor=CARD_BG, lineColor=ACCENT_PURPLE,
                                lineWidth=2)
    ctx_input_text = visual.TextStim(win, text='', pos=(0, -0.65),
                                     color=ACCENT_PURPLE, height=0.04,
                                     bold=True)

    # ── Prediction screen prefix label (pre-created) ──
    prefix_info = visual.TextStim(win, text='', pos=(0, 0.35),
                                  color=TEXT_DIM, height=0.03)

    # ── Context selection cards ──
    ctx_cards = []
    ctx_mains = []
    ctx_subs = []
    ctx_nums = []
    for i, (label, sub, xp) in enumerate([
        ("WITH CONTEXT", "Provide a topic to guide AI", -0.35),
        ("NO CONTEXT", "General conversation", 0.35),
    ]):
        bg = visual.Rect(win, width=0.55, height=0.5, pos=(xp, 0),
                         fillColor=CARD_BG, lineColor=ACCENT_CYAN,
                         lineWidth=2)
        num = visual.TextStim(win, text=str(i+1), pos=(xp, 0.15),
                              color=ACCENT_CYAN, height=0.08, bold=True)
        lbl = visual.TextStim(win, text=label, pos=(xp, 0.02),
                              color=TEXT_WHITE, height=0.045, bold=True)
        slbl = visual.TextStim(win, text=sub, pos=(xp, -0.08),
                               color=TEXT_DIM, height=0.025, wrapWidth=0.5)
        ctx_cards.append(bg)
        ctx_nums.append(num)
        ctx_mains.append(lbl)
        ctx_subs.append(slbl)

    # ── Prediction cards (4 slots) ──
    pred_cards = []
    pred_labels = []
    pred_nums = []
    pred_xpos = [-0.52, -0.17, 0.17, 0.52]
    for i, px in enumerate(pred_xpos):
        bg = visual.Rect(win, width=0.3, height=0.35, pos=(px, 0.05),
                         fillColor=CARD_BG, lineColor=ACCENT_CYAN,
                         lineWidth=2)
        num = visual.TextStim(win, text=str(i+1), pos=(px, 0.17),
                              color=ACCENT_CYAN, height=0.04, bold=True)
        lbl = visual.TextStim(win, text='', pos=(px, 0.0),
                              color=TEXT_WHITE, height=0.05, bold=True,
                              wrapWidth=0.28)
        pred_cards.append(bg)
        pred_nums.append(num)
        pred_labels.append(lbl)

    # ── LLM Response panel ──
    resp_panel = visual.Rect(win, width=1.6, height=1.0, pos=(0, 0.1),
                             fillColor=PANEL_BG, lineColor=ACCENT_PURPLE,
                             lineWidth=2)
    resp_title = visual.TextStim(win, text='AI RESPONSE', pos=(0, 0.52),
                                 color=ACCENT_PURPLE, height=0.05,
                                 bold=True)
    resp_body = visual.TextStim(win, text='', pos=(0, 0.1),
                                color=TEXT_WHITE, height=0.04,
                                wrapWidth=1.4)
    resp_continue = visual.TextStim(win,
                                    text='[ SSVEP Continue — Press SPACE ]',
                                    pos=(0, -0.38), color=FLASH_COLOR,
                                    height=0.035, bold=True)

    # ── Loading text ──
    loading_text = visual.TextStim(win, text='', pos=(0, 0.05),
                                   color=ACCENT_CYAN, height=0.05,
                                   bold=True)
    loading_sub = visual.TextStim(win, text='', pos=(0, -0.05),
                                  color=TEXT_DIM, height=0.03)

    # ── State ──
    state = "CONTEXT_SELECT"
    context_word = ""
    current_word = ""
    sentence = ""
    predictions = []
    llm_response = ""
    flash_char = None
    flash_time = 0
    response_count = 0

    # Threading for API
    api_result = [None]
    api_done = [False]

    def fetch_predictions(prefix, sent, ctx):
        api_done[0] = False
        api_result[0] = None
        def worker():
            try:
                result = predict_words(prefix=prefix, sentence=sent,
                                       context=ctx)
                clean = []
                for w in result:
                    ws = str(w).strip()
                    if re.match(r'^[A-Za-z\'\-]{1,20}$', ws):
                        clean.append(ws)
                api_result[0] = clean[:4]
            except Exception as e:
                print(f"[PREDICT ERROR] {e}")
                api_result[0] = []
            api_done[0] = True
        threading.Thread(target=worker, daemon=True).start()

    def fetch_response(sent, ctx):
        api_done[0] = False
        api_result[0] = None
        def worker():
            try:
                api_result[0] = respond_to_sentence(sentence=sent,
                                                     context=ctx)
            except Exception as e:
                print(f"[RESPONSE ERROR] {e}")
                api_result[0] = f"Error: {e}"
            api_done[0] = True
        threading.Thread(target=worker, daemon=True).start()

    def draw_grid():
        for r in range(6):
            for c in range(6):
                grid_bgs[r][c].draw()
                s = grid_stims[r][c]
                ch = MATRIX[r][c]
                if flash_char and ch == flash_char and (time.time() - flash_time) < 0.25:
                    s.color = FLASH_COLOR
                    s.height = text_size_pop
                else:
                    s.color = READY_COLOR
                    s.height = text_size_base
                s.draw()

    def draw_chrome():
        title_bar.draw()
        title_text.draw()
        status_text.draw()
        if context_word:
            context_display.setText(f'Context: {context_word}')
            context_display.draw()

    def draw_typed_bar():
        typed_panel.draw()
        typed_label.draw()
        disp = sentence + current_word
        typed_display.setText(disp if disp else '...')
        typed_display.draw()
        word_label.draw()
        cursor = '▌' if int(time.time() * 2) % 2 == 0 else ' '
        word_display.setText(current_word + cursor)
        word_display.draw()

    # ── Main Loop ──
    while True:
        keys = event.getKeys()
        if 'escape' in keys:
            break

        # ── CONTEXT SELECT ──
        if state == "CONTEXT_SELECT":
            status_text.setText('Choose context mode')
            draw_chrome()
            for i in range(2):
                ctx_cards[i].draw()
                ctx_nums[i].draw()
                ctx_mains[i].draw()
                ctx_subs[i].draw()
            instruction.setText('Press 1 for Context mode  |  Press 2 for No Context')
            instruction.draw()
            win.flip()

            if '1' in keys:
                state = "CONTEXT_P300"
                context_word = ""
            elif '2' in keys:
                state = "MAIN_SPELLER"
                context_word = ""

        # ── CONTEXT P300 (type context word) ──
        elif state == "CONTEXT_P300":
            status_text.setText('Type your context word, then press ENTER to submit')
            draw_chrome()
            draw_grid()
            ctx_input_box.draw()
            cursor = '▌' if int(time.time() * 2) % 2 == 0 else ''
            ctx_input_text.setText(f'Context: {context_word}{cursor}')
            ctx_input_text.draw()
            instruction.setText('Type letters | ENTER to submit | BACKSPACE to delete')
            instruction.draw()
            win.flip()

            for k in keys:
                if k in VALID_KEYS:
                    context_word += k.upper()
                    flash_char = k.upper()
                    flash_time = time.time()
                elif k == 'backspace' and context_word:
                    context_word = context_word[:-1]
                elif k == 'return' and context_word:
                    state = "MAIN_SPELLER"
                elif k == 'space':
                    context_word += ' '

        # ── MAIN SPELLER ──
        elif state == "MAIN_SPELLER":
            ctx_info = f' (Context: {context_word})' if context_word else ' (No Context)'
            status_text.setText(f'Spell your message{ctx_info}  |  Responses: {response_count}')
            draw_chrome()
            draw_grid()
            draw_typed_bar()
            instruction.setText('Type letters | SPACE = word break | ENTER = submit | 2+ chars → predictions')
            instruction.draw()
            win.flip()

            for k in keys:
                if k in VALID_KEYS:
                    current_word += k.upper()
                    flash_char = k.upper()
                    flash_time = time.time()
                    if len(current_word) >= 2:
                        fetch_predictions(current_word.lower(), sentence,
                                          context_word.lower() if context_word else "")
                        state = "LOADING_PREDICTIONS"
                elif k == 'backspace':
                    if current_word:
                        current_word = current_word[:-1]
                    elif sentence:
                        parts = sentence.rstrip().rsplit(' ', 1)
                        if len(parts) == 2:
                            sentence = parts[0] + ' '
                            current_word = parts[1]
                        else:
                            current_word = parts[0]
                            sentence = ''
                elif k == 'space' and current_word:
                    sentence += current_word + ' '
                    current_word = ''
                elif k == 'return' and (sentence.strip() or current_word.strip()):
                    full = (sentence + current_word).strip()
                    fetch_response(full, context_word.lower() if context_word else "")
                    state = "LOADING_RESPONSE"

        # ── LOADING PREDICTIONS ──
        elif state == "LOADING_PREDICTIONS":
            draw_chrome()
            draw_grid()
            draw_typed_bar()
            dots = '.' * (int(time.time() * 3) % 4)
            loading_text.setText(f'Fetching predictions{dots}')
            loading_text.pos = (0, 0.7)
            loading_text.draw()
            instruction.setText('Contacting AI...')
            instruction.draw()
            win.flip()

            if api_done[0]:
                predictions = api_result[0] or []
                if len(predictions) >= 1:
                    state = "PREDICTION_SELECT"
                else:
                    state = "MAIN_SPELLER"

        # ── PREDICTION SELECT ──
        elif state == "PREDICTION_SELECT":
            status_text.setText('Select a predicted word')
            draw_chrome()
            draw_typed_bar()

            prefix_info.setText(f'Prefix: "{current_word}"')
            prefix_info.draw()

            for i in range(min(4, len(predictions))):
                pred_cards[i].draw()
                pred_nums[i].draw()
                pred_labels[i].setText(predictions[i])
                pred_labels[i].draw()

            instruction.setText('Press 1-4 to select | BACKSPACE to keep typing')
            instruction.draw()
            win.flip()

            for k in keys:
                if k in ('1', '2', '3', '4'):
                    idx = int(k) - 1
                    if idx < len(predictions):
                        sentence += predictions[idx] + ' '
                        current_word = ''
                        state = "MAIN_SPELLER"
                elif k == 'backspace':
                    state = "MAIN_SPELLER"

        # ── LOADING RESPONSE ──
        elif state == "LOADING_RESPONSE":
            draw_chrome()
            dots = '.' * (int(time.time() * 3) % 4)
            loading_text.pos = (0, 0.05)
            loading_text.setText(f'Generating AI response{dots}')
            loading_text.draw()
            full_sent = (sentence + current_word).strip()
            loading_sub.setText(f'"{full_sent}"')
            loading_sub.draw()
            instruction.setText('Please wait...')
            instruction.draw()
            win.flip()

            if api_done[0]:
                llm_response = api_result[0] or "No response received."
                response_count += 1
                state = "LLM_RESPONSE"

        # ── LLM RESPONSE ──
        elif state == "LLM_RESPONSE":
            status_text.setText(f'AI Response #{response_count}')
            draw_chrome()
            resp_panel.draw()
            resp_title.draw()
            resp_body.setText(llm_response)
            resp_body.draw()
            # Pulsing continue button
            pulse = 0.5 + 0.5 * math.sin(time.time() * 3)
            resp_continue.opacity = 0.5 + 0.5 * pulse
            resp_continue.draw()
            instruction.setText('SSVEP Continue — Press SPACE to return to P300 speller')
            instruction.draw()
            win.flip()

            if 'space' in keys or 'return' in keys:
                sentence = ''
                current_word = ''
                state = "MAIN_SPELLER"

    win.close()
    core.quit()


if __name__ == '__main__':
    run()
