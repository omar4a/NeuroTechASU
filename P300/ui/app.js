// BCI P300 Speller Controller

// --- DOM Elements ---
const p300Matrix = document.getElementById('p300-matrix');
const typedTextSpan = document.getElementById('typed-text');
const instructionDisplay = document.getElementById('instruction-display');
const photodiode = document.getElementById('photodiode');

const matrixChars = [
    ['A', 'B', 'C', 'D', 'E', 'F'],
    ['G', 'H', 'I', 'J', 'K', 'L'],
    ['M', 'N', 'O', 'P', 'Q', 'R'],
    ['S', 'T', 'U', 'V', 'W', 'X'],
    ['Y', 'Z', '1', '2', '3', '4'],
    ['5', '6', '7', '8', '9', '_']
];

// Utility: Find row and col for a char
function getCharPos(char) {
    for (let r = 0; r < 6; r++) {
        for (let c = 0; c < 6; c++) {
            if (matrixChars[r][c] === char) return { r, c };
        }
    }
    return null;
}

// Generate Grid HTML
function initMatrix() {
    let html = '';
    for (let r = 0; r < 6; r++) {
        html += `<tr id="row-${r}">`;
        for (let c = 0; c < 6; c++) {
            html += `<td id="cell-${r}-${c}" class="col-${c}" data-char="${matrixChars[r][c]}">${matrixChars[r][c]}</td>`;
        }
        html += `</tr>`;
    }
    p300Matrix.innerHTML = html;
}

// --- WebSocket & State ---
let ws = null;
let p300FlashingEnabled = false;
let currentMode = 1; // 1: RCP, 2: CBP
let targetChar = null; // For applying Target constraints
let flashSequence = [];
let sequenceIdx = 0;
let stopCurrentTrial = false; // Flag for Dynamic Stopping

// Connect to Backend
document.getElementById('connect-btn').addEventListener('click', () => {
    const url = document.getElementById('ws-url').value;
    currentMode = parseInt(document.getElementById('mode-selector').value, 10);

    ws = new WebSocket(url);

    ws.onopen = () => {
        console.log("WebSocket connected.");
        document.getElementById('connection-modal').classList.add('hidden');
        document.getElementById('speller-container').classList.remove('hidden');
        photodiode.classList.remove('hidden');
        instructionDisplay.innerText = "Connected. Waiting for target word...";
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleBackendCommand(data);
    };

    ws.onerror = (e) => alert("WebSocket error. Ensure Python backend is running.");
});

// Connect for Inference
document.getElementById('inference-btn').addEventListener('click', () => {
    const url = document.getElementById('ws-url').value;
    currentMode = parseInt(document.getElementById('mode-selector').value, 10);

    ws = new WebSocket(url);

    ws.onopen = () => {
        console.log("WebSocket connected for Inference.");
        document.getElementById('connection-modal').classList.add('hidden');
        document.getElementById('speller-container').classList.remove('hidden');
        photodiode.classList.remove('hidden');
        instructionDisplay.innerText = "Connected. Waiting for inference to start...";
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleBackendCommand(data);
    };

    ws.onerror = (e) => alert("WebSocket error. Ensure Python backend is running.");
});

// --- Preview Mode ---
document.getElementById('preview-btn').addEventListener('click', () => {
    currentMode = parseInt(document.getElementById('mode-selector').value, 10);
    document.getElementById('connection-modal').classList.add('hidden');
    document.getElementById('speller-container').classList.remove('hidden');
    photodiode.classList.remove('hidden');

    setTimeout(() => {
        handleBackendCommand({ command: "start_spelling", word: "TECHNOLOGY" });
    }, 1000);
});

async function handleBackendCommand(data) {
    switch (data.command) {
        case "start_spelling":
            await processSpellingWord(data.word);
            break;
        case "start_inference":
            await processInference(data.num_trials || 5);
            break;
        case "type_char":
            typedTextSpan.innerText += data.char;
            stopCurrentTrial = true;
            break;
    }
}

// --- Protocol Flow Logic ---
async function processSpellingWord(word) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ event: "experiment_start", word: word }));
    }
    for (let i = 0; i < word.length; i++) {
        targetChar = word[i];

        // Fixation Phase (3s)
        instructionDisplay.innerText = `Focus on '${targetChar}'...`;
        const pos = getCharPos(targetChar);
        const cell = document.getElementById(`cell-${pos.r}-${pos.c}`);
        cell.classList.add('fixation');

        await sleep(3000);
        cell.classList.remove('fixation');

        instructionDisplay.innerText = `Flashing...`;

        // Flashing Phase (e.g., 5-10 blocks)
        p300FlashingEnabled = true;

        const numBlocks = 10;
        // 1 block = 12 flashes (RCP) or 6 flashes (CBP)
        for (let b = 0; b < numBlocks; b++) {
            generateFlashSequence();
            sequenceIdx = 0;

            // Loop through sequence
            while (sequenceIdx < flashSequence.length) {
                const groupToFlash = flashSequence[sequenceIdx];
                await executeFlash(groupToFlash);
                sequenceIdx++;
            }
        }

        p300FlashingEnabled = false;
        typedTextSpan.innerText += targetChar; // locally append
        await sleep(2000); // Wait before next character
    }
    instructionDisplay.innerText = "Experiment Complete.";
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ event: "experiment_complete" }));
    }
}

async function processInference(numTrials) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        const selectedAlgo = document.getElementById('algo-selector').value;
        ws.send(JSON.stringify({ event: "inference_session_start", algorithm: selectedAlgo }));
    }
    for (let i = 0; i < numTrials; i++) {
        targetChar = null; // No known target in freestyle!
        stopCurrentTrial = false; // Reset for each trial

        // Fixation Phase (3s)
        instructionDisplay.innerText = `Find your desired character and focus on it...`;
        await sleep(3000);

        instructionDisplay.innerText = `Flashing...`;

        p300FlashingEnabled = true;
        const numBlocks = 10;
        for (let b = 0; b < numBlocks; b++) {
            if (stopCurrentTrial) break;

            generateFlashSequence();
            sequenceIdx = 0;
            while (sequenceIdx < flashSequence.length) {
                if (stopCurrentTrial) break;
                const groupToFlash = flashSequence[sequenceIdx];
                await executeFlash(groupToFlash);
                sequenceIdx++;
            }

            if (stopCurrentTrial) break;

            // Pause minimally to capture the last flash's epoch (800ms length + 100ms wiggle)
            await sleep(900);
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ event: "evaluate_block" }));
            }
        }

        p300FlashingEnabled = false;

        if (!stopCurrentTrial) {
            instructionDisplay.innerText = "Decoding your signal...";

            // Trigger backend to evaluate this trial as final fallback
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ event: "inference_done", mode: currentMode }));
            }
        } else {
            instructionDisplay.innerText = "Dynamic Stop! Character decoded.";
        }

        // Wait 4 seconds for a break.
        await sleep(4000);
    }
    instructionDisplay.innerText = "Inference Session Complete.";
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

// --- Flashing Paradigms ---
function generateFlashSequence() {
    if (currentMode === 1) {
        // RCP Mode
        const items = ['r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5'];
        const targetPos = targetChar ? getCharPos(targetChar) : null;

        // Simple rejection sampling to satisfy constraints
        let valid = false;
        let seq = [];
        let attempts = 0;

        while (!valid && attempts < 1000) {
            attempts++;
            seq = [...items].sort(() => Math.random() - 0.5);
            valid = true;
            for (let i = 0; i < seq.length - 1; i++) {
                const curr = seq[i];
                const next = seq[i + 1];

                // Constraint 1: Adjacency 
                if (curr[0] === next[0]) { // both r or both c
                    const idx1 = parseInt(curr[1]);
                    const idx2 = parseInt(next[1]);
                    if (Math.abs(idx1 - idx2) === 1) {
                        valid = false; break;
                    }
                }

                // Constraint 2: Target consecutive
                if (targetPos) {
                    const isCurrTargetR = (curr === 'r' + targetPos.r);
                    const isCurrTargetC = (curr === 'c' + targetPos.c);
                    const isNextTargetR = (next === 'r' + targetPos.r);
                    const isNextTargetC = (next === 'c' + targetPos.c);

                    if ((isCurrTargetR && isNextTargetC) || (isCurrTargetC && isNextTargetR)) {
                        valid = false; break;
                    }
                }
            }
        }
        flashSequence = seq;
    } else {
        // CBP Mode
        // Groups 0..5, where Group k takes matrixChars[r][(r*2 + k) % 6]
        const groups = [0, 1, 2, 3, 4, 5].sort(() => Math.random() - 0.5);
        flashSequence = groups.map(k => `g${k}`);
    }
}

// Perform a single flash (ON for 100ms, OFF for 75ms)
const FLASH_ON = 100;
const FLASH_OFF = 100;

async function executeFlash(groupIdentifier) {
    // Collect chars in this group to send to backend
    let chars = [];

    if (groupIdentifier.startsWith('r')) {
        const r = parseInt(groupIdentifier[1]);
        document.getElementById(`row-${r}`).querySelectorAll('td').forEach(t => {
            t.classList.add('highlight');
            chars.push(t.getAttribute('data-char'));
        });
    } else if (groupIdentifier.startsWith('c')) {
        const c = parseInt(groupIdentifier[1]);
        document.querySelectorAll(`.col-${c}`).forEach(t => {
            t.classList.add('highlight');
            chars.push(t.getAttribute('data-char'));
        });
    } else if (groupIdentifier.startsWith('g')) {
        const k = parseInt(groupIdentifier[1]);
        for (let r = 0; r < 6; r++) {
            const c = (r * 2 + k) % 6;
            const td = document.getElementById(`cell-${r}-${c}`);
            td.classList.add('highlight');
            chars.push(matrixChars[r][c]);
        }
    }

    photodiode.classList.add('active');

    // SEND SYNC Payload
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            event: "flash",
            target_group: chars,
            current_target: targetChar
        }));
    }

    await sleep(FLASH_ON);

    // Turn off
    document.querySelectorAll('#p300-matrix td.highlight').forEach(el => el.classList.remove('highlight'));
    photodiode.classList.remove('active');

    await sleep(FLASH_OFF);
}

// Run init
initMatrix();
