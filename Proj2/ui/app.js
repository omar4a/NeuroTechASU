// BCI Speller Controller

// --- DOM Elements ---
const p300Matrix = document.getElementById('p300-matrix');
const topSection = document.getElementById('top-section');
const bottomSection = document.getElementById('bottom-section');
const typedTextSpan = document.getElementById('typed-text');
const photodiode = document.getElementById('photodiode');

const predictionBoxes = [
    document.getElementById('pred-1'),
    document.getElementById('pred-2'),
    document.getElementById('pred-3')
];

// --- P300 Matrix Setup ---
// Standard 6x6 grid. A-Z, 1-9, plus an underscore for Space
const matrixChars = [
    ['A', 'B', 'C', 'D', 'E', 'F'],
    ['G', 'H', 'I', 'J', 'K', 'L'],
    ['M', 'N', 'O', 'P', 'Q', 'R'],
    ['S', 'T', 'U', 'V', 'W', 'X'],
    ['Y', 'Z', '1', '2', '3', '4'],
    ['5', '6', '7', '8', '9', '_']
];

// Generate Grid HTML
function initMatrix() {
    let html = '';
    for (let r = 0; r < 6; r++) {
        html += `<tr id="row-${r}">`;
        for (let c = 0; c < 6; c++) {
            html += `<td id="cell-${r}-${c}" class="col-${c}">${matrixChars[r][c]}</td>`;
        }
        html += `</tr>`;
    }
    p300Matrix.innerHTML = html;
}

// --- WebSocket & State ---
let ws = null;
let p300FlashingEnabled = false;
let ssvepEnabled = false;
let epochOffset = 0;

// Connect to Backend
document.getElementById('connect-btn').addEventListener('click', () => {
    const url = document.getElementById('ws-url').value;
    const context = document.getElementById('context-selector').value;
    
    // Calculate difference between JS standard epoch and performance.now
    epochOffset = Date.now() - performance.now();
    
    ws = new WebSocket(url);
    
    ws.onopen = () => {
        console.log("WebSocket connected.");
        document.getElementById('connection-modal').classList.add('hidden');
        document.getElementById('speller-container').classList.remove('hidden');
        photodiode.classList.remove('hidden');
        
        ws.send(JSON.stringify({
            event: "init",
            context: context,
            timestamp: performance.now() + epochOffset
        }));
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleBackendCommand(data);
    };
    
    ws.onerror = (e) => alert("WebSocket error. Make sure Python backend is running.");
});

// --- Preview Mode ---
document.getElementById('preview-btn').addEventListener('click', () => {
    document.getElementById('connection-modal').classList.add('hidden');
    document.getElementById('speller-container').classList.remove('hidden');
    photodiode.classList.remove('hidden');

    // Simulate backend sending some data for design preview
    setTimeout(() => {
        handleBackendCommand({ command: "update_predictions", words: ["Focus", "Explore", "Chat"] });
        handleBackendCommand({ command: "start_ssvep" }); // Test the SSVEP flicker
        
        setTimeout(() => {
            handleBackendCommand({ command: "start_flashing" }); // Test the P300 flash
        }, 3000);
        
        let i = 0;
        const text = "HELLO_WORLD";
        setInterval(() => {
            if (i < text.length) {
                handleBackendCommand({ command: "type_char", char: text[i] });
                i++;
            }
        }, 1200);

    }, 500);
});

function handleBackendCommand(data) {
    switch(data.command) {
        case "start_flashing":
            p300FlashingEnabled = true;
            flashP300Loop(performance.now());
            break;
        case "stop_flashing":
            p300FlashingEnabled = false;
            clearHighlights();
            break;
        case "type_char":
            typedTextSpan.innerText += data.char;
            break;
        case "update_predictions":
            updatePredictions(data.words);
            break;
        case "start_ssvep":
            startSSVEP();
            break;
        case "stop_ssvep":
            stopSSVEP();
            break;
    }
}

// --- P300 Flashing Logic (rAF based) ---
const FLASH_ON = 200;  // ms
const FLASH_OFF = 100; // ms
let lastFlashTime = 0;
let isFlashPhase = false;

function flashP300Loop(currentTime) {
    if (!p300FlashingEnabled) return;
    
    requestAnimationFrame(flashP300Loop);
    
    const dt = currentTime - lastFlashTime;
    
    if (isFlashPhase) {
        if (dt >= FLASH_ON) {
            clearHighlights();
            photodiode.classList.remove('active');
            isFlashPhase = false;
            lastFlashTime = currentTime;
        }
    } else {
        if (dt >= FLASH_OFF) {
            triggerRandomFlash(currentTime);
            photodiode.classList.add('active');
            isFlashPhase = true;
            lastFlashTime = currentTime;
        }
    }
}

function triggerRandomFlash(absTime) {
    const isRow = Math.random() > 0.5;
    const rcIdx = Math.floor(Math.random() * 6);
    
    if (isRow) {
        document.getElementById(`row-${rcIdx}`).querySelectorAll('td').forEach(td => td.classList.add('highlight'));
    } else {
        document.querySelectorAll(`.col-${rcIdx}`).forEach(td => td.classList.add('highlight'));
    }
    
    // SEND SYNC INSTANTLY
    if(ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            event: "flash",
            target: isRow ? `row_${rcIdx}` : `col_${rcIdx}`,
            timestamp: absTime + epochOffset
        }));
    }
}

function clearHighlights() {
    document.querySelectorAll('#p300-matrix td.highlight').forEach(el => el.classList.remove('highlight'));
}


// --- SSVEP Logic ---
const freqTargets = [
    { el: document.querySelector('#pred-1 .ssvep-freq-target'), freq: 10, state: false, lastToggle: 0 },
    { el: document.querySelector('#pred-2 .ssvep-freq-target'), freq: 12, state: false, lastToggle: 0 },
    { el: document.querySelector('#pred-3 .ssvep-freq-target'), freq: 15, state: false, lastToggle: 0 }
];

function updatePredictions(words) {
    words.forEach((w, i) => {
        if(predictionBoxes[i]) predictionBoxes[i].querySelector('.pred-text').innerText = w;
    });
}

function startSSVEP() {
    ssvepEnabled = true;
    bottomSection.classList.add('dimmed');
    
    // reset timers
    const now = performance.now();
    freqTargets.forEach(t => t.lastToggle = now);
    requestAnimationFrame(ssvepLoop);
}

function stopSSVEP() {
    ssvepEnabled = false;
    bottomSection.classList.remove('dimmed');
    freqTargets.forEach(t => t.el.classList.remove('highlight-bg-only'));
}

function ssvepLoop(currentTime) {
    if(!ssvepEnabled) return;
    
    requestAnimationFrame(ssvepLoop);
    
    freqTargets.forEach((target, index) => {
        const periodMs = 1000 / target.freq;
        const halfPeriod = periodMs / 2;
        
        if (currentTime - target.lastToggle >= halfPeriod) {
            target.state = !target.state;
            target.lastToggle = currentTime;
            
            if (target.state) {
                target.el.classList.add('highlight-bg-only');
            } else {
                target.el.classList.remove('highlight-bg-only');
            }
            
            // Note: SSVEP targets in this MVP do not emit websockets sync markers every flash
            // because SSVEP analysis usually relies on continuous frequency matching 
            // rather than discrete event potentials, but we could emit here if needed.
        }
    });
}

// Run init
initMatrix();
