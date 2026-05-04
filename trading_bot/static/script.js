const connectionStatus = document.getElementById('connection-status');
const tickerLabel = document.getElementById('ticker-label');
const modeBadge = document.getElementById('paper-trading-label');
const midPriceEl = document.getElementById('mid-price');
const signalBadge = document.getElementById('signal-badge');
const snnPredEl = document.getElementById('snn-prediction');
const sentimentEl = document.getElementById('sentiment-score');
const capitalEl = document.getElementById('capital-amount');
const posSizeEl = document.getElementById('position-size');
const consoleWindow = document.getElementById('console-window');
const downloadLobBtn = document.getElementById('download-lob-btn');
const downloadStatus = document.getElementById('download-status');

// Slider Elements
const riskSlider = document.getElementById('risk-slider');
const riskVal = document.getElementById('risk-val-display');
const pollSlider = document.getElementById('poll-slider');
const pollVal = document.getElementById('poll-val-display');

let lastLogCount = 0;
let isFirstFetch = true;

// Utility to format numbers
const formatCurrency = (num) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(num);
const formatNumber = (num, decimals) => Number(num).toFixed(decimals);

function updateSignalUI(signal) {
    signalBadge.className = 'signal-badge'; // Reset classes
    if (signal === 1) {
        signalBadge.textContent = 'LONG';
        signalBadge.classList.add('long');
    } else if (signal === -1) {
        signalBadge.textContent = 'SHORT';
        signalBadge.classList.add('short');
    } else {
        signalBadge.textContent = 'HOLD';
        signalBadge.classList.add('hold');
    }
}

function processLogs(logs) {
    if (!logs || logs.length === 0) return;

    // Only append new logs based on array growth (assuming pure append in backend)
    // Actually, backend keeps last 100, so let's just re-render if it changed or just check length difference.
    // For simplicity, we'll re-render all logs and scroll down.

    consoleWindow.innerHTML = '';
    logs.forEach(log => {
        const div = document.createElement('div');
        div.className = 'log-entry';

        // Simple coloring based on content
        if (log.includes('[ERROR]')) {
            div.classList.add('log-error');
        } else if (log.includes('[WARNING]')) {
            div.classList.add('log-warn');
        } else if (log.includes('SIGNAL') || log.includes('Executing')) {
            div.classList.add('log-highlight');
        } else {
            div.classList.add('log-info');
        }

        div.textContent = log;
        consoleWindow.appendChild(div);
    });

    // Auto-scroll to bottom
    consoleWindow.scrollTop = consoleWindow.scrollHeight;
}

async function fetchState() {
    try {
        const response = await fetch('/api/state');
        if (!response.ok) throw new Error('Network response was not ok');

        const data = await response.json();

        // Update Connection Status
        connectionStatus.classList.add('connected');

        // Update Static Labels
        tickerLabel.textContent = data.ticker;

        modeBadge.textContent = data.paper_trading;
        if (data.paper_trading === "Demo Futures") {
            modeBadge.classList.add('demo-badge');
            modeBadge.classList.remove('paper-badge');
        } else {
            modeBadge.classList.add('paper-badge');
            modeBadge.classList.remove('demo-badge');
        }

        // Update Metrics
        midPriceEl.textContent = formatNumber(data.mid_price, 2);
        snnPredEl.textContent = formatNumber(data.snn_prediction, 4);
        sentimentEl.textContent = formatNumber(data.sentiment_score, 4);
        capitalEl.textContent = formatCurrency(data.capital);
        posSizeEl.textContent = formatNumber(data.position_size, 4);

        // Update Signal Component
        updateSignalUI(data.current_signal);

        // Sync slider positions on first launch only to avoid jitter
        if (isFirstFetch) {
            if (data.risk_per_trade !== undefined) {
                const initRisk = (data.risk_per_trade * 100).toFixed(1);
                riskSlider.value = initRisk;
                riskVal.textContent = initRisk + '%';
            }
            if (data.poll_interval !== undefined) {
                pollSlider.value = data.poll_interval;
                pollVal.textContent = data.poll_interval + 's';
            }
            isFirstFetch = false;
        }

        // Update Logs
        processLogs(data.logs);

    } catch (error) {
        console.error('Error fetching state:', error);
        connectionStatus.classList.remove('connected');
    }
}

// Sliders Interactivity
riskSlider.addEventListener('input', (e) => {
    riskVal.textContent = parseFloat(e.target.value).toFixed(1) + '%';
});

pollSlider.addEventListener('input', (e) => {
    pollVal.textContent = e.target.value + 's';
});

// Post changes to Backend Config
async function updateConfig() {
    const newConfig = {
        risk_per_trade: parseFloat(riskSlider.value) / 100.0,
        poll_interval: parseInt(pollSlider.value, 10)
    };

    try {
        await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(newConfig)
        });
        console.log("Config updated successfully:", newConfig);
    } catch (err) {
        console.error("Failed to update config:", err);
    }
}

riskSlider.addEventListener('change', updateConfig);
pollSlider.addEventListener('change', updateConfig);

async function downloadLobCsv() {
    downloadLobBtn.disabled = true;
    downloadStatus.textContent = 'Preparing...';

    try {
        const response = await fetch('/api/download/lob.csv');
        if (!response.ok) throw new Error('Download request failed');

        const blob = await response.blob();
        const contentDisposition = response.headers.get('Content-Disposition') || '';
        const filenameMatch = contentDisposition.match(/filename="([^"]+)"/);
        const filename = filenameMatch ? filenameMatch[1] : 'lob_data.csv';
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');

        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
        downloadStatus.textContent = 'Downloaded';
    } catch (error) {
        console.error('Failed to download LOB CSV:', error);
        downloadStatus.textContent = 'Failed';
    } finally {
        downloadLobBtn.disabled = false;
        setTimeout(() => {
            downloadStatus.textContent = '';
        }, 3000);
    }
}

downloadLobBtn.addEventListener('click', downloadLobCsv);

// Initial fetch
fetchState();

// Poll every 2 seconds
setInterval(fetchState, 2000);
