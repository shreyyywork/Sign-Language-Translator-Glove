/**
 * ISL Wearable Sensor Glove - Main Application Controller (GitHub Theme)
 * Handles WebSocket telemetry streaming, GitHub UnderlineNav routing,
 * Data Collection, Calibration, ML Training, TinyML C++ Export, and TTS.
 */

class ISLGloveApp {
    constructor() {
        this.ws = null;
        this.visualizer = null;
        this.gestures = [];
        this.activeTab = 'dashboard';
        this.mode = 'simulated';
        
        // Recording state
        this.isRecording = false;
        this.recordingSamples = [];
        this.recordTimer = null;
        this.recordDurationSec = 5;

        // Calibration state
        this.calibrationStep = 0; // 0=idle, 1=relaxed, 2=straight, 3=bent
        this.calibSamples = { relaxed: [], straight: [], bent: [] };

        // TTS & Translation
        this.ttsEnabled = true;
        this.ttsLanguage = 'en'; // 'en' or 'hi'
        this.lastSpokenGesture = null;
        this.lastSpokenTime = 0;
        this.speechDebounceMs = 2500;
        this.translationHistory = [];
        this.sentenceBuffer = [];

        // Hardware Connection state
        this.activeConnTab = 'usb';
        this.hardwareStatus = {
            isConnected: false,
            port: null,
            baudRate: 115200,
            connectionType: 'disconnected',
            packetsReceived: 0,
            packetsDropped: 0,
            uptimeSeconds: 0
        };
        this.webSerialPort = null;
        this.webSerialReader = null;
        this.isWebSerialRunning = false;

        this.init();
    }

    async init() {
        this.initDOM();
        this.initVisualizer();
        await this.loadGestures();
        await this.loadDatasetStats();
        await this.loadEvaluationReport();
        await this.loadSerialPorts();
        this.connectWebSocket();
        this.initSpeechSynth();
        this.checkWebSerialSupport();
    }

    initDOM() {
        // GitHub Underline Tab Navigation
        document.querySelectorAll('.gh-nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const tab = item.getAttribute('data-tab');
                this.switchTab(tab);
            });
        });

        // Mode Toggles
        document.getElementById('btnSimMode')?.addEventListener('click', () => this.setMode('simulated'));
        document.getElementById('btnHwMode')?.addEventListener('click', async () => {
            const isConn = this.hardwareStatus?.is_connected || this.hardwareStatus?.isConnected || this.isWebSerialRunning;
            if (isConn) {
                await this.setMode('hardware');
            } else {
                // If not connected yet, 1-click auto-connect to USB port
                await this.autoConnectUsb();
            }
        });

        // Simulator Gesture Selector
        document.getElementById('simGestureSelect')?.addEventListener('change', (e) => {
            this.setSimGesture(e.target.value);
        });

        // Recording Controls
        document.getElementById('btnStartRecord')?.addEventListener('click', () => this.startRecording());
        document.getElementById('btnStopRecord')?.addEventListener('click', () => this.stopRecording());
        document.getElementById('btnGenerateSimDataset')?.addEventListener('click', () => this.generateSimulatedDataset());

        // Calibration Wizard
        document.getElementById('btnCalibRelaxed')?.addEventListener('click', () => this.recordCalibrationStep('relaxed'));
        document.getElementById('btnCalibStraight')?.addEventListener('click', () => this.recordCalibrationStep('straight'));
        document.getElementById('btnCalibBent')?.addEventListener('click', () => this.recordCalibrationStep('bent'));
        document.getElementById('btnSaveCalibration')?.addEventListener('click', () => this.saveCalibration());

        // ML Training Controls
        document.getElementById('btnTrainBaseline')?.addEventListener('click', () => this.trainBaselineModels());
        document.getElementById('btnTrainLSTM')?.addEventListener('click', () => this.trainLSTMModel());

        // Sentence & Audio
        document.getElementById('btnSpeakSentence')?.addEventListener('click', () => this.speakSentence());
        document.getElementById('btnClearSentence')?.addEventListener('click', () => this.clearSentence());
        document.getElementById('btnToggleTTS')?.addEventListener('click', () => this.toggleTTS());
        document.getElementById('ttsLangSelect')?.addEventListener('change', (e) => {
            this.ttsLanguage = e.target.value;
        });

        // USB Port Connection Controls
        document.getElementById('btnRefreshUsbPorts')?.addEventListener('click', () => this.loadSerialPorts());
        document.getElementById('btnAutoConnectUsb')?.addEventListener('click', () => this.autoConnectUsb());
        document.getElementById('btnConnectUsb')?.addEventListener('click', () => this.connectUsbPort());

        // COM Port / Bluetooth Controls
        document.getElementById('btnRefreshComPorts')?.addEventListener('click', () => this.loadSerialPorts());
        document.getElementById('btnConnectCom')?.addEventListener('click', () => this.connectComPort());

        // Web Serial Browser USB Port
        document.getElementById('btnWebSerialConnect')?.addEventListener('click', () => this.toggleWebSerial());

        // Disconnect Hardware
        document.getElementById('btnDisconnectHardware')?.addEventListener('click', () => this.disconnectHardware());

        // Legacy support
        document.getElementById('btnConnectSerial')?.addEventListener('click', () => this.connectUsbPort());
        document.getElementById('btnRefreshPorts')?.addEventListener('click', () => this.loadSerialPorts());
    }

    initVisualizer() {
        this.visualizer = new HandVisualizer('handCanvas');
    }

    switchTab(tabId) {
        this.activeTab = tabId;
        document.querySelectorAll('.gh-nav-item').forEach(el => {
            el.classList.toggle('active', el.getAttribute('data-tab') === tabId);
        });
        document.querySelectorAll('.tab-content').forEach(el => {
            el.classList.toggle('active', el.id === `tab-${tabId}`);
        });

        // Resize visualizer if switching to visualizer tabs
        if (tabId === 'dashboard' || tabId === 'live') {
            setTimeout(() => this.visualizer?.resize(), 50);
        }
    }

    async loadGestures() {
        try {
            const res = await fetch('/api/gestures');
            const data = await res.json();
            this.gestures = data.gestures || [];

            // Populate selects
            const simSelect = document.getElementById('simGestureSelect');
            const recordSelect = document.getElementById('recordGestureSelect');

            if (simSelect) simSelect.innerHTML = '';
            if (recordSelect) recordSelect.innerHTML = '';

            this.gestures.forEach(g => {
                const opt1 = new Option(`${g.name} (${g.hindi})`, g.id);
                const opt2 = new Option(`${g.name} - ${g.category}`, g.id);
                simSelect?.appendChild(opt1);
                recordSelect?.appendChild(opt2);
            });

            this.renderGesturesGrid();
        } catch (e) {
            console.error('Error loading gestures:', e);
        }
    }

    renderGesturesGrid() {
        const grid = document.getElementById('gesturesVocabularyGrid');
        if (!grid) return;

        grid.innerHTML = '';
        this.gestures.forEach(g => {
            const box = document.createElement('div');
            box.className = 'Box';
            box.style.cursor = 'pointer';
            box.style.marginBottom = '0';
            box.innerHTML = `
                <div class="Box-header">
                    <strong style="font-size:13px; font-family:var(--font-mono);">${g.name}</strong>
                    <span class="Label Label--accent">${g.category}</span>
                </div>
                <div class="Box-body">
                    <div style="font-size:18px; color:var(--color-success-fg); font-weight:600; margin-bottom:6px;">${g.hindi}</div>
                    <p style="font-size:12px; color:var(--color-fg-muted); line-height:1.4;">${g.description}</p>
                </div>
            `;
            box.addEventListener('click', () => {
                this.setSimGesture(g.id);
                const simSel = document.getElementById('simGestureSelect');
                if (simSel) simSel.value = g.id;
            });
            grid.appendChild(box);
        });
    }

    async setMode(newMode) {
        try {
            const res = await fetch('/api/mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: newMode })
            });
            const data = await res.json();
            this.mode = data.mode;
            document.getElementById('btnSimMode')?.classList.toggle('active', this.mode === 'simulated');
            document.getElementById('btnHwMode')?.classList.toggle('active', this.mode === 'hardware');
            
            const dot = document.getElementById('statusModeDot');
            if (dot) {
                dot.className = `status-dot ${this.mode === 'simulated' ? 'sim' : 'online'}`;
            }

            // Toggle dashboard cards between simulator and live hardware
            const simCard = document.getElementById('dashSimulatorCard');
            const hwCard = document.getElementById('dashHardwareStatusCard');
            if (simCard) simCard.style.display = (this.mode === 'simulated') ? 'block' : 'none';
            if (hwCard) hwCard.style.display = (this.mode === 'hardware') ? 'block' : 'none';
        } catch (e) {
            console.error('Error setting mode:', e);
        }
    }

    async setSimGesture(gestureId) {
        try {
            await fetch('/api/simulator/set_gesture', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ gesture: gestureId, user_id: document.getElementById('simUserSelect')?.value || 'U01' })
            });
        } catch (e) {
            console.error('Error updating sim gesture:', e);
        }
    }

    // ==========================================
    // WebSocket Real-time Stream Client
    // ==========================================
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/stream`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('Connected to ISL Sensor WebSocket.');
            const dot = document.getElementById('wsStatusDot');
            const badge = document.getElementById('wsStatusBadge');
            if (dot && badge) {
                dot.className = 'status-dot online';
                badge.className = 'Label Label--success';
                document.getElementById('wsStatusText').textContent = 'Stream Live (50Hz)';
            }
        };

        this.ws.onmessage = (event) => {
            try {
                const payload = JSON.parse(event.data);
                this.handleStreamPayload(payload);
            } catch (e) {
                console.error('WS parse error:', e);
            }
        };

        this.ws.onclose = () => {
            const dot = document.getElementById('wsStatusDot');
            const badge = document.getElementById('wsStatusBadge');
            if (dot && badge) {
                dot.className = 'status-dot offline';
                badge.className = 'Label Label--danger';
                document.getElementById('wsStatusText').textContent = 'Disconnected (Reconnecting...)';
            }
            setTimeout(() => this.connectWebSocket(), 2000);
        };
    }

    handleStreamPayload(payload) {
        if (payload.mode && payload.mode !== this.mode) {
            this.mode = payload.mode;
            document.getElementById('btnSimMode')?.classList.toggle('active', this.mode === 'simulated');
            document.getElementById('btnHwMode')?.classList.toggle('active', this.mode === 'hardware');
            const simCard = document.getElementById('dashSimulatorCard');
            const hwCard = document.getElementById('dashHardwareStatusCard');
            if (simCard) simCard.style.display = (this.mode === 'simulated') ? 'block' : 'none';
            if (hwCard) hwCard.style.display = (this.mode === 'hardware') ? 'block' : 'none';
        }

        if (payload.is_waiting_hardware) {
            const hwStatus = document.getElementById('dashHwStatusText');
            if (hwStatus) {
                hwStatus.textContent = '⏳ Waiting for ESP32 packets...';
                hwStatus.style.color = 'var(--color-attention-fg)';
            }
            return;
        }

        const raw = payload.raw_sample || {};
        const pred = payload.prediction || {};
        const calib = pred.calibrated_features || raw;

        // Update Hardware Status Card if active
        if (this.mode === 'hardware') {
            const hwStatus = document.getElementById('dashHwStatusText');
            if (hwStatus) {
                hwStatus.textContent = '● Streaming Live (50Hz)';
                hwStatus.style.color = 'var(--color-success-fg)';
            }
            const hwRx = document.getElementById('dashHwRxCount');
            if (hwRx && payload.packets_received !== undefined) {
                hwRx.textContent = payload.packets_received.toLocaleString();
            }
        }

        // 1. Update Hand Visualizer
        if (this.visualizer) {
            this.visualizer.update(
                {
                    thumb: calib.thumb,
                    index: calib.index,
                    middle: calib.middle,
                    ring: calib.ring,
                    little: calib.little
                },
                {
                    acc_x: calib.acc_x,
                    acc_y: calib.acc_y,
                    acc_z: calib.acc_z,
                    gyro_x: calib.gyro_x,
                    gyro_y: calib.gyro_y,
                    gyro_z: calib.gyro_z
                }
            );
        }

        // 2. Update Live Telemetry Gauges
        this.updateTelemetryDOM(calib, raw);

        // 3. Update Prediction Displays
        this.updatePredictionDOM(pred);

        // 4. Recording Sample Accumulator
        if (this.isRecording) {
            this.recordingSamples.push(raw);
            const countEl = document.getElementById('recordSampleCounter');
            if (countEl) countEl.textContent = this.recordingSamples.length;
        }

        // 5. Calibration Sample Accumulator
        if (this.calibrationStep > 0) {
            const stepName = ['relaxed', 'straight', 'bent'][this.calibrationStep - 1];
            if (stepName && this.calibSamples[stepName]) {
                this.calibSamples[stepName].push(raw);
            }
        }
    }

    updateTelemetryDOM(calib, raw) {
        const fingers = ['thumb', 'index', 'middle', 'ring', 'little'];
        fingers.forEach(f => {
            const val = calib[f] !== undefined ? calib[f] : 0;
            const pct = Math.min(Math.max(val * 100, 0), 100);
            
            const fillEl = document.getElementById(`flex-${f}-fill`);
            const valEl = document.getElementById(`flex-${f}-val`);
            const rawEl = document.getElementById(`flex-${f}-raw`);

            if (fillEl) fillEl.style.height = `${pct}%`;
            if (valEl) valEl.textContent = `${pct.toFixed(0)}%`;
            if (rawEl) {
                const rawCount = raw[f] !== undefined ? Math.round(raw[f]) : '--';
                rawEl.textContent = `ADC: ${rawCount}`;
            }
        });

        const setVal = (id, v) => {
            const el = document.getElementById(id);
            if (el) el.textContent = (v !== undefined ? Number(v).toFixed(2) : '0.00');
        };

        setVal('val-acc-x', raw.acc_x);
        setVal('val-acc-y', raw.acc_y);
        setVal('val-acc-z', raw.acc_z);
        setVal('val-gyro-x', raw.gyro_x);
        setVal('val-gyro-y', raw.gyro_y);
        setVal('val-gyro-z', raw.gyro_z);
    }

    updatePredictionDOM(pred) {
        const gesture = pred.gesture || 'WAITING...';
        const confidence = pred.confidence || 0.0;
        const textEn = pred.text_en || gesture;
        const textHi = pred.text_hi || '';

        // Dashboard Hero
        const dashName = document.getElementById('dashPredictedGesture');
        const dashHindi = document.getElementById('dashHindiGesture');
        const dashFill = document.getElementById('dashConfidenceFill');
        const dashPct = document.getElementById('dashConfidencePct');
        const dashModel = document.getElementById('dashModelBadge');

        if (dashName) dashName.textContent = textEn;
        if (dashHindi) dashHindi.textContent = textHi;
        if (dashFill) dashFill.style.width = `${(confidence * 100).toFixed(0)}%`;
        if (dashPct) dashPct.textContent = `${(confidence * 100).toFixed(1)}%`;
        if (dashModel) dashModel.textContent = `Model: ${(pred.model_name || 'SVM').toUpperCase()}`;

        // Top-K Pills
        const topKPills = document.getElementById('dashTopKPills');
        if (topKPills && pred.top_k) {
            topKPills.innerHTML = pred.top_k.map(k => `
                <div class="top-k-pill">
                    <strong>${k.gesture}</strong>: ${(k.probability * 100).toFixed(0)}%
                </div>
            `).join('');
        }

        // Live Translation Tab Fullscreen Card
        const liveName = document.getElementById('livePredictedGesture');
        const liveHindi = document.getElementById('liveHindiGesture');
        const livePct = document.getElementById('liveConfidencePct');
        if (liveName) liveName.textContent = textEn;
        if (liveHindi) liveHindi.textContent = textHi;
        if (livePct) livePct.textContent = `${(confidence * 100).toFixed(1)}% Confidence`;

        // Handle TTS Voice Speech
        if (this.ttsEnabled && gesture !== 'UNCERTAIN' && gesture !== 'UNKNOWN' && gesture !== 'MODEL_NOT_TRAINED') {
            const now = Date.now();
            if (confidence >= 0.70 && (gesture !== this.lastSpokenGesture || now - this.lastSpokenTime > this.speechDebounceMs)) {
                this.lastSpokenGesture = gesture;
                this.lastSpokenTime = now;
                const spokenText = (this.ttsLanguage === 'hi' && textHi) ? textHi : textEn;
                this.speak(spokenText, this.ttsLanguage);
                this.addToHistory(gesture, textEn, textHi, confidence);
            }
        }
    }

    // ==========================================
    // Text-to-Speech (TTS) Engine
    // ==========================================
    initSpeechSynth() {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.onvoiceschanged = () => {
                this.voices = window.speechSynthesis.getVoices();
            };
        }
    }

    speak(text, lang = 'en') {
        if (!('speechSynthesis' in window) || !text) return;
        try {
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 0.95;
            utterance.pitch = 1.0;
            utterance.lang = lang === 'hi' ? 'hi-IN' : 'en-US';

            const voices = window.speechSynthesis.getVoices();
            const voice = voices.find(v => v.lang.startsWith(lang));
            if (voice) utterance.voice = voice;

            window.speechSynthesis.speak(utterance);
        } catch (e) {
            console.error('TTS error:', e);
        }
    }

    toggleTTS() {
        this.ttsEnabled = !this.ttsEnabled;
        const btn = document.getElementById('btnToggleTTS');
        if (btn) {
            btn.innerHTML = this.ttsEnabled ? '🔊 Audio TTS: ON' : '🔇 Audio TTS: OFF';
            btn.className = `btn btn-sm ${this.ttsEnabled ? 'btn-primary' : ''}`;
        }
    }

    addToHistory(gestureId, nameEn, nameHi, conf) {
        const timeStr = new Date().toLocaleTimeString();
        const record = { time: timeStr, gesture: gestureId, name: nameEn, hindi: nameHi, conf: conf };
        this.translationHistory.unshift(record);
        if (this.translationHistory.length > 50) this.translationHistory.pop();

        // Update History Table
        const tbody = document.getElementById('historyTableBody');
        if (!tbody) return;

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td style="font-family:var(--font-mono); color:var(--color-fg-muted);">${timeStr}</td>
            <td><strong style="font-family:var(--font-mono);">${nameEn}</strong></td>
            <td style="color:var(--color-success-fg); font-weight:600;">${nameHi}</td>
            <td style="font-family:var(--font-mono); color:var(--color-accent-fg);">${(conf * 100).toFixed(1)}%</td>
            <td>
                <button class="btn btn-sm" onclick="window.islApp.speak('${nameEn}', 'en')">🔊 EN</button>
                <button class="btn btn-sm" onclick="window.islApp.speak('${nameHi}', 'hi')">🔊 HI</button>
            </td>
        `;
        tbody.insertBefore(tr, tbody.firstChild);

        // Update Sentence Builder
        if (this.sentenceBuffer.length === 0 || this.sentenceBuffer[this.sentenceBuffer.length - 1] !== nameEn) {
            this.sentenceBuffer.push(nameEn);
            const sentEl = document.getElementById('currentSentenceText');
            if (sentEl) sentEl.textContent = this.sentenceBuffer.join(' ');
        }
    }

    speakSentence() {
        const sentence = this.sentenceBuffer.join(' ');
        if (sentence) {
            this.speak(sentence, this.ttsLanguage);
        }
    }

    clearSentence() {
        this.sentenceBuffer = [];
        const sentEl = document.getElementById('currentSentenceText');
        if (sentEl) sentEl.textContent = '...';
    }

    // ==========================================
    // TinyML C++ Exporter
    // ==========================================
    async exportTinyML() {
        try {
            const res = await fetch('/api/ml/export_tinyml', { method: 'POST' });
            const data = await res.json();
            alert(`✓ TinyML C++ code generated for ESP32!\nFiles created:\n  - ${data.files.header_path}\n  - ${data.files.source_path}\nClasses: ${data.files.num_classes}`);
            const resultDiv = document.getElementById('tinymlExportResult');
            if (resultDiv) {
                resultDiv.textContent = `✓ Generated ${data.files.header_path} & ${data.files.source_path} (${data.files.num_classes} classes)`;
            }
        } catch (e) {
            console.error('TinyML export error:', e);
            alert('Failed to export TinyML C++ model.');
        }
    }

    // ==========================================
    // Dataset Recording
    // ==========================================
    startRecording() {
        const user = document.getElementById('recordUserSelect')?.value || 'U01';
        const gesture = document.getElementById('recordGestureSelect')?.value || 'HELLO';
        const duration = parseInt(document.getElementById('recordDurationSelect')?.value || '5', 10);

        this.isRecording = true;
        this.recordingSamples = [];
        this.recordDurationSec = duration;

        document.getElementById('recordingStatusBadge')?.style.setProperty('display', 'inline-flex');
        document.getElementById('btnStartRecord')?.setAttribute('disabled', 'true');

        let remaining = duration;
        const countdownEl = document.getElementById('recordCountdown');
        if (countdownEl) countdownEl.textContent = `${remaining}s`;

        this.recordTimer = setInterval(() => {
            remaining -= 1;
            if (countdownEl) countdownEl.textContent = `${remaining}s`;

            if (remaining <= 0) {
                this.stopRecording();
            }
        }, 1000);
    }

    async stopRecording() {
        if (!this.isRecording) return;
        clearInterval(this.recordTimer);
        this.isRecording = false;

        document.getElementById('recordingStatusBadge')?.style.setProperty('display', 'none');
        document.getElementById('btnStartRecord')?.removeAttribute('disabled');

        const user = document.getElementById('recordUserSelect')?.value || 'U01';
        const gesture = document.getElementById('recordGestureSelect')?.value || 'HELLO';

        if (this.recordingSamples.length === 0) {
            alert('No samples were captured during the recording window.');
            return;
        }

        try {
            const res = await fetch('/api/dataset/record', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: user,
                    gesture: gesture,
                    samples: this.recordingSamples,
                    is_simulated: this.mode === 'simulated'
                })
            });
            const data = await res.json();
            alert(`Saved ${this.recordingSamples.length} samples for gesture '${gesture}' (User: ${user})`);
            await this.loadDatasetStats();
        } catch (e) {
            console.error('Error saving recording session:', e);
            alert('Failed to save recording session to CSV.');
        }
    }

    async generateSimulatedDataset() {
        const btn = document.getElementById('btnGenerateSimDataset');
        if (btn) btn.textContent = 'Generating 10,000 Samples...';

        try {
            const res = await fetch('/api/simulator/generate_dataset?samples_per_gesture=100', { method: 'POST' });
            const data = await res.json();
            alert(`Simulated dataset generated: ${data.statistics.total_samples} samples across 20 gestures!`);
            await this.loadDatasetStats();
        } catch (e) {
            console.error('Error generating simulated dataset:', e);
        } finally {
            if (btn) btn.textContent = '⚡ Generate Multi-User Baseline Dataset (10,000 Samples)';
        }
    }

    async loadDatasetStats() {
        try {
            const res = await fetch('/api/dataset/stats?dataset_type=all');
            const data = await res.json();

            document.getElementById('statTotalSamples').textContent = data.total_samples || 0;
            document.getElementById('statRealSamples').textContent = data.real_count || 0;
            document.getElementById('statSimSamples').textContent = data.simulated_count || 0;
            document.getElementById('statUserCount').textContent = (data.users || []).length;

            const navCount = document.getElementById('navDatasetCount');
            if (navCount) {
                navCount.textContent = (data.total_samples >= 1000) ? `${(data.total_samples / 1000).toFixed(0)}k` : data.total_samples;
            }

            // Render distribution table
            const tbody = document.getElementById('datasetDistributionBody');
            if (tbody && data.gesture_counts) {
                tbody.innerHTML = Object.keys(data.gesture_counts).map(g => `
                    <tr>
                        <td><strong style="font-family:var(--font-mono);">${g}</strong></td>
                        <td style="font-family:var(--font-mono);">${data.gesture_counts[g]}</td>
                        <td>
                            <div style="background:#21262d; height:6px; border-radius:3px; overflow:hidden;">
                                <div style="background-color:var(--color-success-fg); height:100%; width:${Math.min((data.gesture_counts[g] / 500) * 100, 100)}%;"></div>
                            </div>
                        </td>
                    </tr>
                `).join('');
            }
        } catch (e) {
            console.error('Error loading dataset stats:', e);
        }
    }

    // ==========================================
    // Calibration Wizard
    // ==========================================
    recordCalibrationStep(stepName) {
        const user = document.getElementById('calibUserSelect')?.value || 'U01';
        this.calibSamples[stepName] = [];

        const stepIdx = { relaxed: 1, straight: 2, bent: 3 }[stepName];
        this.calibrationStep = stepIdx;

        const btn = document.getElementById(`btnCalib${stepName.charAt(0).toUpperCase() + stepName.slice(1)}`);
        btn.innerHTML = `Recording 3s...`;
        btn.setAttribute('disabled', 'true');

        setTimeout(() => {
            this.calibrationStep = 0;
            btn.innerHTML = `✓ ${stepName.toUpperCase()} Recorded (${this.calibSamples[stepName].length} pts)`;
            btn.removeAttribute('disabled');
            btn.className = 'btn btn-sm btn-primary';
        }, 3000);
    }

    async saveCalibration() {
        const user = document.getElementById('calibUserSelect')?.value || 'U01';
        if (!this.calibSamples.relaxed.length || !this.calibSamples.straight.length || !this.calibSamples.bent.length) {
            alert('Please record all 3 calibration steps (Relaxed, Straight, and Bent) before saving.');
            return;
        }

        try {
            const res = await fetch('/api/calibration/compute', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: user,
                    relaxed_samples: this.calibSamples.relaxed,
                    straight_samples: this.calibSamples.straight,
                    bent_samples: this.calibSamples.bent
                })
            });
            const data = await res.json();
            alert(`Calibration profile saved for ${user}! Sensor normalization active.`);
        } catch (e) {
            console.error('Error saving calibration:', e);
        }
    }

    // ==========================================
    // ML Model Training & Evaluation
    // ==========================================
    async trainBaselineModels() {
        const btn = document.getElementById('btnTrainBaseline');
        const evalMode = document.getElementById('trainEvalModeSelect')?.value || 'user_independent';
        const trainUsers = ['U01', 'U02', 'U03', 'U04'];
        const testUsers = ['U05'];

        if (btn) {
            btn.innerHTML = '⏳ Training DT, RF & SVM...';
            btn.setAttribute('disabled', 'true');
        }

        try {
            const res = await fetch('/api/ml/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    evaluation_mode: evalMode,
                    train_users: trainUsers,
                    test_users: testUsers,
                    test_size: 0.2,
                    dataset_type: 'all'
                })
            });
            const data = await res.json();
            alert(`Training Complete! Best model: ${data.summary.best_model.toUpperCase()}`);
            await this.loadEvaluationReport();
        } catch (e) {
            console.error('Training error:', e);
            alert('Training failed. Check server logs.');
        } finally {
            if (btn) {
                btn.innerHTML = '🚀 Train & Compare Models (DT, RF, SVM)';
                btn.removeAttribute('disabled');
            }
        }
    }

    async trainLSTMModel() {
        const btn = document.getElementById('btnTrainLSTM');
        if (btn) {
            btn.innerHTML = '⏳ Training PyTorch LSTM...';
            btn.setAttribute('disabled', 'true');
        }

        try {
            const res = await fetch('/api/ml/train_lstm?epochs=15', { method: 'POST' });
            const data = await res.json();
            alert(`LSTM Dynamic Model trained successfully! Best Validation Accuracy: ${(data.report.best_val_accuracy * 100).toFixed(1)}%`);
        } catch (e) {
            console.error('LSTM training error:', e);
            alert('LSTM training failed.');
        } finally {
            if (btn) {
                btn.innerHTML = '⚡ Train Advanced Time-Series LSTM';
                btn.removeAttribute('disabled');
            }
        }
    }

    async loadEvaluationReport() {
        try {
            const res = await fetch('/api/ml/evaluation');
            if (!res.ok) return;
            const data = await res.json();

            if (data.models) {
                MetricsVisualizer.renderModelComparisonTable('modelComparisonTableBody', data.models);

                const bestModel = data.models[data.best_model] || data.models['svm'] || data.models['random_forest'];
                if (bestModel && bestModel.confusion_matrix_normalized) {
                    MetricsVisualizer.renderConfusionMatrix('cmHeatmapContainer', bestModel.confusion_matrix_normalized, bestModel.classes);
                    MetricsVisualizer.renderPerClassReport('perClassReportContainer', bestModel.per_class_report);
                }

                const navF1 = document.getElementById('navEvalF1');
                if (navF1 && bestModel) {
                    navF1.textContent = `${(bestModel.f1_weighted * 100).toFixed(0)}%`;
                }
            }
        } catch (e) {
            console.error('Error loading evaluation report:', e);
        }
    }

    // ==========================================
    // Hardware Connection Management (USB & COM)
    // ==========================================
    switchConnTab(tabName) {
        this.activeConnTab = tabName;
        document.querySelectorAll('.conn-tab-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.conn-panel').forEach(p => p.classList.remove('active'));

        if (tabName === 'usb') {
            document.getElementById('tabBtnUsb')?.classList.add('active');
            document.getElementById('connPanelUsb')?.classList.add('active');
        } else if (tabName === 'com') {
            document.getElementById('tabBtnCom')?.classList.add('active');
            document.getElementById('connPanelCom')?.classList.add('active');
        } else if (tabName === 'webserial') {
            document.getElementById('tabBtnWebSerial')?.classList.add('active');
            document.getElementById('connPanelWebSerial')?.classList.add('active');
            this.checkWebSerialSupport();
        }
    }

    checkWebSerialSupport() {
        const notice = document.getElementById('webSerialSupportNotice');
        if (!notice) return;
        if ('serial' in navigator) {
            notice.innerHTML = '<span class="Label Label--success" style="font-size:11px;">✓ Web Serial is supported in this browser</span>';
        } else {
            notice.innerHTML = '<span class="Label Label--attention" style="font-size:11px;">⚠️ Web Serial requires Google Chrome, Microsoft Edge, or Opera. Use USB Port or COM Port options above.</span>';
            const btn = document.getElementById('btnWebSerialConnect');
            if (btn) btn.disabled = true;
        }
    }

    async loadSerialPorts() {
        try {
            const res = await fetch('/api/serial/ports');
            const data = await res.json();

            // Populate USB Ports dropdown
            const usbSel = document.getElementById('usbPortSelect');
            if (usbSel) {
                usbSel.innerHTML = '';
                const usbPorts = data.usb_ports || [];
                if (usbPorts.length === 0) {
                    usbSel.appendChild(new Option('No USB UART ports found (Plug in ESP32 USB cable)', 'none'));
                } else {
                    usbPorts.forEach(p => {
                        const label = `${p.port} - ${p.chip || p.description}`;
                        const opt = new Option(label, p.port);
                        if (p.is_esp32_candidate) {
                            opt.style.fontWeight = 'bold';
                        }
                        usbSel.appendChild(opt);
                    });
                    if (data.recommended_usb_port) {
                        usbSel.value = data.recommended_usb_port;
                    }
                }
            }

            // Populate COM Ports dropdown (Bluetooth & others)
            const comSel = document.getElementById('comPortSelect');
            if (comSel) {
                comSel.innerHTML = '';
                const comPorts = (data.bluetooth_ports || []).concat(data.other_com_ports || []);
                if (comPorts.length === 0) {
                    comSel.appendChild(new Option('No Bluetooth or virtual COM ports found', 'none'));
                } else {
                    comPorts.forEach(p => {
                        const typeTag = p.is_bluetooth ? '[Bluetooth SPP]' : '[Serial COM]';
                        comSel.appendChild(new Option(`${p.port} ${typeTag} - ${p.description}`, p.port));
                    });
                }
            }

            // Legacy select for backward compatibility
            const legacySel = document.getElementById('serialPortSelect');
            if (legacySel) {
                legacySel.innerHTML = '';
                const all = data.ports || data.all_ports || [];
                all.forEach(p => {
                    legacySel.appendChild(new Option(`${p.port} - ${p.description}`, p.port));
                });
            }

            // Sync active status
            if (data.active_status) {
                this.updateHardwareUI(data.active_status);
            }
        } catch (e) {
            console.error('Error loading ports:', e);
        }
    }

    async autoConnectUsb() {
        const btn = document.getElementById('btnAutoConnectUsb');
        const origText = btn ? btn.textContent : '';
        if (btn) btn.textContent = '⏳ Scanning & Connecting...';
        try {
            const res = await fetch('/api/serial/auto_connect', { method: 'POST' });
            const data = await res.json();
            if (res.ok) {
                alert(`⚡ Successfully auto-connected to ESP32 on USB Port ${data.port} at 115200 baud!`);
                await this.setMode('hardware');
                await this.loadSerialPorts();
            } else {
                alert(data.detail || 'Failed to auto-detect ESP32 USB port. Please ensure USB cable is securely connected.');
            }
        } catch (e) {
            console.error('Auto connect USB error:', e);
            alert('Failed to connect to USB port.');
        } finally {
            if (btn) btn.textContent = origText || '⚡ 1-Click Auto-Detect & Connect';
        }
    }

    async connectUsbPort() {
        const port = document.getElementById('usbPortSelect')?.value || document.getElementById('serialPortSelect')?.value;
        const baud = document.getElementById('usbBaudSelect')?.value || 115200;
        if (!port || port === 'none') {
            alert('Please select a valid USB Serial port or click Auto-Detect.');
            return;
        }

        try {
            const res = await fetch(`/api/serial/connect?port=${encodeURIComponent(port)}&baud_rate=${baud}&connection_type=usb`, { method: 'POST' });
            const data = await res.json();
            if (res.ok) {
                alert(`Connected to ESP32 on USB Port ${port} (${baud} baud)!`);
                await this.setMode('hardware');
                await this.loadSerialPorts();
            } else {
                alert(data.detail || `Failed to connect to USB Port ${port}.`);
            }
        } catch (e) {
            console.error('USB Connect error:', e);
            alert(`Error connecting to USB port ${port}`);
        }
    }

    async connectComPort() {
        const customPort = document.getElementById('customComInput')?.value?.trim();
        const selectedPort = document.getElementById('comPortSelect')?.value;
        const port = customPort || selectedPort;
        const baud = document.getElementById('comBaudSelect')?.value || 115200;

        if (!port || port === 'none') {
            alert('Please select a COM port from the dropdown or type a custom COM port name (e.g. COM4).');
            return;
        }

        try {
            const res = await fetch(`/api/serial/connect?port=${encodeURIComponent(port)}&baud_rate=${baud}&connection_type=bluetooth`, { method: 'POST' });
            const data = await res.json();
            if (res.ok) {
                alert(`Connected to ESP32 on COM Port ${port} (${baud} baud)!`);
                await this.setMode('hardware');
                await this.loadSerialPorts();
            } else {
                alert(data.detail || `Failed to connect to COM Port ${port}.`);
            }
        } catch (e) {
            console.error('COM Connect error:', e);
            alert(`Error connecting to COM port ${port}`);
        }
    }

    // Legacy method
    async connectSerial() {
        return this.connectUsbPort();
    }

    async toggleWebSerial() {
        if (!('serial' in navigator)) {
            alert('Web Serial API is not supported in this browser. Please use Google Chrome or Microsoft Edge.');
            return;
        }

        if (this.isWebSerialRunning) {
            await this.disconnectWebSerial();
            return;
        }

        try {
            this.webSerialPort = await navigator.serial.requestPort();
            await this.webSerialPort.open({ baudRate: 115200 });
            this.isWebSerialRunning = true;

            const btn = document.getElementById('btnWebSerialConnect');
            if (btn) {
                btn.textContent = '🔌 Disconnect Web Serial';
                btn.className = 'btn btn-danger';
            }

            this.updateHardwareUI({
                is_connected: true,
                port: 'Browser WebUSB',
                connection_type: 'webserial',
                baud_rate: 115200,
                packets_received: 0,
                packets_dropped: 0,
                uptime_seconds: 0
            });

            await this.setMode('hardware');
            this.readWebSerialLoop();
        } catch (e) {
            console.error('Web Serial connection error:', e);
            alert(`Web Serial connection error: ${e.message || e}`);
        }
    }

    async readWebSerialLoop() {
        let buffer = '';
        let rxCount = 0;
        const startTime = Date.now();

        try {
            while (this.webSerialPort && this.webSerialPort.readable && this.isWebSerialRunning) {
                const textDecoder = new TextDecoderStream();
                this.webSerialPort.readable.pipeTo(textDecoder.writable);
                const reader = textDecoder.readable.getReader();
                this.webSerialReader = reader;

                try {
                    while (true) {
                        const { value, done } = await reader.read();
                        if (done) break;
                        if (value) {
                            buffer += value;
                            const lines = buffer.split('\n');
                            buffer = lines.pop();

                            for (const line of lines) {
                                const trimmed = line.trim();
                                if (!trimmed) continue;
                                const sample = this.parseISLPacket(trimmed);
                                if (sample) {
                                    rxCount++;
                                    this.handleStreamPayload({
                                        type: 'sensor_frame',
                                        timestamp: sample.timestamp,
                                        mode: 'hardware',
                                        connection_type: 'webserial',
                                        hardware_port: 'Browser WebUSB',
                                        raw_sample: sample,
                                        prediction: this.localPredict(sample),
                                        is_recording: this.isRecording
                                    });

                                    if (rxCount % 25 === 0) {
                                        const uptime = Math.round((Date.now() - startTime) / 1000);
                                        const pRx = document.getElementById('telemetryPacketsRx');
                                        const pUp = document.getElementById('telemetryUptime');
                                        if (pRx) pRx.textContent = rxCount.toLocaleString();
                                        if (pUp) pUp.textContent = `${uptime}s`;
                                    }
                                }
                            }
                        }
                    }
                } catch (readErr) {
                    console.warn('Web Serial stream ended:', readErr);
                } finally {
                    reader.releaseLock();
                }
            }
        } catch (err) {
            console.error('Web Serial loop error:', err);
        }
    }

    parseISLPacket(line) {
        if (!line.startsWith('$ISL') && !line.startsWith('ISL')) return null;
        let payload = line;
        if (payload.includes('*')) {
            payload = payload.split('*')[0];
        }
        if (payload.startsWith('$')) {
            payload = payload.substring(1);
        }
        const parts = payload.split(',');
        if (parts.length < 12) return null;

        try {
            return {
                thumb: parseFloat(parts[1]),
                index: parseFloat(parts[2]),
                middle: parseFloat(parts[3]),
                ring: parseFloat(parts[4]),
                little: parseFloat(parts[5]),
                acc_x: parseFloat(parts[6]),
                acc_y: parseFloat(parts[7]),
                acc_z: parseFloat(parts[8]),
                gyro_x: parseFloat(parts[9]),
                gyro_y: parseFloat(parts[10]),
                gyro_z: parseFloat(parts[11]),
                timestamp: Date.now() / 1000,
                is_simulated: false
            };
        } catch (e) {
            return null;
        }
    }

    localPredict(sample) {
        return {
            gesture: 'STREAMING',
            confidence: 0.95,
            calibrated_features: sample
        };
    }

    async disconnectWebSerial() {
        this.isWebSerialRunning = false;
        try {
            if (this.webSerialReader) {
                await this.webSerialReader.cancel();
            }
            if (this.webSerialPort) {
                await this.webSerialPort.close();
            }
        } catch (e) {
            console.warn('Error closing Web Serial:', e);
        }
        this.webSerialPort = null;
        this.webSerialReader = null;

        const btn = document.getElementById('btnWebSerialConnect');
        if (btn) {
            btn.textContent = '🔗 Connect via Web Serial (Browser USB)';
            btn.className = 'btn btn-primary';
        }
        await this.disconnectHardware();
    }

    async disconnectHardware() {
        if (this.isWebSerialRunning) {
            this.isWebSerialRunning = false;
            try {
                if (this.webSerialReader) await this.webSerialReader.cancel();
                if (this.webSerialPort) await this.webSerialPort.close();
            } catch (e) {}
            this.webSerialPort = null;
        }

        try {
            await fetch('/api/serial/disconnect', { method: 'POST' });
        } catch (e) {
            console.error('Disconnect error:', e);
        }

        this.updateHardwareUI({
            is_connected: false,
            port: null,
            connection_type: 'disconnected',
            baud_rate: 115200,
            packets_received: 0,
            packets_dropped: 0,
            uptime_seconds: 0
        });

        await this.setMode('simulated');
        await this.loadSerialPorts();
    }

    async setEspCommMode(mode) {
        try {
            const res = await fetch('/api/serial/set_comm_mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode })
            });
            const data = await res.json();
            if (res.ok) {
                alert(`ESP32 communication interface switched to: ${mode.toUpperCase()}`);
            } else {
                alert(data.detail || 'Failed to switch ESP32 mode.');
            }
        } catch (e) {
            console.error('Mode switch error:', e);
        }
    }

    updateHardwareUI(status) {
        this.hardwareStatus = {
            ...status,
            isConnected: Boolean(status && status.is_connected),
            is_connected: Boolean(status && status.is_connected)
        };
        const isConn = Boolean(status && status.is_connected);

        const dot = document.getElementById('hwConnStatusDot');
        const title = document.getElementById('hwConnStatusTitle');
        const badge = document.getElementById('activeConnBadge');
        const btnDisconnect = document.getElementById('btnDisconnectHardware');
        const modeGroup = document.getElementById('espModeCommandGroup');

        const typeEl = document.getElementById('telemetryType');
        const portEl = document.getElementById('telemetryPort');
        const baudEl = document.getElementById('telemetryBaud');
        const rxEl = document.getElementById('telemetryPacketsRx');
        const upEl = document.getElementById('telemetryUptime');

        if (isConn) {
            const connType = (status.connection_type || 'usb').toUpperCase();
            if (dot) dot.className = 'status-dot online';
            if (title) title.textContent = `Connected to ESP32 via ${connType} Port (${status.port || 'COM'})`;
            if (badge) {
                badge.className = 'Label Label--success';
                badge.textContent = `${connType}: ${status.port}`;
            }
            if (btnDisconnect) btnDisconnect.style.display = 'inline-block';
            if (modeGroup) modeGroup.style.display = 'block';

            if (typeEl) typeEl.textContent = connType;
            if (portEl) portEl.textContent = status.port || '--';
            if (baudEl) baudEl.textContent = status.baud_rate ? `${status.baud_rate}` : '115200';
            if (rxEl) rxEl.textContent = (status.packets_received || 0).toLocaleString();
            if (upEl) upEl.textContent = `${status.uptime_seconds || 0}s`;

            const hwBtn = document.getElementById('btnHwMode');
            if (hwBtn) {
                hwBtn.innerHTML = `<span>🔌</span> ${status.port} (${connType})`;
            }
        } else {
            if (dot) dot.className = 'status-dot sim';
            if (title) title.textContent = 'Hardware Offline (Using Biomechanical Simulator)';
            if (badge) {
                badge.className = 'Label Label--secondary';
                badge.textContent = 'Disconnected';
            }
            if (btnDisconnect) btnDisconnect.style.display = 'none';
            if (modeGroup) modeGroup.style.display = 'none';

            if (typeEl) typeEl.textContent = 'None';
            if (portEl) portEl.textContent = '--';
            if (baudEl) baudEl.textContent = '--';
            if (rxEl) rxEl.textContent = '0';
            if (upEl) upEl.textContent = '0s';

            const hwBtn = document.getElementById('btnHwMode');
            if (hwBtn) {
                hwBtn.innerHTML = `<span>🔌</span> ESP32 Hardware`;
            }
        }
    }
}

window.addEventListener('DOMContentLoaded', () => {
    window.islApp = new ISLGloveApp();
});
