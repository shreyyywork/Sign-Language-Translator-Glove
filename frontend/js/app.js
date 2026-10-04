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
        document.getElementById('btnHwMode')?.addEventListener('click', () => this.setMode('hardware'));

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

        // Serial Connect/Disconnect
        document.getElementById('btnConnectSerial')?.addEventListener('click', () => this.connectSerial());
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
        const raw = payload.raw_sample || {};
        const pred = payload.prediction || {};
        const calib = pred.calibrated_features || raw;

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
            if (fillEl) fillEl.style.height = `${pct}%`;
            if (valEl) valEl.textContent = `${pct.toFixed(0)}%`;
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
    // Serial Hardware Management
    // ==========================================
    async loadSerialPorts() {
        try {
            const res = await fetch('/api/serial/ports');
            const data = await res.json();
            const sel = document.getElementById('serialPortSelect');
            if (!sel) return;

            sel.innerHTML = '';
            if (data.ports.length === 0) {
                sel.appendChild(new Option('No hardware COM ports found (Use Simulator)', 'none'));
            } else {
                data.ports.forEach(p => {
                    sel.appendChild(new Option(`${p.port} - ${p.description}`, p.port));
                });
            }
        } catch (e) {
            console.error('Error loading ports:', e);
        }
    }

    async connectSerial() {
        const port = document.getElementById('serialPortSelect')?.value;
        if (!port || port === 'none') {
            alert('Please select a valid hardware COM port or Bluetooth virtual port.');
            return;
        }

        try {
            const res = await fetch(`/api/serial/connect?port=${encodeURIComponent(port)}`, { method: 'POST' });
            if (res.ok) {
                alert(`Successfully connected to ESP32 on ${port}!`);
                this.setMode('hardware');
            } else {
                alert(`Failed to connect to ${port}.`);
            }
        } catch (e) {
            console.error('Serial connect error:', e);
        }
    }
}

window.addEventListener('DOMContentLoaded', () => {
    window.islApp = new ISLGloveApp();
});
