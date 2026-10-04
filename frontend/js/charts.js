/**
 * Charts and Metrics Visualizer for ISL Glove
 * Styled with GitHub Contribution Graph Green Palette and Primer Component Tokens.
 */

class MetricsVisualizer {
    static renderConfusionMatrix(containerId, cmData, classNames) {
        const container = document.getElementById(containerId);
        if (!container || !cmData || !classNames) return;

        let html = '<table class="cm-table">';
        html += '<thead><tr><th style="color:#8b949e; padding:4px;">True \\ Pred</th>';
        
        // Header columns
        classNames.forEach(c => {
            html += `<th title="${c}" style="color:#8b949e; padding:4px;">${c.length > 5 ? c.substring(0, 4) + '.' : c}</th>`;
        });
        html += '</tr></thead><tbody>';

        // Rows
        for (let i = 0; i < cmData.length; i++) {
            const trueLabel = classNames[i] || `Class ${i}`;
            html += `<tr><th title="${trueLabel}" style="color:#8b949e; text-align:right; padding-right:6px;">${trueLabel.length > 6 ? trueLabel.substring(0, 5) + '.' : trueLabel}</th>`;

            for (let j = 0; j < cmData[i].length; j++) {
                const val = cmData[i][j];
                const isDiag = (i === j);
                
                let bgColor;
                let textColor = '#e6edf3';

                // GitHub Contribution Graph Green Levels
                if (isDiag) {
                    if (val >= 0.95) bgColor = '#39d353'; // Level 4 green
                    else if (val >= 0.70) bgColor = '#26a641'; // Level 3 green
                    else if (val >= 0.40) bgColor = '#006d32'; // Level 2 green
                    else if (val > 0.0) bgColor = '#0e4429';   // Level 1 green
                    else bgColor = '#161b22';

                    if (val >= 0.70) textColor = '#0d1117'; // dark text on bright green
                } else if (val > 0.05) {
                    bgColor = '#da3633'; // GitHub Red error
                    textColor = '#ffffff';
                } else {
                    bgColor = '#161b22';
                    textColor = '#484f58';
                }

                const displayVal = (typeof val === 'number' && !Number.isInteger(val)) ? (val * 100).toFixed(0) + '%' : val;
                html += `<td class="cm-cell" style="background:${bgColor}; color:${textColor};" 
                             title="True: ${trueLabel}, Predicted: ${classNames[j]}, Value: ${val}">
                            ${displayVal}
                         </td>`;
            }
            html += '</tr>';
        }

        html += '</tbody></table>';
        container.innerHTML = html;
    }

    static renderModelComparisonTable(tableBodyId, modelsData) {
        const tbody = document.getElementById(tableBodyId);
        if (!tbody || !modelsData) return;

        tbody.innerHTML = '';
        const modelKeys = Object.keys(modelsData);

        // Find highest F1
        let bestKey = '';
        let maxF1 = -1;
        modelKeys.forEach(k => {
            if (modelsData[k].f1_weighted > maxF1) {
                maxF1 = modelsData[k].f1_weighted;
                bestKey = k;
            }
        });

        modelKeys.forEach(key => {
            const m = modelsData[key];
            const isBest = (key === bestKey);
            const tr = document.createElement('tr');
            if (isBest) tr.style.backgroundColor = 'rgba(56, 139, 253, 0.08)';

            tr.innerHTML = `
                <td>
                    <strong style="font-family:var(--font-mono);">${m.model_name.toUpperCase().replace('_', ' ')}</strong>
                    ${isBest ? '<span class="Label Label--success" style="margin-left:8px;">★ BEST MODEL</span>' : ''}
                </td>
                <td><strong style="color:var(--color-accent-fg); font-family:var(--font-mono);">${(m.accuracy * 100).toFixed(1)}%</strong></td>
                <td style="font-family:var(--font-mono);">${(m.precision_weighted * 100).toFixed(1)}%</td>
                <td style="font-family:var(--font-mono);">${(m.recall_weighted * 100).toFixed(1)}%</td>
                <td><strong style="color:var(--color-success-fg); font-family:var(--font-mono);">${(m.f1_weighted * 100).toFixed(1)}%</strong></td>
                <td style="font-family:var(--font-mono); color:var(--color-fg-muted);">${m.training_time_sec ? m.training_time_sec.toFixed(3) + 's' : '-'}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    static renderPerClassReport(containerId, reportDict) {
        const container = document.getElementById(containerId);
        if (!container || !reportDict) return;

        let html = '<table class="cyber-table"><thead><tr><th>Gesture Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr></thead><tbody>';

        const keys = Object.keys(reportDict).filter(k => !['accuracy', 'macro avg', 'weighted avg'].includes(k));
        keys.forEach(k => {
            const row = reportDict[k];
            const f1 = (row['f1-score'] * 100).toFixed(1);
            const prec = (row['precision'] * 100).toFixed(1);
            const rec = (row['recall'] * 100).toFixed(1);
            const f1Color = row['f1-score'] >= 0.9 ? 'var(--color-success-fg)' : (row['f1-score'] >= 0.75 ? 'var(--color-accent-fg)' : 'var(--color-attention-fg)');

            html += `<tr>
                <td><strong style="font-family:var(--font-mono);">${k}</strong></td>
                <td style="font-family:var(--font-mono);">${prec}%</td>
                <td style="font-family:var(--font-mono);">${rec}%</td>
                <td style="font-family:var(--font-mono); color:${f1Color}; font-weight:700;">${f1}%</td>
                <td style="font-family:var(--font-mono); color:var(--color-fg-muted);">${row['support']}</td>
            </tr>`;
        });

        html += '</tbody></table>';
        container.innerHTML = html;
    }
}

window.MetricsVisualizer = MetricsVisualizer;
