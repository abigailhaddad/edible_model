async function checkEdibility() {
    const icons = {
        edible: `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M20 6L9 17l-5-5"/>
        </svg>`,
        'not-edible': `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M18 6L6 18M6 6l12 12"/>
        </svg>`,
        uncertain: `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <path d="M12 16v-4M12 8h.01"/>
        </svg>`,
        error: `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
        </svg>`
    };

    const button = document.querySelector('button');
    const input = document.getElementById('textInput');
    const resultDiv = document.getElementById('result');

    if (!input.value.trim()) return;

    button.disabled = true;
    button.textContent = 'Checking...';

    try {
        const response = await fetch('https://abigailhaddad1.pythonanywhere.com/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text: input.value})
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Error making prediction');
        }

        const data = await response.json();
        const confidence = Math.round(data.confidence * 100);
        const prediction = data.prediction.replace('_', ' ');

        let statusClass = confidence > 80 ?
            (prediction === 'edible' ? 'edible' : 'not-edible') :
            'uncertain';

        resultDiv.innerHTML = `
            <div class="result-card ${statusClass}">
                <div style="color: ${statusClass === 'edible' ? '#16a34a' :
                                   statusClass === 'not-edible' ? '#dc2626' :
                                   '#eab308'}">
                    ${icons[statusClass]}
                </div>
                <div class="verdict">
                    Model predicts: ${prediction} (${confidence}%)
                </div>
            </div>
        `;
    } catch (error) {
        resultDiv.innerHTML = `
            <div class="result-card not-edible">
                <div style="color: #dc2626">${icons.error}</div>
                <div class="verdict">${error.message}</div>
            </div>
        `;
    } finally {
        button.disabled = false;
        button.textContent = 'Check';
    }
}

document.getElementById('textInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        checkEdibility();
    }
});