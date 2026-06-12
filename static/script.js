async function analyzeSentiment() {
    const text = document.getElementById('reviewText').value;
    const resultBox = document.getElementById('result');
    const errorBox = document.getElementById('errorBox');
    const sentimentLabel = document.getElementById('sentimentLabel');
    const confidenceText = document.getElementById('confidenceText');
    const confidenceBar = document.getElementById('confidenceBar');

    resultBox.classList.add('hidden');
    errorBox.classList.add('hidden');
    resultBox.classList.remove('positive', 'negative', 'neutral');
    confidenceBar.style.width = '0%';

    if (!text.trim()) {
        errorBox.textContent = "Please enter some text first!";
        errorBox.classList.remove('hidden');
        return;
    }

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();

        if (data.error) {
            errorBox.textContent = data.error;
            errorBox.classList.remove('hidden');
            return;
        }

        sentimentLabel.textContent = data.sentiment;
        confidenceText.textContent = `Confidence: ${data.confidence}%`;
        confidenceBar.style.width = data.confidence + '%';

        resultBox.classList.add(data.sentiment.toLowerCase());
        resultBox.classList.remove('hidden');

    } catch (err) {
        errorBox.textContent = "Something went wrong. Please try again.";
        errorBox.classList.remove('hidden');
    }
}

async function analyzeBatch() {
    const fileInput = document.getElementById('csvFile');
    const batchResult = document.getElementById('batchResult');
    const batchSummary = document.getElementById('batchSummary');
    const downloadLink = document.getElementById('downloadLink');
    const errorBox = document.getElementById('errorBox');
    const pieChart = document.getElementById('pieChart');
    const barChart = document.getElementById('barChart');

    errorBox.classList.add('hidden');
    batchResult.classList.remove('show');
    batchResult.classList.add('hidden');
    downloadLink.classList.add('hidden');

    if (!fileInput.files.length) {
        errorBox.textContent = "Please choose a CSV file first!";
        errorBox.classList.remove('hidden');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch('/predict_batch', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            errorBox.textContent = data.error;
            errorBox.classList.remove('hidden');
            return;
        }

        let summaryText = `Total Reviews: ${data.total}  |  `;
        for (const [sentiment, count] of Object.entries(data.summary)) {
            const pct = data.percentages[sentiment];
            const conf = data.avg_confidence[sentiment];
            summaryText += `${sentiment}: ${count} (${pct}%, avg conf ${conf}%)   `;
        }
        batchSummary.textContent = summaryText;

        pieChart.src = 'data:image/png;base64,' + data.pie_chart;
        barChart.src = 'data:image/png;base64,' + data.bar_chart;

        const blob = new Blob([data.csv], { type: 'text/csv' });
        downloadLink.href = URL.createObjectURL(blob);
        downloadLink.classList.remove('hidden');

        batchResult.classList.remove('hidden');
        setTimeout(() => {
            batchResult.classList.add('show');
        }, 50);

    } catch (err) {
        errorBox.textContent = "Something went wrong with batch analysis.";
        errorBox.classList.remove('hidden');
    }
}