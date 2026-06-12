from flask import Flask, request, jsonify, render_template
import pickle
import re
import io
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64

nltk.download('stopwords')

app = Flask(__name__)

with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

stop_words = set(stopwords.words('english'))
negation_words = {
    'not', 'no', 'nor', "don't", "doesn't", "didn't",
    "won't", "can't", "isn't", "aren't", "wasn't", "weren't",
    'never', 'neither', 'nobody', 'nothing', 'nowhere',
    'hardly', 'barely', 'scarcely'
}
stop_words = stop_words - negation_words
stemmer = PorterStemmer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [stemmer.stem(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)

label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    if not text.strip():
        return jsonify({'error': 'Please enter some text'}), 400

    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    confidence = round(max(probabilities) * 100, 2)

    return jsonify({
        'sentiment': label_map[prediction],
        'confidence': confidence
    })

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        df = pd.read_csv(file)
    except Exception:
        return jsonify({'error': 'Unable to read CSV file'}), 400

    if 'text' not in df.columns:
        return jsonify({'error': 'CSV must have a "text" column'}), 400

    results = []
    confidences = []
    for text in df['text'].astype(str):
        cleaned = clean_text(text)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]
        results.append(label_map[prediction])
        confidences.append(round(max(proba) * 100, 2))

    df['sentiment'] = results
    df['confidence'] = confidences

    # ── Summary counts ──
    summary = df['sentiment'].value_counts().to_dict()
    total = len(df)
    percentages = {k: round(v / total * 100, 2) for k, v in summary.items()}

    # ── Average confidence per sentiment ──
    avg_conf = df.groupby('sentiment')['confidence'].mean().round(2).to_dict()

    order = ['Positive', 'Neutral', 'Negative']
    counts_ordered = [summary.get(s, 0) for s in order]
    conf_ordered = [avg_conf.get(s, 0) for s in order]
    colors = ['#28a745', '#6c757d', '#dc3545']

    # ── Pie Chart ──
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    nonzero = [(o, c, col) for o, c, col in zip(order, counts_ordered, colors) if c > 0]
    ax1.pie(
        [c for _, c, _ in nonzero],
        labels=[o for o, _, _ in nonzero],
        colors=[col for _, _, col in nonzero],
        autopct='%1.1f%%',
        startangle=90
    )
    ax1.set_title('Sentiment Distribution')
    pie_buf = io.BytesIO()
    fig1.savefig(pie_buf, format='png', bbox_inches='tight')
    plt.close(fig1)
    pie_buf.seek(0)
    pie_b64 = base64.b64encode(pie_buf.read()).decode('utf-8')

    # ── Bar Chart: Count + Avg Confidence ──
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    x = range(len(order))
    bars = ax2.bar(x, counts_ordered, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(order)
    ax2.set_ylabel('Count')
    ax2.set_title('Sentiment Count & Avg Confidence')

    for i, (bar, conf) in enumerate(zip(bars, conf_ordered)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                 f'{int(height)}\n({conf}% conf)',
                 ha='center', va='bottom', fontsize=10)

    bar_buf = io.BytesIO()
    fig2.savefig(bar_buf, format='png', bbox_inches='tight')
    plt.close(fig2)
    bar_buf.seek(0)
    bar_b64 = base64.b64encode(bar_buf.read()).decode('utf-8')

    output = io.StringIO()
    df.to_csv(output, index=False)

    return jsonify({
        'summary': summary,
        'percentages': percentages,
        'avg_confidence': avg_conf,
        'total': total,
        'pie_chart': pie_b64,
        'bar_chart': bar_b64,
        'csv': output.getvalue()
    })


if __name__ == '__main__':
    app.run(debug=True)