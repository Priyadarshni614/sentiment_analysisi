# AI Based Sentiment Analysis System

An AI/ML project that classifies reviews and comments as **Positive**, **Negative**, or **Neutral** using classical Machine Learning. Built as part of the AI & ML Internship at Wyntrix Innovation OPC Private Limited.

---

## 📌 Project Overview

This system analyzes text (reviews, comments, feedback) from multiple domains — e-commerce, movies, food, apps, hotels, and social media — and predicts the sentiment using a trained SVM model with TF-IDF features.

The project includes:
- A trained ML pipeline (TF-IDF + SVM with negation handling)
- A Flask backend serving predictions via REST API
- A web frontend for single-text and batch (CSV) sentiment analysis
- Visual analytics (pie chart, bar chart) for batch results

---

## 🎯 Features

- Single text sentiment prediction with confidence score
- Batch sentiment analysis via CSV upload
- Sentiment distribution pie chart
- Count & average confidence bar chart per sentiment class
- Downloadable CSV with sentiment + confidence columns
- Handles negation (e.g., "not bad", "don't hate") via custom preprocessing

---

## 🧠 Model Details

| Component | Description |
|---|---|
| Vectorization | TF-IDF (unigrams + bigrams, 50,000 features) |
| Model | Linear SVM (calibrated for probability output) |
| Preprocessing | Lowercasing, URL/mention removal, stemming, stopword removal (negation words preserved) |
| Classes | Positive, Neutral, Negative |

### Model Performance

| Model | Training Accuracy | Testing Accuracy |
|---|---|---|
| Naive Bayes | XX.XX% | XX.XX% |
| Logistic Regression | XX.XX% | XX.XX% |
| SVM (Final, with negation fix) | 90.28% | 86.55% |

*(Update Naive Bayes and Logistic Regression scores from your retrained results)*

---

## 📦 Datasets Used

| Dataset | Domain | Source |
|---|---|---|
| Twitter Sentiment140 | Social media | Kaggle |
| IMDb Movie Reviews | Entertainment | Kaggle |
| Amazon Reviews | E-commerce | Kaggle |
| Yelp Reviews | Food / Service | Yelp Open Dataset |
| Google Play Store Reviews | Mobile apps | Kaggle |

All datasets were merged, balanced (200,000 samples per class), cleaned, and used to train the final model.

---

## 🗂️ Project Structure

```
Sentiment_Analysis_Project/
├── app.py                    # Flask backend
├── sentiment_model.pkl        # Trained SVM model (calibrated)
├── tfidf_vectorizer.pkl       # TF-IDF vectorizer
├── cleaned_sentiment_dataset.csv  # Preprocessed training data
├── templates/
│   └── index.html             # Frontend page
├── static/
│   ├── style.css
│   └── script.js
└── notebooks/
    └── sentiment_analysis.ipynb   # Training notebook
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd Sentiment_Analysis_Project
```

### 2. Install dependencies
```bash
pip install flask pandas scikit-learn nltk matplotlib
```

### 3. Download NLTK data (first run only)
```python
import nltk
nltk.download('stopwords')
```

### 4. Run the application
```bash
python app.py
```

### 5. Open in browser
```
http://127.0.0.1:5000
```

---

## 🖥️ Usage

### Single Text Analysis
1. Enter a review or comment in the text box
2. Click **Analyze Sentiment**
3. View the predicted sentiment with a confidence bar

### Batch CSV Analysis
1. Prepare a CSV file with a column named `text`
2. Click **Choose File** and select the CSV
3. Click **Analyze CSV**
4. View summary statistics, pie chart, bar chart, and download results

**Sample CSV format:**
```csv
text
"This product is absolutely amazing!"
"Worst experience ever, never buying again."
"It was okay, nothing special."
```

---



## ⚠️ Known Limitations

- Struggles with sarcasm (e.g., "Oh great, another thing that doesn't work")
- Mixed/contrasting statements (e.g., "I don't hate it but I don't love it") can be misclassified
- Performance varies slightly across domains due to dataset representation

---

## 🚀 Future Improvements

- Fine-tune transformer-based models (BERT/RoBERTa) for better context understanding
- Add aspect-based sentiment analysis (e.g., sentiment per product feature)
- Expand training data for app/tech-domain reviews
- Multi-language support

---

## 👨‍💻 Author

AI & ML Internship Project — Wyntrix Innovation OPC Private Limited

---

## 📄 License

This project is for educational/internship purposes.
