# 🧠 Intelligent Customer Feedback Analytics Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green)
![Scikit--learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)
![SQLite](https://img.shields.io/badge/SQLite-3-lightblue?logo=sqlite)
![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75?logo=plotly)
![License](https://img.shields.io/badge/License-MIT-yellow)

**An end-to-end NLP-powered platform that automatically analyses customer reviews, extracts sentiment, surfaces trending topics, and delivers interactive visual dashboards — all in your browser.**

[Live Demo](#) · [Report Bug](#) · [Request Feature](#)

</div>

---

## 📸 Screenshots

| Dashboard Overview | Keyword Analysis |
|---|---|
| ![Dashboard](https://via.placeholder.com/600x350/2c3e50/ffffff?text=Sentiment+Dashboard) | ![Keywords](https://via.placeholder.com/600x350/16213e/ffffff?text=Keyword+Analysis) |

| Trend Analysis | Export Report |
|---|---|
| ![Trends](https://via.placeholder.com/600x350/0f3460/ffffff?text=Monthly+Trends) | ![Export](https://via.placeholder.com/600x350/533483/ffffff?text=Excel+Export) |

---

## 🎯 Problem Statement

Customer feedback is generated at massive scale — yet most organisations lack automated tools to analyse it efficiently. This platform solves that by providing:

- **Automatic sentiment classification** of free-text reviews
- **Keyword and topic extraction** to identify recurring pain points and praise
- **Time-series trend analysis** to track sentiment shifts over months
- **Persistent storage** so analyses accumulate for longitudinal comparison
- **One-click exportable reports** for business stakeholder presentations

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📂 **CSV Upload** | Upload any customer feedback CSV with a text column |
| 🧹 **Text Preprocessing** | Lowercasing, URL/HTML stripping, punctuation removal, stopword filtering, lemmatisation |
| 💬 **VADER Sentiment Analysis** | Rule-based NLP — no training data required; instant Positive/Negative/Neutral labels |
| 🤖 **ML Classifier** | TF-IDF + Logistic Regression pipeline trained on VADER pseudo-labels (scikit-learn) |
| 🔑 **Keyword Extraction** | TF-IDF top keywords and bigrams per sentiment label and product category |
| ☁️ **Word Cloud** | Visual word cloud filterable by sentiment |
| 📈 **Monthly Trends** | Line charts of sentiment counts and VADER compound score over time |
| 🗺️ **Heatmap** | Category × Month positive sentiment percentage heatmap |
| 🗄️ **SQLite Persistence** | Every processed batch is stored; historical data accumulates |
| 📊 **KPI Dashboard** | Live metrics: total reviews, positive/negative/neutral % |
| 📥 **Excel Export** | Multi-sheet report (Summary, Trends, Keywords, Metadata) |
| 🔍 **Database Explorer** | Browse all stored records and session history |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| NLP / ML | NLTK (VADER, tokeniser, lemmatiser), scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualisation | Plotly (interactive), Matplotlib + WordCloud |
| Dashboard UI | Streamlit |
| Database | SQLite (built-in `sqlite3` module) |
| Export | openpyxl (multi-sheet Excel) |

---

## 📁 Project Structure

```
intelligent-feedback-analytics/
│
├── app.py                       ← Main Streamlit application (entry point)
├── requirements.txt             ← Python dependencies
├── README.md                    ← This file
├── setup.py                     ← NLTK data download helper
│
├── config/
│   └── settings.py              ← Global constants and configuration
│
├── dataset/
│   └── sample_feedback.csv      ← 250-row realistic sample dataset
│
├── src/
│   ├── __init__.py
│   ├── preprocessor.py          ← Text cleaning pipeline
│   ├── sentiment_analyzer.py    ← VADER + Logistic Regression sentiment
│   ├── keyword_extractor.py     ← TF-IDF keyword/key-phrase extraction
│   ├── trend_analyzer.py        ← Time-series and category trend analysis
│   ├── visualizer.py            ← All Plotly & Matplotlib chart functions
│   └── report_generator.py      ← CSV and Excel export helpers
│
├── database/
│   ├── __init__.py
│   └── db_manager.py            ← SQLite CRUD operations
│
├── models/                      ← Persisted ML model (auto-generated)
├── visualizations/              ← Saved chart images (auto-generated)
└── exports/                     ← Downloaded report files (auto-generated)
```

---

## ⚡ Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Git

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/intelligent-feedback-analytics.git
cd intelligent-feedback-analytics
```

### Step 2 — Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Download NLTK data

```bash
python setup.py
```

### Step 5 — Launch the dashboard

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501** 🎉

---

## 📊 Usage Guide

### Using the Sample Dataset

1. Open the sidebar and click **"Browse files"**
2. Navigate to `dataset/sample_feedback.csv`
3. The text column name is `review_text` (already set by default)
4. Click **"🚀 Analyse Now"**

### Using Your Own Dataset

Your CSV should contain at least one text column. Recommended additional columns:

| Column | Type | Example |
|---|---|---|
| `review_text` | string | "Great product, fast delivery!" |
| `rating` | int (1–5) | 4 |
| `date` | YYYY-MM-DD | 2024-06-15 |
| `product_category` | string | Electronics |
| `product_name` | string | Wireless Headphones |

### Navigating the Dashboard

| Tab | What you'll find |
|---|---|
| 🏠 Home & Upload | Dataset preview and quick-start guide |
| 📊 Analysis Dashboard | KPI cards, sentiment pie/bar charts, rating distribution |
| 🔑 Keywords & Topics | TF-IDF keywords, word cloud, per-category keywords |
| 📈 Trends & Insights | Monthly trend lines, heatmap, best/worst products |
| 🗄️ Database Explorer | Browse stored records, session history |
| 📥 Export Report | Download CSV or Excel report |

---

## 🧪 How It Works — Technical Deep Dive

### NLP Pipeline

```
Raw CSV  →  Text Cleaning  →  VADER Scoring  →  Label Assignment  →  ML Classifier
              ↓                    ↓                  ↓                    ↓
         lowercase           compound score     Positive/Negative/   TF-IDF + LR
         remove URLs         range −1 to +1     Neutral              (trained on
         strip HTML          pos/neg/neu                              VADER labels)
         remove punct        components
         remove stopwords
         lemmatise
```

### Sentiment Labelling Rules (VADER)

```python
if compound_score >= +0.05:  → Positive
if compound_score <= -0.05:  → Negative
else:                        → Neutral
```

### Keyword Extraction

TF-IDF (Term Frequency–Inverse Document Frequency) scores each word by how important it is to a specific sub-corpus (e.g. Negative reviews) relative to the entire corpus. This surfaces genuinely discriminative keywords rather than just frequent ones.

---

## 📝 Resume Presentation

### Project Title
> **Intelligent Customer Feedback Analytics Platform** | Python · NLP · Streamlit · SQLite

```

### Skills Demonstrated
`Python` · `Pandas` · `NLTK` · `scikit-learn` · `Plotly` · `Streamlit` · `SQLite` · `NLP` · `Sentiment Analysis` · `Data Visualisation` · `Object-Oriented Design` · `REST-free full-stack data app`

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Darshan** — B.Tech AIML, SRM Institute of Science and Technology  
📧 prithivdarshan10@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/darshan-7039bb352)  
💻 [GitHub](https://github.com/prithivdarshan10-crypto)

---

<div align="center">
  ⭐ If you found this project useful, please give it a star on GitHub!
</div>
