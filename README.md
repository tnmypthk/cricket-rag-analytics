# 🏏 IPL Cricket Analytics RAG System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cricket-rag-analytics-tn2wuq3qlzz8btkhscid8e.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-orange.svg)](https://chromadb.com)
[![Gemini](https://img.shields.io/badge/LLM-Gemini_2.5-green.svg)](https://aistudio.google.com)

> An AI-powered IPL cricket intelligence system combining **Retrieval-Augmented Generation (RAG)** with real-time analytics — ask any question about 1,095 IPL matches in plain English.

🔗 **Live Demo:** [cricket-rag-analytics-tn2wuq3qlzz8btkhscid8e.streamlit.app](https://cricket-rag-analytics-tn2wuq3qlzz8btkhscid8e.streamlit.app)

---

## 📸 Screenshots

| Ask Anything | Team Stats | Head to Head | Records |
|---|---|---|---|
| AI-powered Q&A | Per-season charts | Win/loss breakdown | All-time leaderboards |

---

## 🎯 Features

- **Natural Language Q&A** — Ask any IPL question in plain English, powered by ChromaDB semantic search + Google Gemini 2.5
- **Hybrid Analytics Engine** — Intelligently routes simple aggregate queries (POTM records, win counts) to Pandas and complex contextual questions to the RAG pipeline
- **Team Statistics Dashboard** — Per-team win rates, seasonal performance charts, and top Player of the Match winners
- **Head-to-Head Comparison** — Full historical record between any two IPL franchises with recent match results
- **All-Time Records** — Biggest wins by runs and wickets, most successful teams, top performers

---

## 🏗️ Architecture

```
User Query
    ↓
Entity Recognition (query type classification)
    ↓
┌─────────────────────────────────┐
│  Hybrid Router                  │
│  ├── Aggregate query → Pandas   │
│  └── Contextual query → RAG     │
└─────────────────────────────────┘
    ↓                    ↓
Pandas Analytics    ChromaDB Vector Search
(POTM, win counts)  (semantic similarity)
                         ↓
                    Gemini 2.5 Flash Lite
                    (answer generation)
    ↓                    ↓
        Final Answer to User
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Google Gemini 2.5 Flash Lite |
| Vector Database | ChromaDB (persistent) |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly Express |
| Deployment | Streamlit Cloud |

---

## 📊 Dataset

- **1,095 IPL matches** spanning 17 seasons (2008–2024)
- **15 franchises** including historical teams (Deccan Chargers, Kochi Tuskers, Pune Warriors)
- **Fields:** teams, venue, toss, result, margin, Player of the Match, season
- Source: [Kaggle IPL Complete Dataset](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/tnmypthk/cricket-rag-analytics.git
cd cricket-rag-analytics
```

### 2. Create conda environment
```bash
conda create -n cricket-rag python=3.11 -y
conda activate cricket-rag
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your Google Gemini API key
```

### 5. Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
cricket-rag-analytics/
├── app.py                  # Main Streamlit application
├── requirements.txt
├── .env.example
├── README.md
├── data/
│   ├── raw/                # Original Kaggle CSV files
│   ├── processed/          # Cleaned match data
│   └── embeddings/         # ChromaDB persistent vector store
├── src/
│   ├── ingestion/          # Data loading and preprocessing
│   ├── rag/                # Vector store and RAG pipeline
│   └── analytics/          # Cricket entity recognition
└── notebooks/
    └── exploration.ipynb   # Data exploration and RAG testing
```

---

## 💡 Example Questions

```
"How did Mumbai Indians perform at Wankhede Stadium?"
"Who won the most Player of the Match awards?"
"What was the biggest win margin in IPL history?"
"Mumbai Indians vs Chennai Super Kings head to head"
"Most successful team in IPL history?"
"Which venue hosted the most IPL matches?"
```

---

## 🧠 How RAG Works

Traditional search finds exact keyword matches. RAG uses **semantic similarity** — it understands meaning.

```python
# Your question gets converted to a vector (384 numbers)
query_embedding = model.encode("Mumbai Indians at Wankhede")

# ChromaDB finds the most semantically similar match records
results = collection.query(query_embeddings=[query_embedding], n_results=5)

# Gemini reads those matches and generates a human answer
response = gemini.generate_content(f"Context: {results}\nQuestion: {query}")
```

---

## 📄 Key Learnings

- RAG is powerful for **contextual retrieval** but struggles with aggregate queries — solved with a hybrid Pandas layer
- `pip freeze` creates bloated `requirements.txt` files that break cloud deployments — always use minimal requirements
- ChromaDB persistent storage must be committed to version control for cloud deployments
- Conda environments keep project dependencies isolated and reproducible

---

## 🗺️ Roadmap

- [ ] Add ball-by-ball data for deeper match analysis
- [ ] Player career statistics (batting/bowling averages)
- [ ] Match prediction model using historical head-to-head data
- [ ] Live match score integration via CricAPI
- [ ] Support for Test and ODI formats

---

## 👤 Author

**Tanmay Pathak**
- GitHub: [@tnmypthk](https://github.com/tnmypthk)

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

*Built with 🏏 using Python, ChromaDB, Google Gemini, and Streamlit*
