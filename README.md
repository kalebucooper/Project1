# Project 1 — Streamlit Chatbot + URL Credibility Scoring

## 🧬 Setup

1. Create a virtual environment and install dependencies.
2. Copy `.env.example` → `.env` and add your Perplexity API key.
3. Run the app.

## 🧠 Description

* Paste any URL → shows credibility score, band, stars, and explanation.
* Ask a question → AI chatbot responds (Perplexity API).
* Toggle web search & credibility scoring in the sidebar.
* Export results to CSV.

## 🚀 How to Run This App

### 1. Clone the repository

```bash
git clone https://github.com/kalebucooper/Project1.git
cd Project1
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your API key

Copy the example:

```bash
cp .env.example .env
```

Then open `.env` and paste your own Perplexity API key:

```
PPLX_API_KEY=YOUR_API_KEY_HERE
```

### 5. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at:
🔗 [http://localhost:8501](http://localhost:8501)

