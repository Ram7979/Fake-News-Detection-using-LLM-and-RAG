# Fake News Detector

Streamlit app that uses OpenAI to analyze statements and return a verdict: TRUE, FALSE, MISLEADING, or UNSURE. Results are cached for 24 hours per unique statement.

## Setup

1) Install dependencies
```powershell
pip install -r requirements.txt
```

2) Create `.env` and set your OpenAI key
```powershell
copy .env.example .env
```
Edit `.env` and set:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Run

Start the Streamlit app:
```powershell
& ".venv/Scripts/streamlit.exe" run app.py
```

Open the URL printed in the terminal (e.g., http://localhost:8501).

## Notes

- No backend server required; Streamlit calls OpenAI directly.
- The model will default to FALSE when evidence is insufficient or unclear, and only mark TRUE when multiple credible sources clearly support the claim.
