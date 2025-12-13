from typing import Dict
from openai import OpenAI
import os
from dotenv import load_dotenv


def _build_prompt(statement: str) -> str:
    return f"""You are a professional fact-checker. Determine if this statement is TRUE, FALSE, MISLEADING, or UNSURE.

Statement: {statement}

Evidence (latest first):
None provided

Google Fact Check JSON (if any): None

Rules:
- If no reliable sources or insufficient evidence → Default to FALSE (assume misinformation unless proven otherwise)
- Cite sources as [1][2] etc. only if you are certain
- If sources conflict → MISLEADING
- Only mark TRUE if multiple credible sources clearly confirm the statement
- Output format EXACTLY:

Verdict: TRUE / FALSE / MISLEADING / UNSURE
Confidence: High / Medium / Low
Explanation: (2-4 sentences max)
Sources:
[1] Title - URL
import requests
[2] ..."""


def _parse_response(text: str) -> Dict:
    lines = [l.strip() for l in text.strip().split("\n")]
    verdict = "UNSURE"
    confidence = "Low"
    explanation = ""
    sources: list[str] = []
    parsing_sources = False
    for line in lines:
        if line.startswith("Verdict:"):
            verdict = line.replace("Verdict:", "").strip()
        elif line.startswith("Confidence:"):
            confidence = line.replace("Confidence:", "").strip()
        elif line.startswith("Explanation:"):
            explanation = line.replace("Explanation:", "").strip()
        elif line.startswith("Sources:"):
            parsing_sources = True
        elif parsing_sources and line.startswith("["):
            sources.append(line)
    return {
        "verdict": verdict,
        "confidence": confidence,
        "explanation": explanation,
        "sources": sources,
    }


def check_with_openai(statement: str) -> Dict:
    # Load .env each call to pick up changes without restart
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return {
            "verdict": "UNSURE",
            "confidence": "Low",
            "explanation": "Missing OPENAI_API_KEY in environment.",
            "sources": [],
        }

    client = OpenAI(api_key=api_key)

    prompt = _build_prompt(statement)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer strictly in the requested format."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    text = resp.choices[0].message.content if resp.choices else ""
    if not text:
        return {
            "verdict": "UNSURE",
            "confidence": "Low",
            "explanation": "No response from model.",
            "sources": [],
        }

    return _parse_response(text)

def search_with_serper(query: str) -> list:
    """
    Perform a web search using Serper API and return a list of results.
    """
    load_dotenv()
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise ValueError("SERPER_API_KEY not set in environment variables.")
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query}
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    # Return top 5 organic results (title and link)
    results = []
    for item in data.get("organic", [])[:5]:
        title = item.get("title", "")
        link = item.get("link", "")
        results.append(f"{title} - {link}")
    return results
