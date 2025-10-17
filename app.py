#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project 1 — Streamlit Chatbot + URL Credibility Scoring
- Sidebar toggles for Web Search + Credibility display
- Perplexity model picker
- CSV export of scored URLs
- Clean UI cards
Run: streamlit run app.py
"""

import os
import re
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

import requests
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# ---- Import local credibility scorer ----
USING_FALLBACK_SCORER = False
try:
    from credibility import evaluate_source  # must return {'score': 0..1, 'explanation': '...'}
except Exception:
    USING_FALLBACK_SCORER = True
    def evaluate_source(url: str, debug: bool=False):
        return {"score": 0.5 if url else 0.0, "explanation": "fallback scorer (credibility.py not found)"}  # safe fallback

# ---- Environment ----
load_dotenv()
PPLX_API_KEY = (os.getenv("PPLX_API_KEY") or "").strip()
PPLX_ENDPOINT = "https://api.perplexity.ai/chat/completions"

# ---- Perplexity Models ----
PPLX_MODELS = [
    ("sonar-pro",              "General high-quality model"),
    ("sonar",                  "Balanced general model"),
    ("sonar-small",            "Faster, smaller model"),
    ("sonar-reasoning-pro",    "Reasoning-optimized (pro)"),
    ("sonar-reasoning",        "Reasoning-optimized"),
]

# ---- Regex & roles ----
URL_RE = re.compile(r'(https?://[^\s)>\]]+)', re.IGNORECASE)
ALLOWED_ROLES = {"system", "user", "assistant"}

# ---------------- Helpers ----------------
def extract_urls(text: str) -> List[str]:
    if not text:
        return []
    return URL_RE.findall(text)

def band_from_score(score01: float) -> str:
    if score01 >= 0.85: return "High"
    if score01 >= 0.65: return "Moderate"
    return "Low"

def stars_from_score(score01: float) -> str:
    stars = round(score01 * 5)  # 0..5
    return "⭐" * stars + "☆" * (5 - stars)

def sanitize_messages_for_chat_completions(
    msgs: List[Dict[str, str]],
    system_prompt: Optional[str]
) -> List[Dict[str, str]]:
    """
    Repair messages to alternate strictly after any system message(s).
    Keeps only roles in {system,user,assistant} and non-empty content.
    Ensures the last message is a 'user' for /chat/completions.
    """
    cleaned: List[Dict[str, str]] = []
    if system_prompt:
        cleaned.append({"role": "system", "content": system_prompt})

    raw = [m for m in msgs if m.get("role") in ALLOWED_ROLES and str(m.get("content", "")).strip()]

    # If we start in an odd state after system, inject a small assistant bridge
    if cleaned and raw and raw[0]["role"] != "user":
        cleaned.append({"role": "assistant", "content": "Okay—how can I help?"})

    for m in raw:
        if not cleaned:
            cleaned.append(m)
        else:
            prev_role = cleaned[-1]["role"]
            if m["role"] == prev_role:
                # Coalesce consecutive same-role messages
                cleaned[-1]["content"] = f"{cleaned[-1]['content']}\n{m['content']}"
            else:
                cleaned.append(m)

    # Ensure final turn is user
    if not cleaned or cleaned[-1]["role"] != "user":
        cleaned.append({"role": "user", "content": "Please continue."})

    return cleaned

def call_perplexity(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    enable_web_search: bool
) -> Dict[str, Any]:
    if not PPLX_API_KEY:
        raise RuntimeError("Missing PPLX_API_KEY. Put it in .env or export in your shell.")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "extra_body": {
            "disable_search": not enable_web_search,
        },
    }
    headers = {
        "Authorization": f"Bearer {PPLX_API_KEY}",
        "Content-Type": "application/json",
    }
    r = requests.post(PPLX_ENDPOINT, headers=headers, data=json.dumps(payload), timeout=60)
    try:
        data = r.json()
    except Exception:
        r.raise_for_status()
        raise
    if r.status_code >= 400:
        raise RuntimeError(f"Error code: {r.status_code} - {json.dumps(data)}")
    return data

def get_assistant_text(api_resp: Dict[str, Any]) -> str:
    try:
        return api_resp["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(api_resp, indent=2)

# ---------------- UI ----------------
st.set_page_config(page_title="Project 1 — Credibility + Chat", page_icon="🔎", layout="wide")

# Subtle CSS polish
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
.card {
  border: 1px solid rgba(120,120,120,0.2);
  border-radius: 14px;
  padding: 1rem 1rem;
  background: rgba(250,250,250,0.65);
  margin-bottom: 0.75rem;
}
.metric-box {
  border-radius: 12px;
  padding: 0.75rem 0.85rem;
  background: #f8fafc;
  border: 1px solid rgba(120,120,120,0.18);
}
.small-cap { color: #64748b; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

left, right = st.columns([0.82, 0.18])
with left:
    st.markdown("### 🔎 Project 1 — URL Credibility + Chat")
    st.caption("Paste a URL to score credibility. Ask anything else to chat. Use the sidebar to toggle web search and scoring.")

with right:
    st.markdown(
        f"""
<div class="metric-box">
<b>Key:</b><br>
<span class="small-cap">{'✅ Perplexity key detected' if PPLX_API_KEY else '⚠️ No PPLX_API_KEY found'}</span><br>
<span class="small-cap">{'✅ Credibility: active' if not USING_FALLBACK_SCORER else '⚠️ Using fallback scorer (place credibility.py next to app.py)'}</span>
</div>
""",
        unsafe_allow_html=True
    )

# Sidebar Controls
with st.sidebar:
    st.subheader("⚙️ Controls")

    # Model picker
    model_labels = [f"{m} — {desc}" for (m, desc) in PPLX_MODELS]
    default_model = "sonar-pro"
    default_index = next((i for i,(m,_) in enumerate(PPLX_MODELS) if m == default_model), 0)
    model_choice = st.selectbox("Model", model_labels, index=default_index)
    model_selected = PPLX_MODELS[model_labels.index(model_choice)][0]

    # Toggles
    enable_web_search = st.toggle("Enable Web Search (chat)", value=True)
    show_scores = st.toggle("Show Credibility Scores (URLs)", value=True)

    # Sampling
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    top_p = st.slider("Top-p", min_value=0.5, max_value=1.0, value=0.95, step=0.01)

    # System prompt
    system_prompt = st.text_area(
        "System prompt (optional)",
        value=(
            "You are a concise, source-aware assistant. "
            "When web search is enabled, cite recent sources with titles and URLs. "
            "If the user message includes a URL, the app will score its credibility."
        ),
        height=110,
    )

    st.divider()

    # ---- CSV Export of scored URLs ----
    if "scores" not in st.session_state:
        st.session_state.scores = []  # list of dicts

    if st.session_state.scores:
        df = pd.DataFrame(st.session_state.scores)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Export scored URLs as CSV",
            data=csv_bytes,
            file_name=f"credibility_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        st.caption(f"{len(st.session_state.scores)} scored URL(s) ready to export.")
    else:
        st.caption("No scored URLs yet to export.")

    if USING_FALLBACK_SCORER and show_scores:
        st.warning("Using fallback scorer — add credibility.py (with evaluate_source) next to app.py.", icon="⚠️")
    if not PPLX_API_KEY:
        st.error("PPLX_API_KEY not found — chat will fail.", icon="🚫")

# Session chat state
if "chat" not in st.session_state:
    st.session_state.chat: List[Dict[str, str]] = []

# Conversation history (compact)
with st.expander("💬 Conversation history", expanded=False):
    if not st.session_state.chat:
        st.info("No messages yet.")
    for m in st.session_state.chat:
        role = m["role"]
        if role == "user":
            st.chat_message("user").markdown(m["content"])
        elif role == "assistant":
            st.chat_message("assistant").markdown(m["content"])
        else:
            st.caption(f"{role}: {m['content']}")

# ---- URL scoring renderer
def render_credibility_for_urls(urls: List[str], show_scores_flag: bool):
    st.subheader("🔐 Credibility results")
    for url in urls:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"**URL:** {url}")
        if not show_scores_flag:
            st.info("Scoring hidden (toggle is off).")
            st.markdown("</div>", unsafe_allow_html=True)
            continue

        try:
            res = evaluate_source(url, debug=False)  # {'score': 0..1, 'explanation': str}
            s01 = float(res.get("score", 0.0))
            score100 = round(s01 * 100, 1)
            band = band_from_score(s01)
            stars = stars_from_score(s01)

            c1, c2, c3 = st.columns([1.2, 1, 2])
            with c1:
                st.metric("Score", f"{score100}/100")
                st.write(f"Band: **{band}**")
            with c2:
                st.markdown(f"### {stars}")
            with c3:
                st.caption("Explanation")
                st.write(res.get("explanation", "(no details)"))

            # Record this result for CSV export
            st.session_state.scores.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "url": url,
                "score": score100,
                "band": band,
                "stars": stars,
                "explanation": res.get("explanation", ""),
            })

        except Exception as e:
            st.error(f"Scoring failed for {url}: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

# ---- Chat handler
def handle_user_message(text: str):
    urls = extract_urls(text)
    if urls:
        st.session_state.chat.append({"role": "user", "content": text})
        render_credibility_for_urls(urls, show_scores_flag=show_scores)
        st.session_state.chat.append({"role": "assistant", "content": "I’ve scored the URL(s) above. Paste another URL or ask a question."})
        return

    # Chat path
    st.session_state.chat.append({"role": "user", "content": text})
    msgs = sanitize_messages_for_chat_completions(st.session_state.chat, system_prompt)

    with st.spinner("Thinking…"):
        try:
            resp = call_perplexity(
                messages=msgs,
                model=model_selected,
                temperature=temperature,
                top_p=top_p,
                max_tokens=800,
                enable_web_search=enable_web_search,
            )
            assistant_text = get_assistant_text(resp)
        except Exception as e:
            assistant_text = f"⚠️ Chat error: {e}"

    st.session_state.chat.append({"role": "assistant", "content": assistant_text})
    st.chat_message("assistant").markdown(assistant_text)

# ---- Render last assistant reply if any (so refresh keeps it visible)
if st.session_state.chat and st.session_state.chat[-1]["role"] == "assistant":
    st.chat_message("assistant").markdown(st.session_state.chat[-1]["content"])

# ---- Input box
user_input = st.chat_input("Type a URL to score credibility, or ask a question…")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    handle_user_message(user_input)

st.markdown("---")
st.caption("Project 1 — Deliverable 3: URL credibility scoring integrated into a Streamlit chatbot. CSV export available in the sidebar.")
