"""
app.py — Interface de démonstration du pipeline de mapping référentiels.

Lancement :
    streamlit run app.py
"""
import os
import sys
import time
import json
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── Matplotlib theme adaptatif ────────────────────────────────────────────────
import subprocess, json as _json
# Streamlit passe la préférence via la query string — on choisit dark par défaut
# Les deux palettes sont définies ; le graphique reçoit le bon contexte via st.query_params
_MPLRC_DARK = {
    "figure.facecolor": "#060d0a", "axes.facecolor": "#091a11",
    "axes.edgecolor": "#1a3d28",   "axes.labelcolor": "#a5d6a7",
    "axes.titlecolor": "#00e676",  "xtick.color": "#4caf50",
    "ytick.color": "#4caf50",      "text.color": "#e8f5e9",
    "grid.color": "#1a3d28",       "figure.dpi": 110,
}
_MPLRC_LIGHT = {
    "figure.facecolor": "#f4fdf7", "axes.facecolor": "#eaf7ef",
    "axes.edgecolor": "#b8dfc4",   "axes.labelcolor": "#1a4d2e",
    "axes.titlecolor": "#00b248",  "xtick.color": "#2e7d32",
    "ytick.color": "#2e7d32",      "text.color": "#0a1f12",
    "grid.color": "#c8e6c9",       "figure.dpi": 110,
}
plt.rcParams.update(_MPLRC_DARK)   # défaut dark ; mis à jour en runtime si light

import pipeline_standalone as ps
from pipeline_standalone import (
    list_frameworks, load_framework, get_expected_count,
    run_similarity, run_scorer, run_cleanup, _export_results,
    EMBEDDING_MODEL, SEMANTIC_THRESHOLD, TOP_K, LLM_MODEL,
    LLM_MIN_CONFIDENCE, LLM_BATCH_SIZE,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RiskHunter — Mapping Pipeline",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── RiskHunter Brand CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Google Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  /* ── Variables ── */
  :root {
    --rh-bg:          #05100a;
    --rh-bg2:         #091a11;
    --rh-bg3:         #0d2218;
    --rh-green:       #00e676;
    --rh-green-dim:   #00c853;
    --rh-green-glow:  rgba(0, 230, 118, 0.15);
    --rh-green-border:rgba(0, 230, 118, 0.25);
    --rh-text:        #e8f5e9;
    --rh-muted:       #4caf50aa;
    --rh-card:        rgba(9, 26, 17, 0.92);
    --rh-radius:      10px;
  }

  /* ── Global background & font ── */
  html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: radial-gradient(ellipse at 60% 10%, #0d2e18 0%, #05100a 60%) !important;
    color: var(--rh-text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
  }

  /* Subtle animated grid overlay */
  [data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image:
      linear-gradient(rgba(0,230,118,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,230,118,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071510 0%, #040e08 100%) !important;
    border-right: 1px solid var(--rh-green-border) !important;
  }
  [data-testid="stSidebar"] * { color: var(--rh-text) !important; }

  /* ── Title in sidebar ── */
  [data-testid="stSidebar"] h1 {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--rh-green) !important;
    font-size: 1.3rem !important;
    letter-spacing: 0.05em;
  }

  /* ── Main title ── */
  h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    letter-spacing: -0.02em;
  }
  h1 span.accent { color: var(--rh-green); }

  h2, h3 {
    color: var(--rh-green) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
  }

  /* ── Metric cards (st.metric) ── */
  [data-testid="stMetric"] {
    background: var(--rh-card) !important;
    border: 1px solid var(--rh-green-border) !important;
    border-radius: var(--rh-radius) !important;
    padding: 16px 20px !important;
    box-shadow: 0 0 20px var(--rh-green-glow);
    transition: box-shadow 0.3s;
  }
  [data-testid="stMetric"]:hover {
    box-shadow: 0 0 32px rgba(0,230,118,0.28);
  }
  [data-testid="stMetricValue"] {
    color: var(--rh-green) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
  }
  [data-testid="stMetricLabel"] {
    color: var(--rh-muted) !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  [data-testid="stMetricDelta"] { color: var(--rh-green-dim) !important; }

  /* ── Primary button ── */
  [data-testid="stBaseButton-primary"] > button,
  button[kind="primary"] {
    background: var(--rh-green) !important;
    color: #05100a !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    box-shadow: 0 0 18px rgba(0,230,118,0.35);
    transition: all 0.2s;
  }
  [data-testid="stBaseButton-primary"] > button:hover,
  button[kind="primary"]:hover {
    background: #00ff8a !important;
    box-shadow: 0 0 28px rgba(0,255,138,0.5);
    transform: translateY(-1px);
  }

  /* ── Secondary buttons ── */
  button[kind="secondary"], [data-testid="stBaseButton-secondary"] > button {
    background: transparent !important;
    border: 1px solid var(--rh-green-border) !important;
    color: var(--rh-green) !important;
    border-radius: 6px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    transition: all 0.2s;
  }
  button[kind="secondary"]:hover { background: var(--rh-green-glow) !important; }

  /* ── Selectbox & inputs ── */
  [data-testid="stSelectbox"] > div > div,
  [data-testid="stTextInput"] > div > div > input,
  [data-testid="stPasswordInput"] > div > div > input {
    background: var(--rh-bg3) !important;
    border: 1px solid var(--rh-green-border) !important;
    border-radius: 6px !important;
    color: var(--rh-text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
  }
  [data-testid="stSelectbox"] svg { color: var(--rh-green) !important; }

  /* ── Sliders ── */
  [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: var(--rh-green) !important;
    border-color: var(--rh-green) !important;
    box-shadow: 0 0 8px rgba(0,230,118,0.5);
  }
  [data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stSliderTrack"] > div:first-child {
    background: var(--rh-bg3) !important;
  }
  [data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stSliderTrack"] > div:last-child {
    background: var(--rh-green) !important;
  }

  /* ── Tabs ── */
  [data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--rh-green-border) !important;
    gap: 4px;
  }
  [data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--rh-muted) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 8px 18px !important;
    transition: all 0.2s;
  }
  [data-testid="stTabs"] [aria-selected="true"] {
    color: var(--rh-green) !important;
    background: var(--rh-green-glow) !important;
    border-bottom: 2px solid var(--rh-green) !important;
  }

  /* ── DataFrames / tables ── */
  [data-testid="stDataFrame"] {
    border: 1px solid var(--rh-green-border) !important;
    border-radius: var(--rh-radius) !important;
    overflow: hidden;
  }
  [data-testid="stDataFrame"] thead th {
    background: var(--rh-bg3) !important;
    color: var(--rh-green) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border-bottom: 1px solid var(--rh-green-border) !important;
  }
  [data-testid="stDataFrame"] tbody tr:hover td {
    background: var(--rh-green-glow) !important;
  }

  /* ── Status / progress ── */
  [data-testid="stStatusWidget"] {
    background: var(--rh-card) !important;
    border: 1px solid var(--rh-green-border) !important;
    border-radius: var(--rh-radius) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
  }

  /* ── Divider ── */
  hr { border-color: var(--rh-green-border) !important; }

  /* ── Info / warning boxes ── */
  [data-testid="stAlert"] {
    border-radius: var(--rh-radius) !important;
    border-left: 3px solid var(--rh-green) !important;
    background: var(--rh-card) !important;
  }

  /* ── Checkbox ── */
  [data-testid="stCheckbox"] label span {
    color: var(--rh-text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
  }

  /* ── Scrollbars ── */
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: var(--rh-bg); }
  ::-webkit-scrollbar-thumb { background: var(--rh-green-dim); border-radius: 3px; }

  /* ── Download buttons ── */
  [data-testid="stDownloadButton"] button {
    background: transparent !important;
    border: 1px solid var(--rh-green-border) !important;
    color: var(--rh-green) !important;
    border-radius: 6px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500;
    transition: all 0.2s;
  }
  [data-testid="stDownloadButton"] button:hover {
    background: var(--rh-green-glow) !important;
    border-color: var(--rh-green) !important;
  }

  /* ── Caption / small text ── */
  [data-testid="stCaptionContainer"] p, .stCaption {
    color: var(--rh-muted) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
  }

  /* ── Multiselect ── */
  [data-baseweb="tag"] {
    background: var(--rh-green-glow) !important;
    border: 1px solid var(--rh-green-border) !important;
    color: var(--rh-green) !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
  }

  /* ── Relation type badges ── */
  .badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    font-weight: 600; letter-spacing: 0.04em; margin: 2px;
  }
  .equivalence { background: rgba(0,230,118,0.15); color: #00e676; border: 1px solid rgba(0,230,118,0.3); }
  .A_couvre_B  { background: rgba(255,214,0,0.12);  color: #ffd600; border: 1px solid rgba(255,214,0,0.3); }
  .B_couvre_A  { background: rgba(255,111,0,0.12);  color: #ff9100; border: 1px solid rgba(255,111,0,0.3); }
  .partielle   { background: rgba(30,136,229,0.12); color: #40c4ff; border: 1px solid rgba(30,136,229,0.3); }
  .aucun_lien  { background: rgba(80,80,80,0.2);    color: #9e9e9e; border: 1px solid rgba(120,120,120,0.2); }

  /* ── Hero banner ── */
  .rh-hero {
    background: linear-gradient(135deg, var(--rh-bg3) 0%, rgba(0,40,20,0.6) 100%);
    border: 1px solid var(--rh-green-border);
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
  }
  .rh-hero::before {
    content: '';
    position: absolute; top: -40%; right: -10%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,230,118,0.08) 0%, transparent 70%);
    pointer-events: none;
  }
  .rh-hero h2 {
    font-size: 1.5rem !important;
    margin: 0 0 6px !important;
    color: #fff !important;
  }
  .rh-hero h2 span { color: var(--rh-green); }
  .rh-hero p { color: #88b898 !important; font-size: 0.9rem; margin: 0; }

  /* ── Stat row ── */
  .stat-row {
    display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0;
  }
  .stat-chip {
    background: var(--rh-bg3);
    border: 1px solid var(--rh-green-border);
    border-radius: 8px;
    padding: 10px 18px;
    text-align: center;
    flex: 1; min-width: 120px;
  }
  .stat-chip .sv { font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 700; color: var(--rh-green); }
  .stat-chip .sl { font-size: 0.72rem; color: var(--rh-muted); text-transform: uppercase; letter-spacing: 0.07em; margin-top: 2px; }

  /* ── Framework list ── */
  .fw-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 8px; margin-top: 12px;
  }
  .fw-chip {
    background: var(--rh-bg3); border: 1px solid var(--rh-green-border);
    border-radius: 6px; padding: 7px 12px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    color: var(--rh-green); letter-spacing: 0.03em;
    transition: background 0.2s;
  }
  .fw-chip:hover { background: var(--rh-green-glow); }

  /* ════════════════════════════════════════════════════════════
     LIGHT MODE — RiskHunter brand (white + green gradient)
     ════════════════════════════════════════════════════════════ */
  @media (prefers-color-scheme: light) {
    :root {
      --rh-bg:           #ffffff;
      --rh-bg2:          #f4fdf7;
      --rh-bg3:          #eaf7ef;
      --rh-green:        #00b248;
      --rh-green-dim:    #00c853;
      --rh-green-glow:   rgba(0, 178, 72, 0.10);
      --rh-green-border: rgba(0, 178, 72, 0.22);
      --rh-text:         #0a1f12;
      --rh-muted:        #3a8a52;
      --rh-card:         rgba(255,255,255,0.9);
    }

    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stApp"] {
      background: radial-gradient(ellipse at 15% 15%, #b8f0d4 0%, #e8fdf0 30%, #ffffff 70%) !important;
    }

    /* ── Native Streamlit header / toolbar ── */
    header[data-testid="stHeader"] {
      background: rgba(244, 253, 247, 0.92) !important;
      backdrop-filter: blur(8px) !important;
      border-bottom: 1px solid rgba(0,178,72,0.15) !important;
    }
    header[data-testid="stHeader"] * { color: #0a1f12 !important; }
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] a,
    header[data-testid="stHeader"] svg { color: #0a1f12 !important; fill: #0a1f12 !important; }
    [data-testid="stToolbar"] { color: #0a1f12 !important; }
    [data-testid="stDecoration"] { background: var(--rh-green) !important; }

    [data-testid="stAppViewContainer"]::before {
      background-image:
        linear-gradient(rgba(0,178,72,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,178,72,0.04) 1px, transparent 1px);
    }

    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #f0fdf4 0%, #e8fbef 100%) !important;
      border-right: 1px solid var(--rh-green-border) !important;
    }
    [data-testid="stSidebar"] * { color: #0a1f12 !important; }
    [data-testid="stSidebar"] h1 { color: var(--rh-green) !important; }

    h1 { color: #0a1f12 !important; }
    h2, h3 { color: var(--rh-green) !important; }

    [data-testid="stMetric"] {
      background: rgba(255,255,255,0.95) !important;
      box-shadow: 0 2px 16px rgba(0,178,72,0.12);
    }
    [data-testid="stMetricValue"]  { color: var(--rh-green) !important; }
    [data-testid="stMetricLabel"]  { color: #3a8a52 !important; }
    [data-testid="stMetricDelta"]  { color: var(--rh-green-dim) !important; }

    [data-testid="stBaseButton-primary"] > button,
    button[kind="primary"] {
      background: var(--rh-green) !important;
      color: #ffffff !important;
      box-shadow: 0 2px 14px rgba(0,178,72,0.30);
    }
    [data-testid="stBaseButton-primary"] > button:hover,
    button[kind="primary"]:hover {
      background: #00c853 !important;
      box-shadow: 0 4px 22px rgba(0,200,83,0.40);
    }

    button[kind="secondary"],
    [data-testid="stBaseButton-secondary"] > button {
      color: var(--rh-green) !important;
      border-color: var(--rh-green-border) !important;
    }

    [data-testid="stSelectbox"] > div > div,
    [data-testid="stTextInput"] > div > div > input,
    [data-testid="stPasswordInput"] > div > div > input {
      background: #ffffff !important;
      border-color: var(--rh-green-border) !important;
      color: #0a1f12 !important;
    }

    [data-testid="stTabs"] [data-baseweb="tab-list"] {
      border-bottom-color: var(--rh-green-border) !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] { color: #3a8a52 !important; }
    [data-testid="stTabs"] [aria-selected="true"] {
      color: var(--rh-green) !important;
      background: var(--rh-green-glow) !important;
      border-bottom-color: var(--rh-green) !important;
    }

    [data-testid="stDataFrame"] {
      border-color: var(--rh-green-border) !important;
    }
    [data-testid="stDataFrame"] thead th {
      background: #eaf7ef !important;
      color: var(--rh-green) !important;
      border-bottom-color: var(--rh-green-border) !important;
    }
    [data-testid="stDataFrame"] tbody tr:hover td {
      background: rgba(0,178,72,0.06) !important;
    }

    [data-testid="stStatusWidget"] {
      background: rgba(255,255,255,0.95) !important;
      border-color: var(--rh-green-border) !important;
    }

    hr { border-color: var(--rh-green-border) !important; }

    [data-testid="stAlert"] {
      background: rgba(255,255,255,0.9) !important;
      border-left-color: var(--rh-green) !important;
    }

    [data-baseweb="tag"] {
      background: rgba(0,178,72,0.08) !important;
      color: var(--rh-green) !important;
      border-color: var(--rh-green-border) !important;
    }

    .equivalence { background: rgba(0,178,72,0.10); color: #00823a; border-color: rgba(0,178,72,0.25); }
    .A_couvre_B  { background: rgba(245,170,0,0.10); color: #b37800; border-color: rgba(245,170,0,0.25); }
    .B_couvre_A  { background: rgba(230,90,0,0.10);  color: #a03800; border-color: rgba(230,90,0,0.25); }
    .partielle   { background: rgba(0,120,200,0.08); color: #005fa3; border-color: rgba(0,120,200,0.2); }
    .aucun_lien  { background: rgba(100,100,100,0.08); color: #666; border-color: rgba(150,150,150,0.2); }

    .rh-hero {
      background: linear-gradient(135deg, #e6f9ee 0%, rgba(200,245,220,0.5) 100%);
      border-color: rgba(0,178,72,0.2);
    }
    .rh-hero::before {
      background: radial-gradient(circle, rgba(0,200,83,0.12) 0%, transparent 70%);
    }
    .rh-hero h2    { color: #0a1f12 !important; }
    .rh-hero h2 span { color: var(--rh-green); }
    .rh-hero p     { color: #3a6b48 !important; }

    .stat-chip { background: #ffffff; border-color: var(--rh-green-border); }
    .stat-chip .sv { color: var(--rh-green); }
    .stat-chip .sl { color: #3a8a52; }

    .fw-chip {
      background: #ffffff;
      border-color: rgba(0,178,72,0.18);
      color: var(--rh-green);
    }
    .fw-chip:hover { background: rgba(0,178,72,0.07); }

    [data-testid="stDownloadButton"] button {
      color: var(--rh-green) !important;
      border-color: var(--rh-green-border) !important;
    }

    [data-testid="stCaptionContainer"] p, .stCaption {
      color: #3a8a52 !important;
    }

    ::-webkit-scrollbar-track { background: #f4fdf7; }
    ::-webkit-scrollbar-thumb { background: #00c853; }
  }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 16px;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
        <span style="font-size:1.4rem;">✦</span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:1.15rem;font-weight:700;color:#00e676;letter-spacing:0.04em;">RISK HUNTER</span>
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#4caf5088;letter-spacing:0.1em;text-transform:uppercase;padding-left:34px;">Mapping Pipeline</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    frameworks = list_frameworks()
    fw_names = sorted(frameworks.keys())

    st.subheader("📂 Référentiels")
    ref_a_name = st.selectbox("Référentiel A", fw_names,
                               index=fw_names.index("cis-controls-v8") if "cis-controls-v8" in fw_names else 0)
    ref_b_name = st.selectbox("Référentiel B", fw_names,
                               index=fw_names.index("nistCsfV2") if "nistCsfV2" in fw_names else 1)

    st.divider()
    st.subheader("⚙️ Paramètres")
    top_k     = st.slider("Top-K (paires / exigence)", 1, 20, TOP_K)
    sem_thresh = st.slider("Seuil sémantique", 0.1, 0.9, SEMANTIC_THRESHOLD, step=0.05)

    st.divider()
    st.subheader("🤖 Étape 4 — LLM")
    run_llm = st.checkbox("Activer le scoring LLM", value=False)
    if run_llm:
        # Priorité : st.secrets (Streamlit Cloud) > .env local > saisie manuelle
        _secret_key = st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=_secret_key,
            placeholder="sk-... (laisse vide si configuré dans Streamlit secrets)",
        )
        llm_model = st.selectbox("Modèle", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                                  index=0)
        min_conf  = st.slider("Confiance minimale (étape 5)", 0.0, 1.0, LLM_MIN_CONFIDENCE, 0.05,
                              help="Relations avec confiance < seuil seront supprimées à l'étape 5")
        skip_cleanup = st.checkbox("Désactiver étape 5 (garder aucun_lien)", value=False)
    else:
        api_key      = ""
        llm_model    = LLM_MODEL
        min_conf     = LLM_MIN_CONFIDENCE
        skip_cleanup = False

    st.divider()
    run_btn = st.button("🚀 Lancer le pipeline", type="primary", use_container_width=True)
    st.caption("Made with ❤️ by Sensey | \t Github Profile : https://github.com/sorooty")


# ── Theme detection (matplotlib palette) ─────────────────────────────────────
_is_light = st.get_option("theme.base") == "light"
plt.rcParams.update(_MPLRC_LIGHT if _is_light else _MPLRC_DARK)
_heatmap_cmap = "RdYlGn"   # contraste clair : rouge=faible, vert=fort
_hist_color  = "#00b248" if _is_light else "#00e676"
_pie_colors  = (["#00b248","#f5aa00","#e65a00","#1e88e5","#9e9e9e"]
                if _is_light else
                ["#00e676","#ffd600","#ff9100","#40c4ff","#616161"])


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="rh-hero">
  <h2>✦ RISK HUNTER <span>/ Mapping Pipeline</span></h2>
  <p>Automated semantic mapping between security & compliance frameworks — powered by embeddings + LLM verification</p>
</div>
""", unsafe_allow_html=True)
st.markdown(f"**`{ref_a_name}`** &nbsp;↔&nbsp; **`{ref_b_name}`**")

if ref_a_name == ref_b_name:
    st.warning("⚠️ Sélectionne deux référentiels différents.")
    st.stop()

# ─ State ─────────────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None

# ─ Pipeline run ──────────────────────────────────────────────────────────────
if run_btn:
    st.session_state.results = None
    results = {}

    # ── Étape 1 ──────────────────────────────────────────────────────────────
    with st.status("**Étape 1** — Parsing & normalisation ...", expanded=True) as s:
        t0 = time.time()
        ref_A = load_framework(frameworks[ref_a_name], framework_name=ref_a_name)
        ref_B = load_framework(frameworks[ref_b_name], framework_name=ref_b_name)
        exp_A = get_expected_count(ref_a_name)
        exp_B = get_expected_count(ref_b_name)
        lbl_A = f"{len(ref_A)}/{exp_A}" if exp_A else str(len(ref_A))
        lbl_B = f"{len(ref_B)}/{exp_B}" if exp_B else str(len(ref_B))
        ok_A  = "✅" if (exp_A is None or len(ref_A) == exp_A) else "⚠️"
        ok_B  = "✅" if (exp_B is None or len(ref_B) == exp_B) else "⚠️"
        s.update(label=f"{ok_A} Étape 1 — A : {lbl_A} exigences · {ok_B} B : {lbl_B} exigences  ({time.time()-t0:.1f}s)")
    results["ref_A"] = ref_A
    results["ref_B"] = ref_B

    # ── Étape 2 ──────────────────────────────────────────────────────────────
    with st.status("**Étape 2** — Similarité sémantique (embeddings) ...", expanded=True) as s:
        t0 = time.time()
        candidates, matrix = run_similarity(
            ref_A, ref_B,
            force=True,
            semantic_threshold=sem_thresh,
            top_k=top_k,
            fw_a_name=ref_a_name,
            fw_b_name=ref_b_name,
        )
        total_combinations = len(ref_A) * len(ref_B)
        pct = 100 * len(candidates) / total_combinations
        s.update(label=f"✅ Étape 2 — {len(candidates)} paires ({pct:.1f}% de {total_combinations}) en {time.time()-t0:.1f}s")
    results["matrix"]     = matrix
    results["candidates"] = candidates

    # ── Étape 3 ──────────────────────────────────────────────────────────────
    with st.status("**Étape 3** — Génération de la heatmap ...", expanded=True) as s:
        t0 = time.time()
        # Taille cappée : max 20×14 pour rester lisible dans l'UI
        fig_w = min(20, max(10, len(ref_B) * 0.13))
        fig_h = min(14, max(7,  len(ref_A) * 0.08))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        # vmin dynamique au 5e percentile → contraste maximal sur la plage réelle
        vmin_dynamic = float(np.percentile(matrix, 5))
        sns.heatmap(matrix,
                    xticklabels=[r.id for r in ref_B],
                    yticklabels=[r.id for r in ref_A],
                    cmap=_heatmap_cmap,
                    vmin=vmin_dynamic, vmax=1.0, ax=ax,
                    linewidths=0, cbar_kws={"label": "Similarité cosinus"})
        ax.set_title(f"Similarité sémantique — {ref_a_name} (A) ↔ {ref_b_name} (B)", fontsize=11)
        ax.set_xlabel("Ref B", fontsize=9)
        ax.set_ylabel("Ref A", fontsize=9)
        ax.tick_params(axis="x", labelsize=4, rotation=90)
        ax.tick_params(axis="y", labelsize=4)
        plt.tight_layout()
        results["heatmap_fig"] = fig
        s.update(label=f"✅ Étape 3 — Heatmap générée en {time.time()-t0:.1f}s")

    # ── Étape 4 (optionnel) ───────────────────────────────────────────────────
    relations = None
    removed   = []
    if run_llm and api_key:
        with st.status("**Étape 4** — Scoring LLM ...", expanded=True) as s:
            t0 = time.time()
            os.environ["OPENAI_API_KEY"] = api_key
            relations = run_scorer(
                candidates, ref_A, ref_B,
                force=True,
                llm_model=llm_model,
                batch_size=LLM_BATCH_SIZE,
            )
            s.update(label=f"✅ Étape 4 — {len(relations)} relations scorées en {time.time()-t0:.1f}s")

        # ── Étape 5 — Nettoyage ──────────────────────────────────────────────
        if not skip_cleanup:
            with st.status("**Étape 5** — Nettoyage des liaisons inutiles ...", expanded=True) as s:
                relations, removed = run_cleanup(relations, min_confidence=min_conf)
                s.update(
                    label=(f"✅ Étape 5 — {len(relations)} relations conservées, "
                           f"{len(removed)} supprimées (aucun_lien / conf < {min_conf:.2f})")
                )
    elif run_llm and not api_key:
        st.error("❌ Clé OpenAI manquante.")

    results["relations"] = relations
    results["removed"]   = removed
    st.session_state.results = results
    st.success("✅ Pipeline terminé !")


# ─ Display results ────────────────────────────────────────────────────────────
if st.session_state.results:
    r = st.session_state.results
    ref_A      = r["ref_A"]
    ref_B      = r["ref_B"]
    matrix     = r["matrix"]
    candidates = r["candidates"]
    relations  = r["relations"]
    removed    = r.get("removed", [])

    # ── Métriques ─────────────────────────────────────────────────────────────
    total_comb = len(ref_A) * len(ref_B)
    pct = 100 * len(candidates) / total_comb

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📋 Exigences A", len(ref_A))
    with col2:
        st.metric("📋 Exigences B", len(ref_B))
    with col3:
        st.metric("🔗 Paires candidates", len(candidates), delta=f"{pct:.1f}% des {total_comb}")
    with col4:
        if relations:
            st.metric("✅ Relations conservées", len(relations),
                      delta=f"-{len(removed)} supprimées" if removed else None)
        else:
            st.metric("💡 LLM", "Non lancé")

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs(["🗺️ Heatmap", "🔗 Paires candidates", "✅ Mapping final", "📥 Export"])

    # ── Tab 1 : Heatmap ───────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Matrice de similarité sémantique")
        st.caption("Couleur = score cosinus entre les titres. Plus c'est vert, plus les exigences sont proches sémantiquement.")
        st.pyplot(r["heatmap_fig"], use_container_width=True)

    # ── Tab 2 : Paires candidates ─────────────────────────────────────────────
    with tabs[1]:
        st.subheader(f"Paires candidates — {len(candidates)} paires")
        df_c = pd.DataFrame([c.to_dict() for c in candidates])

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            min_score = st.slider("Score min", 0.0, 1.0, float(df_c["semantic_score"].min()), 0.01, key="min_cand")
        with col_f2:
            search = st.text_input("🔍 Filtrer par titre", "", key="search_cand")

        filtered = df_c[df_c["semantic_score"] >= min_score]
        if search:
            mask = (filtered["title_A"].str.contains(search, case=False, na=False) |
                    filtered["title_B"].str.contains(search, case=False, na=False))
            filtered = filtered[mask]

        st.dataframe(
            filtered.sort_values("semantic_score", ascending=False),
            use_container_width=True,
            column_config={
                "semantic_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1, format="%.2f"),
                "title_A": st.column_config.TextColumn("Titre A", width=300),
                "title_B": st.column_config.TextColumn("Titre B", width=300),
            },
            hide_index=True,
        )

        # Distribution des scores
        fig_dist, ax_dist = plt.subplots(figsize=(7, 3))
        ax_dist.hist(df_c["semantic_score"], bins=30, color=_hist_color, edgecolor="none", alpha=0.85)
        ax_dist.set_xlabel("Score cosinus")
        ax_dist.set_ylabel("Paires")
        ax_dist.set_title("Distribution des scores sémantiques")
        plt.tight_layout()
        st.pyplot(fig_dist, use_container_width=False)

    # ── Tab 3 : Mapping final ─────────────────────────────────────────────────
    with tabs[2]:
        if not relations:
            st.info("💡 Active **Scoring LLM** dans la barre latérale et relance le pipeline pour obtenir le mapping final.")
        else:
            df_r = pd.DataFrame([rel.to_dict() for rel in relations])

            COLORS = {
                "equivalence": "🟢", "A_couvre_B": "🟡",
                "B_couvre_A": "🟠", "partielle": "🔵", "aucun_lien": "⚫",
            }

            # Filtre
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                rt_filter = st.multiselect(
                    "Type de relation",
                    options=df_r["relation_type"].unique().tolist(),
                    default=df_r["relation_type"].unique().tolist(),
                )
            with col_r2:
                conf_min = st.slider("Confiance min", 0.0, 1.0, 0.0, 0.05)

            mask = df_r["relation_type"].isin(rt_filter) & (df_r["confidence"] >= conf_min)
            df_show = df_r[mask].sort_values("confidence", ascending=False)

            st.caption(f"{len(df_show)} relations affichées")

            col_config = {
                "relation_type":   st.column_config.TextColumn("Type", width=120),
                "title_A":         st.column_config.TextColumn(f"Titre {ref_a_name}", width=260),
                "title_B":         st.column_config.TextColumn(f"Titre {ref_b_name}", width=260),
                "semantic_score":  st.column_config.ProgressColumn("Sémantique", min_value=0, max_value=1, format="%.2f"),
                "coverage_A_to_B": st.column_config.ProgressColumn("Cov A→B", min_value=0, max_value=1, format="%.2f"),
                "coverage_B_to_A": st.column_config.ProgressColumn("Cov B→A", min_value=0, max_value=1, format="%.2f"),
                "confidence":      st.column_config.ProgressColumn("Confiance", min_value=0, max_value=1, format="%.2f"),
                "justification":   st.column_config.TextColumn("Justification", width=400,
                    help="Explication du LLM pour ce choix de relation"),
            }
            st.dataframe(df_show, use_container_width=True, column_config=col_config, hide_index=True)

            # Répartition types de relation
            st.subheader("Types de relation")
            vc = df_r["relation_type"].value_counts()
            fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
            ax_pie.pie(vc.values, labels=[f"{COLORS.get(l,'')} {l}" for l in vc.index],
                       colors=_pie_colors[:len(vc)], autopct="%1.0f%%", startangle=90)
            ax_pie.set_title("Mapping final")
            plt.tight_layout()
            st.pyplot(fig_pie)

    # ── Tab 4 : Export ────────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Export des résultats")

        df_c = pd.DataFrame([c.to_dict() for c in candidates])
        csv_candidates = df_c.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Télécharger les paires candidates (CSV)",
            data=csv_candidates,
            file_name=f"candidates_{ref_a_name}_vs_{ref_b_name}.csv",
            mime="text/csv",
        )

        if relations:
            df_r = pd.DataFrame([rel.to_dict() for rel in relations])
            csv_relations = df_r.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Télécharger le mapping final (CSV)",
                data=csv_relations,
                file_name=f"mapping_{ref_a_name}_vs_{ref_b_name}.csv",
                mime="text/csv",
            )

            # Excel
            try:
                xlsx_path = ps.OUTPUT_DIR / f"mapping_{ref_a_name}_vs_{ref_b_name}.xlsx"
                if xlsx_path.exists():
                    with open(xlsx_path, "rb") as f:
                        st.download_button(
                            "📥 Télécharger le mapping final (Excel)",
                            data=f.read(),
                            file_name=xlsx_path.name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
            except Exception:
                pass

        # JSON des exigences normalisées
        json_A = json.dumps([r.to_dict() for r in ref_A], ensure_ascii=False, indent=2).encode("utf-8")
        json_B = json.dumps([r.to_dict() for r in ref_B], ensure_ascii=False, indent=2).encode("utf-8")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.download_button(f"📥 {ref_a_name}.json", json_A,
                               file_name=f"{ref_a_name}_normalized.json", mime="application/json")
        with col_e2:
            st.download_button(f"📥 {ref_b_name}.json", json_B,
                               file_name=f"{ref_b_name}_normalized.json", mime="application/json")

else:
    # ── Écran d'accueil ────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin: 32px 0 24px;">
      <p style="color:#88b898;font-size:1rem;max-width:600px;line-height:1.6;">
        Sélectionne deux référentiels dans la barre latérale, configure les paramètres
        et clique sur <strong style="color:#00e676;">Lancer le pipeline</strong> pour démarrer l'analyse.
      </p>
    </div>
    """, unsafe_allow_html=True)

    frameworks = list_frameworks()
    st.markdown(f"""
    <div style="margin-bottom:8px;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#00e676;text-transform:uppercase;letter-spacing:0.1em;">
        ◆ {len(frameworks)} référentiels disponibles
      </span>
    </div>
    <div class="fw-grid">
      {"".join(f'<div class="fw-chip">{name}</div>' for name in sorted(frameworks.keys()))}
    </div>
    """, unsafe_allow_html=True)


