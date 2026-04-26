"""
Tax Auditor AI — Enterprise UI
Professional two-year comparison dashboard with charts.
"""

import os
import re
import threading
import uuid
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Background analysis result store ──────────────────────────────────────────
_bg_results: dict = {}
_bg_lock = threading.Lock()

from ingestion import ingest_file, collection_stats, delete_collection, ingest_directory
from agent import chat, chat_stream, build_user_message, analyze_documents, analyze_transactions, savings_recommendations
from transaction_parser import parse_transactions, summarize_transactions

@st.cache_resource(show_spinner="Loading IRS knowledge base…")
def _ensure_irs_loaded():
    """Auto-ingest IRS docs on first startup (cloud or local)."""
    from ingestion import collection_stats as _cs
    if _cs()["irs_regulations"] == 0:
        irs_dir = Path(__file__).parent / "irs_docs"
        if irs_dir.exists():
            ingest_directory(str(irs_dir), "irs_regulations")

_ensure_irs_loaded()

USER_UPLOADS_DIR = Path(__file__).parent / "user_uploads"
USER_UPLOADS_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="TaxAI Pro — Tax Compliance & Advisory",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #07090f; color: #e2e8f0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 100% !important; }

/* ── Topnav ── */
.topnav {
  background: linear-gradient(90deg,#0d1b2e 0%,#0f2645 60%,#0d1b2e 100%);
  border-bottom: 2px solid #1e4a8a;
  padding: 0 2rem; height: 64px;
  display: flex; align-items: center; justify-content: space-between;
  box-shadow: 0 2px 20px rgba(0,0,0,.5);
}
.topnav-logo {
  width:36px;height:36px;
  background:linear-gradient(135deg,#1a5c96,#00a89c);
  border-radius:8px;display:flex;align-items:center;justify-content:center;
  font-size:18px;font-weight:900;color:#fff;
}
.topnav-name  { font-size:20px;font-weight:700;color:#fff;letter-spacing:-.3px; }
.topnav-tag   { font-size:11px;color:#64a6d8;margin-top:1px; }
.nav-badge    { background:#1a3a6a;border:1px solid #2a5a9a;color:#64a6d8;
                padding:4px 12px;border-radius:20px;font-size:11px;font-weight:500; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background:#0d1b2e !important;
  border-right:1px solid #1a3a5c !important;
}
section[data-testid="stSidebar"] * { color:#c8d8e8 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color:#ffffff !important; }

/* ── Sidebar buttons — FULL BRIGHTNESS ── */
section[data-testid="stSidebar"] .stButton > button {
  border-radius:8px !important;
  font-weight:700 !important;
  font-size:13px !important;
  letter-spacing:.3px !important;
  width:100% !important;
  padding:10px 16px !important;
  border:none !important;
  opacity:1 !important;
  cursor:pointer !important;
  transition:filter .2s, transform .1s !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
  filter:brightness(1.15) !important;
  transform:translateY(-1px) !important;
}
/* primary = How to Save Money */
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background:linear-gradient(135deg,#f5a623,#e08e0b) !important;
  color:#0d1b2e !important;
  box-shadow:0 4px 14px rgba(245,166,35,.35) !important;
}
/* secondary buttons */
section[data-testid="stSidebar"] .stButton > button[kind="secondary"],
section[data-testid="stSidebar"] .stButton > button:not([kind="primary"]) {
  background:linear-gradient(135deg,#1a5c96,#0f3d6b) !important;
  color:#ffffff !important;
  box-shadow:0 4px 14px rgba(26,92,150,.3) !important;
}
/* clear button – red tint */
section[data-testid="stSidebar"] .stButton:last-of-type > button {
  background:linear-gradient(135deg,#7b1a1a,#5a1010) !important;
  color:#ffcccc !important;
  box-shadow:0 4px 14px rgba(180,30,30,.3) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background:#0d1b2e;border-radius:12px;padding:4px;gap:4px;
  border:1px solid #1a3a6a;
}
.stTabs [data-baseweb="tab"] {
  background:transparent !important;border-radius:8px !important;
  color:#8aadcc !important;font-weight:600 !important;
  font-size:13px !important;padding:10px 22px !important;
}
.stTabs [aria-selected="true"] {
  background:linear-gradient(135deg,#1a5c96,#0f3d6b) !important;
  color:#ffffff !important;
  box-shadow:0 2px 10px rgba(26,92,150,.4) !important;
}

/* ── Cards ── */
.metric-card {
  background:linear-gradient(135deg,#0d1f35,#0f2848);
  border:1px solid #1a3a6a;border-radius:12px;
  padding:20px 24px;text-align:center;
  transition:transform .2s,box-shadow .2s;
}
.metric-card:hover{transform:translateY(-2px);box-shadow:0 8px 25px rgba(0,100,200,.15);}
.metric-label{font-size:11px;font-weight:600;color:#64a6d8;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;}
.metric-value{font-size:28px;font-weight:700;color:#fff;letter-spacing:-.5px;}
.delta-up  {font-size:12px;color:#2ecc71;margin-top:4px;}
.delta-down{font-size:12px;color:#e74c3c;margin-top:4px;}
.delta-neu {font-size:12px;color:#64a6d8;margin-top:4px;}

/* ── Section header ── */
.sec-hdr{display:flex;align-items:center;gap:10px;
  border-bottom:2px solid #1a3a6a;padding-bottom:10px;margin:24px 0 16px 0;}
.sec-icon{width:32px;height:32px;background:linear-gradient(135deg,#1a5c96,#00a89c);
  border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:15px;}
.sec-title{font-size:16px;font-weight:700;color:#fff;}
.sec-sub  {font-size:12px;color:#64a6d8;margin-left:4px;}

/* ── Info boxes ── */
.info-box    {background:#0a1929;border-left:4px solid #1a5c96;border-radius:0 8px 8px 0;padding:12px 16px;margin:8px 0;font-size:13px;color:#c8d8e8;}
.warning-box {background:#1a1200;border-left:4px solid #f5a623;border-radius:0 8px 8px 0;padding:12px 16px;margin:8px 0;font-size:13px;color:#f5d48a;}
.success-box {background:#041a0e;border-left:4px solid #27ae60;border-radius:0 8px 8px 0;padding:12px 16px;margin:8px 0;font-size:13px;color:#7de8a8;}

/* ── Savings tab cards ── */
.save-card {
  background:linear-gradient(135deg,#0a1f0a,#0d2b10);
  border:1px solid #1a6a2a;border-radius:14px;
  padding:22px 26px;margin-bottom:14px;
}
.save-card-title{font-size:16px;font-weight:700;color:#2ecc71;margin-bottom:6px;}
.save-amount{font-size:26px;font-weight:800;color:#27ae60;letter-spacing:-.5px;}
.save-label {font-size:11px;color:#7de8a8;text-transform:uppercase;letter-spacing:.8px;}
.save-rule  {font-size:12px;color:#a8d8b0;margin-top:8px;font-style:italic;}
.save-steps {font-size:13px;color:#c8e8d0;margin-top:10px;line-height:1.7;}

.risk-critical{background:#1a0505;border:1px solid #8b1a1a;border-radius:10px;padding:14px 18px;margin-bottom:10px;}
.risk-high    {background:#1a0e00;border:1px solid #8b5a00;border-radius:10px;padding:14px 18px;margin-bottom:10px;}
.risk-medium  {background:#0e0e1a;border:1px solid #3a3a8b;border-radius:10px;padding:14px 18px;margin-bottom:10px;}

/* ── Score badge ── */
.score-badge{display:inline-block;padding:6px 20px;border-radius:30px;font-size:22px;font-weight:800;}
.score-green {background:rgba(39,174,96,.15);color:#2ecc71;border:1px solid rgba(39,174,96,.4);}
.score-orange{background:rgba(230,126,34,.15);color:#f39c12;border:1px solid rgba(230,126,34,.4);}
.score-red   {background:rgba(192,57,43,.15);color:#e74c3c;border:1px solid rgba(192,57,43,.4);}

/* ── Chat messages — bright text ── */
.stChatMessage {
  background:#0d1b2e !important;
  border:1px solid #1a3a6a !important;
  border-radius:12px !important;
}
.stChatMessage p,
.stChatMessage li,
.stChatMessage span,
.stChatMessage div {
  color:#e8f0f8 !important;
  opacity:1 !important;
}
.stChatMessage [data-testid="stMarkdownContainer"] p {
  color:#e8f0f8 !important;
  font-size:14px !important;
  line-height:1.75 !important;
}
/* user bubble */
[data-testid="stChatMessageContent"] {
  color:#e8f0f8 !important;
}
hr{border-color:#1a3a6a !important;}
</style>
""", unsafe_allow_html=True)

# ── Top nav ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topnav">
  <div style="display:flex;align-items:center;gap:12px">
    <div class="topnav-logo">T</div>
    <div>
      <div class="topnav-name">TaxAI Pro</div>
      <div class="topnav-tag">Tax Compliance &amp; Advisory Platform</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:12px">
    <span class="nav-badge">Powered by Claude AI</span>
    <span class="nav-badge">871 IRS Regulation Chunks</span>
    <span class="nav-badge">Tax Year 2023 – 2024</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
def _extract_score(reply):
    for line in reply.splitlines():
        if "COMPLIANCE SCORE:" in line:
            try:
                s = line.split("COMPLIANCE SCORE:")[1].split("/")[0].strip()
                st.session_state.compliance_score = int("".join(filter(str.isdigit, s)))
            except Exception:
                pass

def _extract_savings(reply):
    for line in reply.splitlines():
        if "TOTAL POTENTIAL SAVINGS" in line.upper():
            m = re.search(r"\$[\d,]+", line)
            if m:
                st.session_state.total_savings = m.group(0)

def _ingest_tx(files, year):
    dfs = []
    col = USER_COL if year == "2024" else PRIOR_COL
    for uf in files:
        dest = USER_UPLOADS_DIR / uf.name
        dest.write_bytes(uf.read())
        ingest_file(str(dest), col)
        try:
            dfs.append(parse_transactions(str(dest)))
        except Exception:
            pass
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        key = "tx_summary_24" if year == "2024" else "tx_summary_23"
        st.session_state[key] = summarize_transactions(combined)

def _ingest_docs(files, year):
    col = USER_COL if year == "2024" else PRIOR_COL
    for uf in files:
        dest = USER_UPLOADS_DIR / uf.name
        dest.write_bytes(uf.read())
        ingest_file(str(dest), col)

def _metrics_row(s, label):
    cols = st.columns(4)
    data = [
        ("Income Found",      f"${s['total_income']:,.0f}",      label,       "delta-neu"),
        ("Tax Deductible",    f"${s['total_deductible']:,.0f}",   "Identified","delta-up"),
        ("Business Expenses", f"${s['business_expenses']:,.0f}",  "Schedule C","delta-neu"),
        ("Transactions",      f"{s['total_transactions']:,}",     "Analyzed",  "delta-neu"),
    ]
    for col, (lbl, val, sub, cls) in zip(cols, data):
        col.markdown(f"""<div class="metric-card">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value">{val}</div>
            <div class="{cls}">{sub}</div>
        </div>""", unsafe_allow_html=True)

def _bg_savings_worker(sid: str, history: list, user_col: str, prior_col: str) -> None:
    try:
        reply, updated_history = savings_recommendations(
            history, user_col=user_col, prior_col=prior_col)
        with _bg_lock:
            _bg_results[sid] = {"status": "done", "reply": reply, "history": updated_history}
    except Exception as exc:
        with _bg_lock:
            _bg_results[sid] = {"status": "error", "reply": str(exc), "history": history}


def _bar_chart(s, title, colorscale):
    cats = list(s["deductible_by_category"].keys())
    vals = list(s["deductible_by_category"].values())
    fig = go.Figure(go.Bar(
        x=vals, y=cats, orientation="h",
        marker=dict(color=vals, colorscale=colorscale, showscale=False),
        text=[f"${v:,.0f}" for v in vals],
        textposition="outside",
        textfont=dict(color="#c8d8e8", size=11),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#fff", size=14), x=0),
        paper_bgcolor="#07090f", plot_bgcolor="#0d1b2e",
        xaxis=dict(showgrid=True, gridcolor="#1a3a6a", tickfont=dict(color="#8aadcc")),
        yaxis=dict(showgrid=False, tickfont=dict(color="#c8d8e8")),
        margin=dict(l=0, r=70, t=40, b=20), height=360,
    )
    return fig

# ── Session state ──────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:10]

_sid = st.session_state.session_id
USER_COL  = f"user_docs_{_sid}"
PRIOR_COL = f"prior_docs_{_sid}"

for k, v in [("history",[]),("compliance_score",None),("total_savings",None),
             ("tx_summary_24",None),("tx_summary_23",None),
             ("savings_report",None),("analysis_report",None),
             ("bg_analysis_running",False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Poll for background analysis result (runs on every rerun) ─────────────────
if st.session_state.get("bg_analysis_running"):
    result = _bg_results.get(_sid)
    if result is not None:
        if result["status"] == "done":
            st.session_state.savings_report = result["reply"]
            st.session_state.history = result["history"]
            _extract_savings(result["reply"])
        st.session_state.bg_analysis_running = False
        with _bg_lock:
            _bg_results.pop(_sid, None)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Aria — AI Tax Advisor")
    st.caption("Upload your data in the year tabs, then run analysis below.")
    st.divider()

    stats = collection_stats(user_col=USER_COL, prior_col=PRIOR_COL)
    st.markdown("**Knowledge Base**")
    c1, c2, c3 = st.columns(3)
    c1.metric("IRS",   stats["irs_regulations"],    "chunks")
    c2.metric("2024",  stats["user_documents"],     "chunks")
    c3.metric("2023",  stats["prior_year_returns"],  "chunks")

    st.divider()
    st.markdown("**▶ Run Analysis**")

    # ── Background status badge ──────────────────────────────────────────────
    if st.session_state.get("bg_analysis_running"):
        st.markdown(
            '<div style="background:#1a2a00;border:1px solid #3a6a00;border-radius:8px;'
            'padding:8px 12px;font-size:12px;color:#a8e060;text-align:center;margin-bottom:8px">'
            '⚙️ Aria is analyzing in background…</div>',
            unsafe_allow_html=True)
    elif st.session_state.savings_report:
        st.markdown(
            '<div style="background:#041a0e;border:1px solid #27ae60;border-radius:8px;'
            'padding:8px 12px;font-size:12px;color:#7de8a8;text-align:center;margin-bottom:8px">'
            '✅ Savings report ready — see 💰 tab</div>',
            unsafe_allow_html=True)

    if st.button("💰  How to Save Money", use_container_width=True, type="primary"):
        with st.spinner("Finding every saving opportunity..."):
            reply, st.session_state.history = savings_recommendations(
                st.session_state.history, user_col=USER_COL, prior_col=PRIOR_COL)
            st.session_state.savings_report = reply
            _extract_savings(reply)
        st.success("Done! Open the 💰 Savings Report tab to see your results.")
        st.rerun()

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    if st.button("📋  Full Tax Analysis", use_container_width=True):
        with st.spinner("Running full compliance analysis..."):
            reply, st.session_state.history = analyze_documents(
                st.session_state.history, user_col=USER_COL)
            st.session_state.analysis_report = reply
            _extract_score(reply)
        st.rerun()

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    if st.button("🔍  Analyze Transactions", use_container_width=True):
        with st.spinner("Scanning bank transactions..."):
            reply, st.session_state.history = analyze_transactions(
                st.session_state.history, user_col=USER_COL)
            st.session_state.analysis_report = reply
            _extract_score(reply)
        st.rerun()

    st.divider()

    if st.button("🗑️  Clear Session", use_container_width=True):
        delete_collection(USER_COL)
        delete_collection(PRIOR_COL)
        old_sid = _sid
        for k in ["history","compliance_score","total_savings",
                  "tx_summary_24","tx_summary_23","savings_report","analysis_report",
                  "bg_analysis_running"]:
            st.session_state[k] = [] if k == "history" else None
        del st.session_state["session_id"]
        with _bg_lock:
            _bg_results.pop(old_sid, None)
        st.rerun()

    # score + savings badges
    if st.session_state.compliance_score is not None:
        score = st.session_state.compliance_score
        cls = "score-green" if score >= 75 else "score-orange" if score >= 50 else "score-red"
        st.markdown(f"""<div style="text-align:center;margin-top:16px">
            <div style="font-size:11px;color:#64a6d8;margin-bottom:6px">COMPLIANCE SCORE</div>
            <div class="score-badge {cls}">{score} / 100</div>
        </div>""", unsafe_allow_html=True)

    if st.session_state.total_savings:
        st.markdown(f"""<div style="text-align:center;margin-top:12px">
            <div style="font-size:11px;color:#64a6d8;margin-bottom:6px">POTENTIAL SAVINGS / YEAR</div>
            <div class="score-badge score-green">{st.session_state.total_savings}</div>
        </div>""", unsafe_allow_html=True)

# ── Main tabs ──────────────────────────────────────────────────────────────────
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

tab_24, tab_23, tab_compare, tab_savings, tab_chat = st.tabs([
    "📅  Tax Year 2024",
    "📅  Tax Year 2023",
    "📊  Year-over-Year Comparison",
    "💰  Savings Report",
    "💬  Chat with Aria",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — 2024
# ══════════════════════════════════════════════════════════════════════════════
with tab_24:
    st.markdown("""<div class="sec-hdr">
        <div class="sec-icon">📁</div>
        <span class="sec-title">Upload 2024 Financial Data</span>
        <span class="sec-sub">— Bank statements + tax documents</span>
    </div>""", unsafe_allow_html=True)

    col_tx, col_docs = st.columns(2)
    with col_tx:
        st.markdown("##### 🏦 Bank & Transaction Files")
        st.markdown('<div class="info-box">CSV or Excel exports from your bank, credit cards, or brokerage.</div>', unsafe_allow_html=True)
        tx24 = st.file_uploader("2024 transactions", type=["csv","xlsx","xls"],
                                 accept_multiple_files=True, label_visibility="collapsed", key="tx24")
        if tx24 and st.button("Ingest 2024 Transactions", type="primary", key="btn_tx24"):
            with st.spinner("Parsing..."):
                _ingest_tx(tx24, "2024")
            st.markdown('<div class="success-box">✅ Transactions ingested. Aria is analyzing in background…</div>', unsafe_allow_html=True)
            if not st.session_state.get("bg_analysis_running"):
                st.session_state.bg_analysis_running = True
                threading.Thread(
                    target=_bg_savings_worker,
                    args=(_sid, list(st.session_state.history), USER_COL, PRIOR_COL),
                    daemon=True,
                ).start()
            st.rerun()

    with col_docs:
        st.markdown("##### 📄 Tax Documents")
        st.markdown('<div class="info-box">W-2s, 1099s, tax profile TXT, or any PDF tax documents.</div>', unsafe_allow_html=True)
        docs24 = st.file_uploader("2024 docs", type=["pdf","txt","md"],
                                   accept_multiple_files=True, label_visibility="collapsed", key="docs24")
        if docs24 and st.button("Ingest 2024 Documents", type="primary", key="btn_docs24"):
            with st.spinner("Processing..."):
                _ingest_docs(docs24, "2024")
            st.markdown('<div class="success-box">✅ Documents ingested. Aria is analyzing in background…</div>', unsafe_allow_html=True)
            if not st.session_state.get("bg_analysis_running"):
                st.session_state.bg_analysis_running = True
                threading.Thread(
                    target=_bg_savings_worker,
                    args=(_sid, list(st.session_state.history), USER_COL, PRIOR_COL),
                    daemon=True,
                ).start()

    if st.session_state.tx_summary_24:
        st.markdown("""<div class="sec-hdr" style="margin-top:28px">
            <div class="sec-icon">📈</div>
            <span class="sec-title">2024 Transaction Summary</span>
        </div>""", unsafe_allow_html=True)
        _metrics_row(st.session_state.tx_summary_24, "Tax Year 2024")
        s24 = st.session_state.tx_summary_24
        if s24["deductible_by_category"]:
            st.markdown("<br>", unsafe_allow_html=True)
            col_c, col_t = st.columns([3,2])
            col_c.plotly_chart(_bar_chart(s24, "2024 — Deductible Expenses by Category",
                               [[0,"#1a3a6a"],[0.5,"#1a5c96"],[1,"#00a89c"]]),
                               use_container_width=True)
            col_t.dataframe(
                pd.DataFrame([{"Category":k,"Amount":f"${v:,.2f}"}
                              for k,v in s24["deductible_by_category"].items()]),
                use_container_width=True, hide_index=True, height=360)

    # Analysis report in this tab
    if st.session_state.analysis_report:
        st.markdown("""<div class="sec-hdr" style="margin-top:28px">
            <div class="sec-icon">📋</div>
            <span class="sec-title">Full Tax Analysis Result</span>
        </div>""", unsafe_allow_html=True)
        st.markdown(
            f"""<div style="background:#0a0f1a;border:1px solid #1a3a6a;
                           border-radius:14px;padding:28px 32px;
                           line-height:1.8;font-size:14px;color:#c8d8e8;">
            {st.session_state.analysis_report.replace(chr(10),'<br>')}
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — 2023
# ══════════════════════════════════════════════════════════════════════════════
with tab_23:
    st.markdown("""<div class="sec-hdr">
        <div class="sec-icon">📁</div>
        <span class="sec-title">Upload 2023 Financial Data</span>
        <span class="sec-sub">— Prior year bank statements + filed tax returns</span>
    </div>""", unsafe_allow_html=True)

    col_tx, col_docs = st.columns(2)
    with col_tx:
        st.markdown("##### 🏦 Bank & Transaction Files")
        st.markdown('<div class="info-box">2023 CSV bank exports for year-over-year comparison.</div>', unsafe_allow_html=True)
        tx23 = st.file_uploader("2023 transactions", type=["csv","xlsx","xls"],
                                 accept_multiple_files=True, label_visibility="collapsed", key="tx23")
        if tx23 and st.button("Ingest 2023 Transactions", type="primary", key="btn_tx23"):
            with st.spinner("Parsing..."):
                _ingest_tx(tx23, "2023")
            st.markdown('<div class="success-box">✅ 2023 transactions ingested. Aria is analyzing in background…</div>', unsafe_allow_html=True)
            if not st.session_state.get("bg_analysis_running"):
                st.session_state.bg_analysis_running = True
                threading.Thread(
                    target=_bg_savings_worker,
                    args=(_sid, list(st.session_state.history), USER_COL, PRIOR_COL),
                    daemon=True,
                ).start()
            st.rerun()

    with col_docs:
        st.markdown("##### 📄 Prior Year Tax Returns")
        st.markdown('<div class="info-box">Filed 2023 Form 1040, W-2s, or prior year tax profile TXT.</div>', unsafe_allow_html=True)
        docs23 = st.file_uploader("2023 docs", type=["pdf","txt","md"],
                                   accept_multiple_files=True, label_visibility="collapsed", key="docs23")
        if docs23 and st.button("Ingest 2023 Documents", type="primary", key="btn_docs23"):
            with st.spinner("Processing..."):
                _ingest_docs(docs23, "2023")
            st.markdown('<div class="success-box">✅ 2023 documents ingested. Aria is analyzing in background…</div>', unsafe_allow_html=True)
            if not st.session_state.get("bg_analysis_running"):
                st.session_state.bg_analysis_running = True
                threading.Thread(
                    target=_bg_savings_worker,
                    args=(_sid, list(st.session_state.history), USER_COL, PRIOR_COL),
                    daemon=True,
                ).start()

    if st.session_state.tx_summary_23:
        st.markdown("""<div class="sec-hdr" style="margin-top:28px">
            <div class="sec-icon">📈</div>
            <span class="sec-title">2023 Transaction Summary</span>
        </div>""", unsafe_allow_html=True)
        _metrics_row(st.session_state.tx_summary_23, "Tax Year 2023")
        s23 = st.session_state.tx_summary_23
        if s23["deductible_by_category"]:
            st.markdown("<br>", unsafe_allow_html=True)
            col_c, col_t = st.columns([3,2])
            col_c.plotly_chart(_bar_chart(s23, "2023 — Deductible Expenses by Category",
                               [[0,"#2d1a4a"],[0.5,"#5c1a96"],[1,"#a000c8"]]),
                               use_container_width=True)
            col_t.dataframe(
                pd.DataFrame([{"Category":k,"Amount":f"${v:,.2f}"}
                              for k,v in s23["deductible_by_category"].items()]),
                use_container_width=True, hide_index=True, height=360)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    s24 = st.session_state.tx_summary_24
    s23 = st.session_state.tx_summary_23

    if not s24 and not s23:
        st.markdown('<div class="warning-box">📊 Upload bank statements in both year tabs to unlock comparison charts.</div>', unsafe_allow_html=True)
    else:
        st.markdown("""<div class="sec-hdr">
            <div class="sec-icon">📊</div>
            <span class="sec-title">Year-over-Year Overview</span>
        </div>""", unsafe_allow_html=True)

        inc24 = s24["total_income"]      if s24 else 0
        inc23 = s23["total_income"]      if s23 else 0
        ded24 = s24["total_deductible"]  if s24 else 0
        ded23 = s23["total_deductible"]  if s23 else 0
        biz24 = s24["business_expenses"] if s24 else 0
        biz23 = s23["business_expenses"] if s23 else 0

        def _delta(v4, v3):
            if v3 == 0: return '<div class="delta-neu">No prior data</div>'
            p = (v4-v3)/v3*100
            c = "delta-up" if p>=0 else "delta-down"
            a = "▲" if p>=0 else "▼"
            return f'<div class="{c}">{a} {abs(p):.1f}% vs 2023</div>'

        kc = st.columns(4)
        for col, lbl, v4, v3, fmt in [
            (kc[0],"Total Income",      inc24,inc23,"${:,.0f}"),
            (kc[1],"Tax Deductible",    ded24,ded23,"${:,.0f}"),
            (kc[2],"Business Expenses", biz24,biz23,"${:,.0f}"),
            (kc[3],"Transactions",      (s24["total_transactions"] if s24 else 0),
                                        (s23["total_transactions"] if s23 else 0),"{:,}"),
        ]:
            col.markdown(f"""<div class="metric-card">
                <div class="metric-label">{lbl}</div>
                <div class="metric-value">{fmt.format(v4)}</div>
                {_delta(v4,v3)}
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            for name, vals, color in [
                ("2023",[inc23,ded23,biz23],"#7c3aed"),
                ("2024",[inc24,ded24,biz24],"#00a89c"),
            ]:
                fig.add_trace(go.Bar(
                    name=name, x=["Total Income","Tax Deductible","Business Expenses"], y=vals,
                    marker_color=color,
                    text=[f"${v:,.0f}" for v in vals],
                    textposition="outside", textfont=dict(color="#c8d8e8",size=10),
                ))
            fig.update_layout(
                title=dict(text="Income & Deductions — 2023 vs 2024",font=dict(color="#fff",size=14),x=0),
                barmode="group", paper_bgcolor="#07090f", plot_bgcolor="#0d1b2e",
                xaxis=dict(tickfont=dict(color="#c8d8e8"),showgrid=False),
                yaxis=dict(tickfont=dict(color="#8aadcc"),gridcolor="#1a3a6a",tickprefix="$",tickformat=",.0f"),
                legend=dict(font=dict(color="#c8d8e8"),bgcolor="#0d1b2e",bordercolor="#1a3a6a",borderwidth=1),
                margin=dict(l=0,r=20,t=40,b=20), height=340,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if s24:
                tc = s24.get("transaction_types",{})
                colors = ["#00a89c","#1a5c96","#f5a623","#7c3aed","#e74c3c","#2ecc71"]
                fig2 = go.Figure(go.Pie(
                    labels=list(tc.keys()), values=list(tc.values()), hole=0.55,
                    marker=dict(colors=colors[:len(tc)],line=dict(color="#07090f",width=2)),
                    textfont=dict(color="#fff",size=11),
                ))
                fig2.update_layout(
                    title=dict(text="2024 Transaction Types",font=dict(color="#fff",size=14),x=0),
                    paper_bgcolor="#07090f",
                    legend=dict(font=dict(color="#c8d8e8"),bgcolor="#0d1b2e",bordercolor="#1a3a6a",borderwidth=1),
                    margin=dict(l=0,r=0,t=40,b=20), height=340,
                    annotations=[dict(text="2024",x=0.5,y=0.5,font=dict(size=18,color="white"),showarrow=False)],
                )
                st.plotly_chart(fig2, use_container_width=True)

        if s24 and s23:
            st.markdown("""<div class="sec-hdr">
                <div class="sec-icon">🔍</div>
                <span class="sec-title">Deductible Expenses — Category Comparison</span>
            </div>""", unsafe_allow_html=True)

            all_cats = sorted(set(list(s24["deductible_by_category"]) + list(s23["deductible_by_category"])))
            v23c = [s23["deductible_by_category"].get(c,0) for c in all_cats]
            v24c = [s24["deductible_by_category"].get(c,0) for c in all_cats]
            changes = [a-b for a,b in zip(v24c,v23c)]

            col_a, col_b = st.columns([3,2])
            with col_a:
                fig3 = go.Figure()
                for name, vals, color in [("2023",v23c,"#7c3aed"),("2024",v24c,"#00a89c")]:
                    fig3.add_trace(go.Bar(name=name,y=all_cats,x=vals,orientation="h",
                                         marker_color=color,opacity=0.85))
                fig3.update_layout(
                    title=dict(text="Deductions by Category — 2023 vs 2024",font=dict(color="#fff",size=14),x=0),
                    barmode="group", paper_bgcolor="#07090f", plot_bgcolor="#0d1b2e",
                    xaxis=dict(tickfont=dict(color="#8aadcc"),gridcolor="#1a3a6a",tickprefix="$",tickformat=",.0f"),
                    yaxis=dict(tickfont=dict(color="#c8d8e8"),showgrid=False),
                    legend=dict(font=dict(color="#c8d8e8"),bgcolor="#0d1b2e",bordercolor="#1a3a6a",borderwidth=1),
                    margin=dict(l=0,r=20,t=40,b=20), height=420,
                )
                st.plotly_chart(fig3, use_container_width=True)

            with col_b:
                bar_colors = ["#2ecc71" if c>=0 else "#e74c3c" for c in changes]
                fig4 = go.Figure(go.Bar(
                    y=all_cats, x=changes, orientation="h",
                    marker_color=bar_colors,
                    text=[f"${abs(c):,.0f}" for c in changes],
                    textposition="outside", textfont=dict(color="#c8d8e8",size=10),
                ))
                fig4.update_layout(
                    title=dict(text="Change: 2023 → 2024",font=dict(color="#fff",size=14),x=0),
                    paper_bgcolor="#07090f", plot_bgcolor="#0d1b2e",
                    xaxis=dict(tickfont=dict(color="#8aadcc"),gridcolor="#1a3a6a",
                               tickprefix="$",tickformat=",.0f",zeroline=True,zerolinecolor="#3a5a7a"),
                    yaxis=dict(tickfont=dict(color="#c8d8e8"),showgrid=False),
                    margin=dict(l=0,r=70,t=40,b=20), height=420,
                )
                st.plotly_chart(fig4, use_container_width=True)

            st.markdown("""<div class="sec-hdr">
                <div class="sec-icon">📋</div>
                <span class="sec-title">Detailed Change Table</span>
            </div>""", unsafe_allow_html=True)
            rows = []
            for cat,v3,v4 in zip(all_cats,v23c,v24c):
                chg = v4-v3
                pct = (chg/v3*100) if v3>0 else 0
                rows.append({"Category":cat,"2023":f"${v3:,.2f}","2024":f"${v4:,.2f}",
                             "Change":f"+${chg:,.2f}" if chg>=0 else f"-${abs(chg):,.2f}",
                             "% Change":f"+{pct:.1f}%" if pct>=0 else f"{pct:.1f}%"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SAVINGS REPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab_savings:
    if not st.session_state.savings_report:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;margin-top:20px">
            <div style="font-size:48px;margin-bottom:16px">💰</div>
            <div style="font-size:22px;font-weight:700;color:#fff;margin-bottom:12px">
                No Savings Report Yet
            </div>
            <div style="font-size:14px;color:#8aadcc;max-width:440px;margin:0 auto;line-height:1.7">
                Click <strong style="color:#f5a623">How to Save Money</strong> in the sidebar
                to generate your personalized tax savings recommendations — specific dos &amp; don'ts,
                dollar estimates, and year-over-year changes.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        report = st.session_state.savings_report

        # ── Hero banner ──
        total_m = re.search(r"TOTAL POTENTIAL SAVINGS[:\s\*]+\$?([\d,]+)", report, re.IGNORECASE)
        total_val = f"${total_m.group(1)}" if total_m else "See report below"
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#062010,#0a2a15);
                    border:2px solid #27ae60;border-radius:16px;
                    padding:28px 40px;text-align:center;margin-bottom:28px;
                    box-shadow:0 4px 30px rgba(39,174,96,.2)">
            <div style="font-size:11px;font-weight:700;color:#7de8a8;
                        text-transform:uppercase;letter-spacing:3px;margin-bottom:8px">
                Total Potential Tax Savings Per Year
            </div>
            <div style="font-size:58px;font-weight:900;color:#2ecc71;
                        letter-spacing:-2px;line-height:1.1">
                {total_val}
            </div>
            <div style="font-size:13px;color:#a8d8b0;margin-top:10px">
                Based on your uploaded financial data &amp; IRS regulations
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Parse into sections ──
        def _section(text, start_marker, *end_markers):
            """Extract text between start_marker and the first matching end_marker."""
            i = text.find(start_marker)
            if i == -1:
                return ""
            i += len(start_marker)
            end = len(text)
            for em in end_markers:
                j = text.find(em, i)
                if j != -1 and j < end:
                    end = j
            return text[i:end].strip()

        # Dos — What to do lines
        todos = re.findall(r'- What to do:\s*(.+)', report)
        # Don'ts / risks implied by recommendation context
        deadlines = re.findall(r'- Deadline:\s*(.+)', report)
        rec_titles = re.findall(r'\*\*RECOMMENDATION \d+:\s*(.+?)\*\*', report)
        savings_per_rec = re.findall(r'- Estimated Annual Savings:\s*(.+)', report)

        # ── Dos & Don'ts card ──
        if todos or rec_titles:
            st.markdown("""<div class="sec-hdr">
                <div class="sec-icon">✅</div>
                <span class="sec-title">What To Do — Action Items</span>
                <span class="sec-sub">— Ranked by savings</span>
            </div>""", unsafe_allow_html=True)

            col_do, col_dont = st.columns(2)
            with col_do:
                st.markdown("""<div style="background:linear-gradient(135deg,#041a0e,#062010);
                    border:1px solid #1a6a2a;border-radius:12px;padding:20px 22px;height:100%">
                    <div style="font-size:13px;font-weight:700;color:#2ecc71;margin-bottom:14px;
                                text-transform:uppercase;letter-spacing:1px">✅ DO These Now</div>""",
                    unsafe_allow_html=True)
                for i, (title, action) in enumerate(zip(rec_titles, todos)):
                    savings_str = savings_per_rec[i] if i < len(savings_per_rec) else ""
                    st.markdown(f"""<div style="border-bottom:1px solid #1a4a2a;padding:10px 0;margin-bottom:6px">
                        <div style="font-size:13px;font-weight:600;color:#7de8a8">{title.strip()}</div>
                        <div style="font-size:12px;color:#c8e8d0;margin-top:4px">{action.strip()}</div>
                        {f'<div style="font-size:11px;color:#2ecc71;margin-top:4px;font-weight:600">{savings_str.strip()}</div>' if savings_str else ''}
                    </div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_dont:
                st.markdown("""<div style="background:linear-gradient(135deg,#1a0a00,#2a1200);
                    border:1px solid #6a3a1a;border-radius:12px;padding:20px 22px;height:100%">
                    <div style="font-size:13px;font-weight:700;color:#f5a623;margin-bottom:14px;
                                text-transform:uppercase;letter-spacing:1px">⚠️ Act Before These Deadlines</div>""",
                    unsafe_allow_html=True)
                for i, title in enumerate(rec_titles):
                    deadline = deadlines[i] if i < len(deadlines) else "Before tax filing"
                    st.markdown(f"""<div style="border-bottom:1px solid #4a2a0a;padding:10px 0;margin-bottom:6px">
                        <div style="font-size:13px;font-weight:600;color:#f5d48a">{title.strip()}</div>
                        <div style="font-size:12px;color:#e8d0a8;margin-top:4px">🗓 {deadline.strip()}</div>
                    </div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # ── Year-over-Year Changes ──
        yoy_raw = _section(report, "YEAR-OVER-YEAR ANALYSIS", "TOTAL POTENTIAL SAVINGS", "---\n\n**TOTAL")
        if yoy_raw:
            st.markdown("""<div class="sec-hdr" style="margin-top:28px">
                <div class="sec-icon">📅</div>
                <span class="sec-title">Year-over-Year Changes</span>
                <span class="sec-sub">— What changed and what to do differently</span>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"""<div style="background:#0a1020;border:1px solid #2a4a6a;
                border-radius:12px;padding:24px 28px;line-height:1.85;font-size:14px;color:#c8d8e8;">
                {yoy_raw.replace(chr(10),'<br>')}
            </div>""", unsafe_allow_html=True)

        # ── Full detailed report (collapsible) ──
        with st.expander("📄 Full Detailed Report", expanded=False):
            st.markdown(report)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Savings Report", key="refresh_savings_tab"):
            with st.spinner("Finding savings opportunities..."):
                reply, st.session_state.history = savings_recommendations(
                    st.session_state.history, user_col=USER_COL, prior_col=PRIOR_COL)
                st.session_state.savings_report = reply
                _extract_savings(reply)
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("""<div class="sec-hdr">
        <div class="sec-icon">🤖</div>
        <span class="sec-title">Chat with Aria</span>
        <span class="sec-sub">— Ask anything about your taxes</span>
    </div>""", unsafe_allow_html=True)

    # ── Compact key actions quick-reference (only if savings report exists) ──
    if st.session_state.savings_report:
        report = st.session_state.savings_report
        todos = re.findall(r'- What to do:\s*(.+)', report)
        rec_titles = re.findall(r'\*\*RECOMMENDATION \d+:\s*(.+?)\*\*', report)
        savings_per_rec = re.findall(r'- Estimated Annual Savings:\s*(.+)', report)
        if rec_titles:
            with st.expander("💡 Key Tax Actions — Quick Reference", expanded=True):
                for i, title in enumerate(rec_titles[:6]):
                    action = todos[i].strip() if i < len(todos) else ""
                    savings_str = savings_per_rec[i].strip() if i < len(savings_per_rec) else ""
                    badge = f'<span style="color:#2ecc71;font-weight:700;margin-left:8px">{savings_str}</span>' if savings_str else ""
                    action_html = f'<br><span style="color:#8aadcc;font-size:12px">{action}</span>' if action else ""
                    st.markdown(
                        f'<div style="padding:7px 0;border-bottom:1px solid #1a3a4a;font-size:13px;color:#c8d8e8">'
                        f'<span style="color:#7de8a8;font-weight:600">✅ {title.strip()}</span>{badge}'
                        f'{action_html}</div>',
                        unsafe_allow_html=True
                    )

    # ── Chat history ──
    for msg in st.session_state.history:
        role = msg["role"]
        content = msg["content"]
        if "<retrieved_context>" in content:
            content = content.split("<user_question>")[1].replace("</user_question>","").strip()
        if "<current_year_data>" in content:
            content = content.split("<request>")[1].replace("</request>","").strip()
        with st.chat_message("user" if role=="user" else "assistant",
                             avatar="🧑" if role=="user" else "🤖"):
            st.markdown(content)

    user_input = st.chat_input("Ask Aria anything: 'What if I max my 401k?' · 'Should I elect S-Corp?' · 'What deductions am I missing?'")
    if user_input:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)
        with st.chat_message("assistant", avatar="🤖"):
            reply = st.write_stream(
                chat_stream(st.session_state.history, user_input, user_col=USER_COL)
            )
        rag_wrapped = build_user_message(user_input, user_col=USER_COL)
        st.session_state.history = st.session_state.history + [
            {"role": "user",      "content": rag_wrapped},
            {"role": "assistant", "content": reply},
        ]
        _extract_score(reply)
        _extract_savings(reply)
        st.rerun()
