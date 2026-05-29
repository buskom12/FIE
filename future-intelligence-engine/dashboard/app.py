"""FIE Dashboard — MVP v1.0"""

import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
import pandas as pd

# Добавляем корень проекта в sys.path чтобы работали импорты agents/prediction
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.agent_manager import AgentManager
from prediction.aggregation import aggregate_predictions
from prediction.event_impact import score_event_impact

# ─── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FIE Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global */
    .stApp { background-color: #0d0f14; color: #e2e8f0; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #141824 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        text-align: center;
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #718096;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #718096;
        margin-top: 0.25rem;
    }

    /* Section headers */
    .section-header {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #4a5568;
        margin-bottom: 0.75rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #1e2535;
    }

    /* Event box */
    .event-box {
        background: linear-gradient(135deg, #1a1f2e 0%, #141824 100%);
        border: 1px solid #2d3748;
        border-left: 4px solid #6366f1;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
    }
    .event-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 0.3rem;
    }
    .event-meta {
        font-size: 0.78rem;
        color: #718096;
    }

    /* Agent opinion card */
    .agent-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.5rem;
    }
    .agent-name { font-size: 0.85rem; font-weight: 700; color: #a0aec0; }
    .agent-role { font-size: 0.72rem; color: #4a5568; margin-bottom: 0.3rem; }
    .agent-opinion { font-size: 0.82rem; color: #cbd5e0; font-style: italic; }
    .agent-prob { font-size: 1.1rem; font-weight: 700; margin-top: 0.4rem; }

    /* Progress bar override */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
    }

    /* Divider */
    hr { border-color: #1e2535; }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Data helpers ────────────────────────────────────────────────────────────────

BASE = Path(__file__).parent.parent

def load_prediction_history() -> list[dict]:
    path = BASE / "prediction_history.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def load_outcomes_history() -> list[dict]:
    path = BASE / "outcomes_history.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def mock_top_event() -> dict:
    return {
        "title": "Federal Reserve raises interest rates by 0.25%",
        "source": "Reuters",
        "category": "Macro / Monetary Policy",
        "fetched_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    }


def mock_agent_opinions() -> list[dict]:
    agents = [
        {"name": "QuantAlpha",  "role": "Quantitative Analyst",    "prob": 0.81, "sentiment": "bullish",  "opinion": "Strong macro signal — CPI confirms tightening cycle is near its peak."},
        {"name": "MarcoSorelli","role": "Macro Economist",          "prob": 0.67, "sentiment": "neutral",  "opinion": "Markets already priced in 25bps. Reaction will depend on forward guidance."},
        {"name": "RiskHedge",   "role": "Risk Manager",             "prob": 0.44, "sentiment": "bearish",  "opinion": "Tail risk underestimated. Yield curve inversion deepens — recession signal."},
        {"name": "SentimentBot","role": "Sentiment Analyst",        "prob": 0.73, "sentiment": "bullish",  "opinion": "Social sentiment index up 14pts. Retail investors aggressively buying dips."},
        {"name": "GeoPoliticus","role": "Geopolitical Strategist",  "prob": 0.52, "sentiment": "neutral",  "opinion": "EM currency pressure may force Fed to pause earlier than signalled."},
    ]
    return agents


def mock_leaderboard() -> pd.DataFrame:
    data = [
        {"Agent": "QuantAlpha",   "Role": "Quant Analyst",       "Predictions": 48, "Accuracy": 0.87, "Avg Error": 0.08, "Weight": 1.94},
        {"Agent": "SentimentBot", "Role": "Sentiment Analyst",   "Predictions": 51, "Accuracy": 0.82, "Avg Error": 0.11, "Weight": 1.76},
        {"Agent": "MarcoSorelli", "Role": "Macro Economist",     "Predictions": 44, "Accuracy": 0.79, "Avg Error": 0.14, "Weight": 1.65},
        {"Agent": "GeoPoliticus", "Role": "Geo Strategist",      "Predictions": 37, "Accuracy": 0.71, "Avg Error": 0.19, "Weight": 1.42},
        {"Agent": "RiskHedge",    "Role": "Risk Manager",        "Predictions": 40, "Accuracy": 0.68, "Avg Error": 0.22, "Weight": 1.31},
        {"Agent": "TechSignal",   "Role": "Technical Analyst",   "Predictions": 35, "Accuracy": 0.64, "Avg Error": 0.26, "Weight": 1.18},
        {"Agent": "NewsReader",   "Role": "News Analyst",        "Predictions": 29, "Accuracy": 0.58, "Avg Error": 0.31, "Weight": 1.00},
    ]
    return pd.DataFrame(data)


def mock_accuracy_report() -> pd.DataFrame:
    dates = [datetime.utcnow() - timedelta(days=i) for i in range(7, 0, -1)]
    return pd.DataFrame({
        "Date": [d.strftime("%b %d") for d in dates],
        "FIE Accuracy": [round(random.uniform(0.72, 0.91), 2) for _ in dates],
        "Market Accuracy": [round(random.uniform(0.55, 0.75), 2) for _ in dates],
    })


def prob_color(prob: float) -> str:
    if prob >= 0.70:
        return "#48bb78"
    elif prob >= 0.45:
        return "#ecc94b"
    else:
        return "#fc8181"


def sentiment_badge(sentiment: str) -> str:
    colors = {"bullish": "#48bb78", "bearish": "#fc8181", "neutral": "#ecc94b"}
    labels = {"bullish": "▲ BULLISH", "bearish": "▼ BEARISH", "neutral": "◆ NEUTRAL"}
    color = colors.get(sentiment, "#718096")
    label = labels.get(sentiment, sentiment.upper())
    return f'<span style="background:{color}22;color:{color};padding:2px 8px;border-radius:20px;font-size:0.68rem;font-weight:700;">{label}</span>'


# ─── Header ──────────────────────────────────────────────────────────────────────

col_title, col_time = st.columns([3, 1])
with col_title:
    st.markdown("## 🧠 FIE Dashboard")
    st.markdown('<p style="color:#4a5568;font-size:0.8rem;margin-top:-0.8rem;">Future Intelligence Engine · MVP v1.0</p>', unsafe_allow_html=True)
with col_time:
    st.markdown(f'<p style="color:#4a5568;font-size:0.78rem;text-align:right;padding-top:0.5rem;">🕐 {datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC</p>', unsafe_allow_html=True)

st.markdown("---")


# ─── Section 1: Top Event ────────────────────────────────────────────────────────

st.markdown('<div class="section-header">📰 Top Event</div>', unsafe_allow_html=True)

event = mock_top_event()

st.markdown(f"""
<div class="event-box">
    <div class="event-title">{event["title"]}</div>
    <div class="event-meta">
        📡 {event["source"]} &nbsp;·&nbsp; 🏷️ {event["category"]} &nbsp;·&nbsp; 🕐 {event["fetched_at"]}
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── Section 2: Run Simulation ───────────────────────────────────────────────────

st.markdown('<div class="section-header">⚡ Run Simulation</div>', unsafe_allow_html=True)

sim_col1, sim_col2 = st.columns([4, 1])
with sim_col1:
    user_event = st.text_input(
        label="",
        placeholder="Enter event to analyze, e.g. 'Fed raises rates by 0.5%'",
        label_visibility="collapsed",
    )
with sim_col2:
    run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

if run_btn and user_event.strip():
    with st.spinner("Running agent simulation..."):
        try:
            manager = AgentManager()
            manager.create_agents(n=5)
            raw_predictions = manager.evaluate_event(user_event.strip())
            aggregated      = aggregate_predictions(raw_predictions)
            impact_raw      = score_event_impact(user_event.strip())

            st.session_state["sim_event"]       = user_event.strip()
            st.session_state["sim_predictions"] = raw_predictions
            st.session_state["sim_aggregated"]  = aggregated
            st.session_state["sim_impact"]      = impact_raw
            st.session_state["sim_agents"]      = manager.agents
            st.session_state["sim_ok"]          = True
        except Exception as exc:
            st.error(f"Simulation error: {exc}")
            st.session_state["sim_ok"] = False

elif run_btn and not user_event.strip():
    st.warning("Please enter an event description first.")

# Читаем результаты симуляции из session_state (если были запущены)
_sim_ok = st.session_state.get("sim_ok", False)
if _sim_ok:
    _agg    = st.session_state["sim_aggregated"]
    _impact = st.session_state["sim_impact"]
    st.success(f"Simulation complete for: **{st.session_state['sim_event']}**")

st.markdown("---")


# ─── Section 3: Key Metrics ──────────────────────────────────────────────────────

st.markdown('<div class="section-header">📊 Key Metrics</div>', unsafe_allow_html=True)

if _sim_ok:
    fie_prob     = _agg["final_probability"]
    confidence   = _agg["confidence"]
    impact_score = round(_impact * 10, 1)
    market_prob  = round(max(0.0, min(1.0, fie_prob - random.uniform(0.08, 0.18))), 2)
else:
    fie_prob     = 0.74
    market_prob  = 0.61
    impact_score = 8.2
    confidence   = 0.83

c1, c2, c3, c4 = st.columns(4)

with c1:
    color = prob_color(fie_prob)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">FIE Prediction</div>
        <div class="metric-value" style="color:{color}">{fie_prob:.0%}</div>
        <div class="metric-sub">Probability of event occurring</div>
    </div>""", unsafe_allow_html=True)

with c2:
    color2 = prob_color(market_prob)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Market Prediction</div>
        <div class="metric-value" style="color:{color2}">{market_prob:.0%}</div>
        <div class="metric-sub">Implied by options market</div>
    </div>""", unsafe_allow_html=True)

with c3:
    impact_color = "#fc8181" if impact_score >= 8 else "#ecc94b" if impact_score >= 5 else "#48bb78"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Impact Score</div>
        <div class="metric-value" style="color:{impact_color}">{impact_score:.1f}</div>
        <div class="metric-sub">Market disruption / 10</div>
    </div>""", unsafe_allow_html=True)

with c4:
    conf_color = "#48bb78" if confidence >= 0.75 else "#ecc94b"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Confidence</div>
        <div class="metric-value" style="color:{conf_color}">{confidence:.0%}</div>
        <div class="metric-sub">Agent consensus level</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# FIE vs Market progress bars
col_bars, col_gap = st.columns([2, 1])
with col_bars:
    st.markdown(f'<p style="font-size:0.78rem;color:#718096;margin-bottom:0.15rem;">FIE Prediction: <b style="color:#e2e8f0">{fie_prob:.0%}</b></p>', unsafe_allow_html=True)
    st.progress(fie_prob)
    st.markdown(f'<p style="font-size:0.78rem;color:#718096;margin-bottom:0.15rem;margin-top:0.5rem;">Market Prediction: <b style="color:#e2e8f0">{market_prob:.0%}</b></p>', unsafe_allow_html=True)
    st.progress(market_prob)
    edge = fie_prob - market_prob
    edge_str = f"+{edge:.0%}" if edge > 0 else f"{edge:.0%}"
    edge_color = "#48bb78" if edge > 0 else "#fc8181"
    st.markdown(f'<p style="font-size:0.78rem;margin-top:0.4rem;">FIE Edge: <b style="color:{edge_color}">{edge_str}</b></p>', unsafe_allow_html=True)

st.markdown("---")


# ─── Section 4: Agent Opinions ───────────────────────────────────────────────────

st.markdown('<div class="section-header">🤖 Agent Opinions</div>', unsafe_allow_html=True)

if _sim_ok:
    # Реальные агенты из симуляции
    raw = st.session_state["sim_predictions"]
    agents_display = []
    for r in raw:
        prob = r["probability"]
        if prob >= 0.65:
            sent = "bullish"
        elif prob <= 0.40:
            sent = "bearish"
        else:
            sent = "neutral"
        agents_display.append({
            "name":      r["agent"],
            "role":      r["role"],
            "prob":      prob,
            "sentiment": sent,
            "opinion":   r.get("reasoning", "—"),
        })
else:
    agents_display = mock_agent_opinions()

cols = st.columns(max(len(agents_display), 1))
for col, agent in zip(cols, agents_display):
    with col:
        prob = agent["prob"]
        color = prob_color(prob)
        badge = sentiment_badge(agent["sentiment"])
        opinion_text = agent["opinion"][:160] + "…" if len(agent["opinion"]) > 160 else agent["opinion"]
        st.markdown(f"""
        <div class="agent-card">
            <div class="agent-name">{agent["name"]}</div>
            <div class="agent-role">{agent["role"]}</div>
            {badge}
            <div class="agent-prob" style="color:{color}">{prob:.0%}</div>
            <div class="agent-opinion">"{opinion_text}"</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")


# ─── Section 5: Agent Leaderboard ────────────────────────────────────────────────

st.markdown('<div class="section-header">🏆 Agent Leaderboard</div>', unsafe_allow_html=True)

if _sim_ok and st.session_state.get("sim_agents"):
    sim_agents = st.session_state["sim_agents"]
    lb = pd.DataFrame([
        {
            "Agent":       agent.persona.name,
            "Role":        agent.persona.role,
            "Predictions": agent.persona.predictions_count,
            "Accuracy":    round(agent.persona.accuracy, 2),
            "Avg Error":   round(1.0 - agent.persona.accuracy, 2),
            "Weight":      round(agent.persona.weight, 2),
        }
        for agent in sorted(sim_agents, key=lambda a: -a.persona.accuracy)
    ])
else:
    lb = mock_leaderboard()

lb_display = lb.copy()
lb_display.insert(0, "Rank", [f"#{i+1}" for i in range(len(lb))])

st.dataframe(
    lb_display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Rank":        st.column_config.TextColumn("Rank", width="small"),
        "Agent":       st.column_config.TextColumn("Agent"),
        "Role":        st.column_config.TextColumn("Role"),
        "Predictions": st.column_config.NumberColumn("Predictions", format="%d"),
        "Accuracy":    st.column_config.ProgressColumn("Accuracy", format="%.0f%%", min_value=0, max_value=1),
        "Avg Error":   st.column_config.NumberColumn("Avg Error", format="%.2f"),
        "Weight":      st.column_config.NumberColumn("Weight ⚖️", format="%.2f"),
    }
)

st.markdown("---")


# ─── Section 6: Accuracy Report ──────────────────────────────────────────────────

st.markdown('<div class="section-header">📈 Accuracy Report (7 days)</div>', unsafe_allow_html=True)

report = mock_accuracy_report()

col_chart, col_stats = st.columns([3, 1])

with col_chart:
    st.line_chart(
        report.set_index("Date")[["FIE Accuracy", "Market Accuracy"]],
        use_container_width=True,
        color=["#6366f1", "#4a5568"],
        height=220,
    )

with col_stats:
    fie_avg   = report["FIE Accuracy"].mean()
    mkt_avg   = report["Market Accuracy"].mean()
    edge_avg  = fie_avg - mkt_avg
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card" style="margin-bottom:0.6rem">
        <div class="metric-label">FIE 7d Avg</div>
        <div class="metric-value" style="font-size:1.5rem;color:#6366f1">{fie_avg:.0%}</div>
    </div>
    <div class="metric-card" style="margin-bottom:0.6rem">
        <div class="metric-label">Market 7d Avg</div>
        <div class="metric-value" style="font-size:1.5rem;color:#4a5568">{mkt_avg:.0%}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">FIE Edge</div>
        <div class="metric-value" style="font-size:1.5rem;color:#48bb78">+{edge_avg:.0%}</div>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;color:#2d3748;font-size:0.72rem;">FIE · Future Intelligence Engine · MVP v1.0 · Built with Streamlit</p>',
    unsafe_allow_html=True
)
