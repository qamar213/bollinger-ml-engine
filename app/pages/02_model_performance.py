"""
pages/02_model_performance.py — VolCast · Model Performance
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.styles import GLOBAL_CSS
from app.utils import get_model_metrics, get_feature_importance, get_cv_metrics

st.set_page_config(page_title="Model Performance · VolCast", page_icon="📈", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.markdown('<div class="topnav-wrap">', unsafe_allow_html=True)
nav_logo, nav_home, nav_sig, nav_perf, nav_pad = st.columns([2, 1, 1.3, 1.5, 3])
with nav_logo:
    st.markdown('<span class="topnav-logo">VolCast</span>', unsafe_allow_html=True)
with nav_home:
    st.markdown('<a href="/?e=1" target="_self" class="nav-plain-link">Dashboard</a>', unsafe_allow_html=True)
with nav_sig:
    st.markdown('<a href="/signal_explorer" target="_self" class="nav-plain-link">Signal Explorer</a>', unsafe_allow_html=True)
with nav_perf:
    st.markdown('<a href="/model_performance" target="_self" class="nav-plain-link">Model Performance</a>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="hero-wrap">
  <p class="hero-title">Model Performance</p>
  <p class="hero-sub">Test-set metrics &nbsp;·&nbsp; Cross-validation &nbsp;·&nbsp; Feature importance</p>
</div>
""", unsafe_allow_html=True)

metrics    = get_model_metrics()
importance = get_feature_importance()
cv         = get_cv_metrics()

# Test-set metric cards
if metrics:
    st.markdown('<p class="section-heading">Test-Set Results</p>', unsafe_allow_html=True)
    cols = st.columns(5)
    specs = [
        ("Accuracy",  "accuracy",  ".1%"),
        ("Precision", "precision", ".1%"),
        ("Recall",    "recall",    ".1%"),
        ("F1 Score",  "f1",        ".3f"),
        ("ROC-AUC",   "roc_auc",   ".3f"),
    ]
    for col, (label, key, fmt) in zip(cols, specs):
        val = metrics.get(key, 0)
        with col:
            st.markdown(f"""<div class="perf-card">
              <div class="perf-value">{val:{fmt}}</div>
              <div class="perf-label">{label}</div>
            </div>""", unsafe_allow_html=True)
else:
    st.warning("No test metrics found. Run the pipeline first.")

st.markdown("<br>", unsafe_allow_html=True)

# CV results
if cv:
    st.markdown('<p class="section-heading">Cross-Validation · 5-Fold Walk-Forward</p>', unsafe_allow_html=True)
    folds = cv.get("folds", [])
    mean  = cv.get("mean",  {})
    if folds:
        fold_df  = pd.DataFrame(folds)[["fold", "precision", "recall", "f1", "roc_auc"]]
        mean_row = pd.DataFrame([{"fold": "Mean", **{k: mean[k] for k in ["precision","recall","f1","roc_auc"]}}])
        display  = pd.concat([fold_df, mean_row], ignore_index=True)
        display[["precision","recall","f1","roc_auc"]] = display[["precision","recall","f1","roc_auc"]].map(
            lambda x: f"{x:.3f}" if isinstance(x, float) else x
        )
        st.table(
            display.rename(columns={"fold":"Fold","precision":"Precision","recall":"Recall","f1":"F1","roc_auc":"ROC-AUC"})
        )

        numeric_folds = [f for f in folds if isinstance(f.get("fold"), int)]
        if numeric_folds:
            fp = pd.DataFrame(numeric_folds)
            fig_cv = go.Figure()
            for metric, color in [("roc_auc","#1a1a1a"),("precision","#555"),("f1","#888")]:
                fig_cv.add_trace(go.Bar(
                    name=metric.replace("_"," ").title(),
                    x=[f"Fold {f['fold']}" for f in numeric_folds],
                    y=fp[metric], marker_color=color,
                ))
            fig_cv.update_layout(
                barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#1a1a1a"), legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#1a1a1a")),
                margin=dict(l=0,r=0,t=10,b=0), height=300,
                yaxis=dict(range=[0,1], gridcolor="#bababa", tickfont=dict(color="#1a1a1a")),
                xaxis=dict(gridcolor="#bababa", tickfont=dict(color="#1a1a1a")),
            )
            st.plotly_chart(fig_cv, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Feature importance
if importance is not None and not importance.empty:
    st.markdown('<p class="section-heading">Top 25 Features by Importance</p>', unsafe_allow_html=True)
    top25 = importance.head(25).reset_index()
    top25.columns = ["feature", "importance"]
    fig_imp = px.bar(
        top25.sort_values("importance"), x="importance", y="feature",
        orientation="h", color="importance",
        color_continuous_scale=["#e0e0e0","#888","#1a1a1a"],
    )
    fig_imp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1a1a1a"), coloraxis_showscale=False,
        margin=dict(l=0,r=0,t=10,b=0), height=550,
        xaxis=dict(gridcolor="#bababa", tickfont=dict(color="#1a1a1a"), title=dict(text="importance", font=dict(color="#1a1a1a"))),
        yaxis=dict(gridcolor="#bababa", tickfont=dict(color="#1a1a1a"), title=""),
    )
    st.plotly_chart(fig_imp, use_container_width=True)
else:
    st.warning("No feature importance data. Run the pipeline first.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Volatility regime predictions by XGBoost trained on 6 years of daily OHLCV data across 50 S&P 500 tickers.
  &nbsp;·&nbsp; Not financial advice.
</div>""", unsafe_allow_html=True)
