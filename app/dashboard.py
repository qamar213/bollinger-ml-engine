"""
dashboard.py — VolCast · Home
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import streamlit.components.v1 as components

from app.styles import GLOBAL_CSS
from app.utils import (
    TICKERS, load_model,
    _build_features_for_inference, fetch_ticker_history,
)

ROOT = Path(__file__).resolve().parent.parent
WATCHLIST_PATH = ROOT / "data" / "watchlist.json"


def load_watchlist() -> list:
    if WATCHLIST_PATH.exists():
        try:
            return json.loads(WATCHLIST_PATH.read_text())
        except Exception:
            return []
    return []


def save_watchlist(tickers: list) -> None:
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    WATCHLIST_PATH.write_text(json.dumps(tickers))


def get_vol_prediction(ticker: str, ticker_idx: int):
    """Return (proba, pred) for a single ticker using live data."""
    try:
        df = fetch_ticker_history(ticker, period="2y")
        if df is None or len(df) < 250:
            return None, None
        X = _build_features_for_inference(df, ticker, ticker_idx)
        if X is None:
            return None, None
        model = load_model()
        if model is None:
            return None, None
        proba = float(model.predict_proba(X).iloc[0])
        pred  = int(model.predict(X).iloc[0])
        return proba, pred
    except Exception:
        return None, None


st.set_page_config(
    page_title="VolCast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── Entrance gate — persists via query param across page navigations ───────────
if st.query_params.get("e") == "1":
    st.session_state.entered = True
if "entered" not in st.session_state:
    st.session_state.entered = False

if not st.session_state.entered:
    # Strip all Streamlit chrome for the entrance screen
    st.markdown("""
<style>
.block-container {
  padding: 0 !important;
  padding-top: 0 !important;
  max-width: 100vw !important;
}
[data-testid="stVerticalBlock"] { gap: 0 !important; }
/* Style the Enter button */
[data-testid="stButton"] > button {
  background: #1a1a1a !important;
  color: #f4f3f0 !important;
  border: none !important;
  border-radius: 100px !important;
  font-size: 0.88rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  padding: 0.7rem 2.5rem !important;
  transition: opacity 0.2s !important;
  margin-top: 0.5rem !important;
}
[data-testid="stButton"] > button:hover { opacity: 0.75 !important; }
</style>""", unsafe_allow_html=True)

    ENTRANCE_GLOBE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@900&display=swap" rel="stylesheet">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { width: 100%; height: 100%; background: #d1d1d1; overflow: hidden; }

  #wrap {
    width: 100%; height: 100%;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 0;
  }

  #wordmark {
    font-family: 'Inter', -apple-system, sans-serif;
    font-size: clamp(3.5rem, 10vw, 6.5rem);
    font-weight: 900;
    letter-spacing: -0.04em;
    color: #1a1a1a;
    line-height: 1;
    text-align: center;
    animation: fadeUp 1s cubic-bezier(0.16,1,0.3,1) 0.1s both;
    z-index: 2;
  }

  #tagline {
    font-family: 'Inter', -apple-system, sans-serif;
    font-size: 0.9rem;
    color: #aaa;
    font-weight: 400;
    letter-spacing: 0.02em;
    margin-top: 0.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
    animation: fadeUp 1s cubic-bezier(0.16,1,0.3,1) 0.25s both;
    z-index: 2;
  }

  #globe-wrap {
    position: relative;
    width: min(380px, 80vw);
    height: min(380px, 80vw);
    animation: fadeUp 1.1s cubic-bezier(0.16,1,0.3,1) 0.35s both;
  }
  canvas { width: 100% !important; height: 100% !important; }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
  }
</style>
</head>
<body>
<div id="wrap">
  <div id="wordmark">VolCast</div>
  <div id="tagline">Volatility regime prediction &nbsp;·&nbsp; 50 S&P 500 tickers</div>
  <div id="globe-wrap">
    <canvas id="c"></canvas>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function () {
  const canvas = document.getElementById('c');
  const wrap   = document.getElementById('globe-wrap');
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene  = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
  camera.position.set(0, 0, 3.4);

  function resize() {
    const s = wrap.clientWidth;
    renderer.setSize(s, s, false);
  }
  resize();
  window.addEventListener('resize', resize);

  // Particle globe — Fibonacci sphere
  const N = 1400;
  const pos = new Float32Array(N * 3);
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < N; i++) {
    const y = 1 - (i / (N - 1)) * 2;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const phi = i * golden;
    pos[i*3]   = r * Math.cos(phi);
    pos[i*3+1] = y;
    pos[i*3+2] = r * Math.sin(phi);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const mat = new THREE.PointsMaterial({
    color: 0x1a1a1a, size: 0.025,
    sizeAttenuation: true, transparent: true, opacity: 0.55,
  });
  const globe = new THREE.Points(geo, mat);
  scene.add(globe);

  // Arc lines between random globe points
  function rndSphere() {
    const u = Math.random(), v = Math.random();
    const th = 2 * Math.PI * u, ph = Math.acos(2 * v - 1);
    return new THREE.Vector3(
      Math.sin(ph) * Math.cos(th),
      Math.cos(ph),
      Math.sin(ph) * Math.sin(th),
    );
  }
  for (let i = 0; i < 20; i++) {
    const p1 = rndSphere(), p2 = rndSphere();
    const pts = [];
    for (let t = 0; t <= 1; t += 0.025) {
      pts.push(p1.clone().lerp(p2, t).normalize().multiplyScalar(1 + 0.2 * Math.sin(Math.PI * t)));
    }
    const lg = new THREE.BufferGeometry().setFromPoints(pts);
    const lm = new THREE.LineBasicMaterial({ color: 0x888888, transparent: true, opacity: 0.15 });
    scene.add(new THREE.Line(lg, lm));
  }

  // Mouse parallax
  let mx = 0, my = 0;
  document.addEventListener('mousemove', e => {
    mx = (e.clientX / window.innerWidth  - 0.5) * 2;
    my = (e.clientY / window.innerHeight - 0.5) * 2;
  });

  function animate() {
    requestAnimationFrame(animate);
    globe.rotation.y += 0.003;
    globe.rotation.x += (my * 0.25 - globe.rotation.x) * 0.05;
    globe.rotation.y += mx * 0.001;
    renderer.render(scene, camera);
  }
  animate();
})();
</script>
</body>
</html>
"""

    components.html(ENTRANCE_GLOBE, height=580, scrolling=False)

    # Enter button — centered via columns
    _, btn_mid, _ = st.columns([2, 1, 2])
    with btn_mid:
        if st.button("Enter →", use_container_width=True):
            st.session_state.entered = True
            st.query_params["e"] = "1"
            st.rerun()

    st.stop()

# ── Main dashboard ─────────────────────────────────────────────────────────────

# ── Top nav ────────────────────────────────────────────────────────────────────
st.markdown('<div class="topnav-wrap">', unsafe_allow_html=True)
nav_logo, nav_home, nav_sig, nav_perf, nav_pad = st.columns([2, 1, 1.3, 1.5, 3])
with nav_logo:
    st.markdown('<span class="topnav-logo">VolCast</span>', unsafe_allow_html=True)
with nav_home:
    st.markdown('<a href="/?e=1" target="_self" class="nav-plain-link">Dashboard</a>', unsafe_allow_html=True)
with nav_sig:
    st.page_link("pages/01_signal_explorer.py", label="Signal Explorer")
with nav_perf:
    st.page_link("pages/02_model_performance.py", label="Model Performance")
st.markdown('</div>', unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <p class="hero-title">VolCast</p>
  <p class="hero-sub">Volatility regime prediction &nbsp;·&nbsp; 50 S&P 500 tickers</p>
</div>
""", unsafe_allow_html=True)

# ── Watchlist state ────────────────────────────────────────────────────────────
if "watchlist" not in st.session_state:
    st.session_state.watchlist = load_watchlist()

# ── Add to watchlist ───────────────────────────────────────────────────────────
st.markdown('<p class="section-heading">Watchlist</p>', unsafe_allow_html=True)

add_col, btn_col = st.columns([3, 1])
with add_col:
    available = [t for t in TICKERS if t not in st.session_state.watchlist]
    pick = st.selectbox("Add a ticker to track", ["— select —"] + available, label_visibility="collapsed")
with btn_col:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("+ Add", use_container_width=True) and pick != "— select —":
        st.session_state.watchlist.append(pick)
        save_watchlist(st.session_state.watchlist)
        st.rerun()

# ── Watchlist cards ────────────────────────────────────────────────────────────
ticker_index = {t: i for i, t in enumerate(TICKERS)}

if not st.session_state.watchlist:
    st.markdown("""
    <div style="background:#eeecea;border:1px solid #e0deda;border-radius:8px;
                padding:2.5rem;text-align:center;margin-top:0.5rem">
      <div style="font-size:1.4rem;margin-bottom:0.5rem">📌</div>
      <div style="font-weight:700;color:#1a1a1a;margin-bottom:0.3rem">No tickers bookmarked yet</div>
      <div style="font-size:0.85rem;color:#aaa">Add tickers above to track their volatility predictions here.</div>
    </div>""", unsafe_allow_html=True)
else:
    cols = st.columns(3)
    to_remove = None
    for i, ticker in enumerate(st.session_state.watchlist):
        col = cols[i % 3]
        with col:
            with st.spinner(f""):
                proba, pred = get_vol_prediction(ticker, ticker_index.get(ticker, 0))

            if proba is None:
                status_label = "Unavailable"
                status_color = "#aaa"
                badge_bg     = "#eeecea"
                badge_border = "#e0deda"
                proba_str    = "—"
            elif pred == 1:
                status_label = "Expanding"
                status_color = "#b45309"
                badge_bg     = "#fef3c7"
                badge_border = "#fcd34d"
                proba_str    = f"{proba:.0%}"
            else:
                status_label = "Contracting"
                status_color = "#166534"
                badge_bg     = "#dcfce7"
                badge_border = "#86efac"
                proba_str    = f"{1 - proba:.0%}"

            st.markdown(f"""
            <div class="signal-card" style="margin-bottom:0.75rem">
              <div class="signal-header">
                <span class="signal-ticker">{ticker}</span>
                <span style="font-size:0.68rem;font-weight:700;padding:0.15rem 0.5rem;
                             border-radius:4px;background:{badge_bg};
                             color:{status_color};border:1px solid {badge_border}">
                  {status_label}
                </span>
              </div>
              <div style="font-size:1.6rem;font-weight:800;color:#1a1a1a;
                          letter-spacing:-1px;line-height:1;margin:0.5rem 0 0.2rem">
                {proba_str}
              </div>
              <div style="font-size:0.75rem;color:#aaa">confidence</div>
            </div>""", unsafe_allow_html=True)

            if st.button("Remove", key=f"rm_{ticker}", use_container_width=True):
                to_remove = ticker

    if to_remove:
        st.session_state.watchlist.remove(to_remove)
        save_watchlist(st.session_state.watchlist)
        st.rerun()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Volatility regime predictions by XGBoost trained on 6 years of daily OHLCV data across 50 S&P 500 tickers.
  &nbsp;·&nbsp; Not financial advice.
</div>""", unsafe_allow_html=True)
