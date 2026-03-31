"""Shared CSS injected into every page."""

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
  background-color: #d1d1d1 !important;
  color: #1a1a1a !important;
  font-family: 'Inter', sans-serif !important;
}

/* Hide all Streamlit chrome including sidebar */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebarNav"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
.block-container { padding-top: 0 !important; padding-bottom: 3rem !important; max-width: 1100px !important; }

/* ── Top nav bar ── */
.topnav-logo {
  font-family: 'Inter', sans-serif;
  font-size: 1.15rem;
  font-weight: 900;
  letter-spacing: -0.5px;
  color: #1a1a1a;
  display: block;
  padding: 0.6rem 0;
}
.topnav-wrap {
  border-bottom: 1px solid #bababa;
  margin-bottom: 2rem;
  padding-bottom: 0.2rem;
}

/* Nuclear reset — strip every element inside the nav bar */
.topnav-wrap *,
.topnav-wrap *:hover,
.topnav-wrap *:focus,
.topnav-wrap *:active {
  background: transparent !important;
  background-color: transparent !important;
  box-shadow: none !important;
  border-color: transparent !important;
  outline: none !important;
}
/* Force all text/links/buttons in nav to be black */
.topnav-wrap a,
.topnav-wrap button,
.topnav-wrap span,
.topnav-wrap p {
  color: #1a1a1a !important;
  font-size: 0.88rem !important;
  font-weight: 500 !important;
  text-decoration: none !important;
}
.topnav-wrap a:hover,
.topnav-wrap button:hover { opacity: 0.5 !important; }
.topnav-wrap a[data-testid="stPageLink-NavLink"][aria-current],
.topnav-wrap a[data-testid="stPageLink-NavLink"][aria-current] * {
  font-weight: 700 !important;
}

/* Plain anchor used for Dashboard nav link (preserves ?e=1 param) */
.nav-plain-link {
  font-size: 0.88rem !important;
  font-weight: 500 !important;
  color: #1a1a1a !important;
  text-decoration: none !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  height: 100% !important;
  padding: 0.5rem 0 !important;
  text-align: center !important;
  transition: opacity 0.15s !important;
}
.nav-plain-link:hover { opacity: 0.6 !important; color: #1a1a1a !important; }

/* ── Hero ── */
.hero-wrap { margin-bottom: 2.5rem; border-bottom: 1px solid #bababa; padding-bottom: 2rem; }
.hero-title {
  font-family: 'Inter', sans-serif;
  font-size: 3.5rem;
  font-weight: 900;
  letter-spacing: -2px;
  line-height: 1.05;
  color: #1a1a1a;
  margin: 0 0 0.6rem 0;
}
.hero-sub {
  font-size: 1.05rem;
  color: #777;
  font-weight: 400;
  margin: 0;
}

/* ── Section heading ── */
.section-heading {
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: #aaa;
  margin: 2rem 0 1rem 0;
  border: none;
}

/* ── Stat cards ── */
.stat-card {
  background: #dedede;
  border: 1px solid #c0c0c0;
  border-radius: 8px;
  padding: 1.5rem 1.4rem;
  transition: box-shadow 0.15s;
}
.stat-card:hover { box-shadow: 0 2px 12px rgba(0,0,0,0.07); }
.stat-icon  { font-size: 1.2rem; margin-bottom: 0.8rem; display: block; }
.stat-value { font-size: 2.1rem; font-weight: 800; color: #1a1a1a; line-height: 1; letter-spacing: -1px; }
.stat-label { font-size: 0.75rem; color: #999; margin-top: 0.4rem; font-weight: 500; }

/* ── Signal cards ── */
.signal-card {
  background: #dedede;
  border: 1px solid #c0c0c0;
  border-radius: 8px;
  padding: 1.1rem 1.3rem;
  margin-bottom: 0.75rem;
  transition: box-shadow 0.15s;
  cursor: default;
}
.signal-card:hover { box-shadow: 0 2px 12px rgba(0,0,0,0.07); }
.signal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem; }
.signal-ticker { font-size: 1rem; font-weight: 700; color: #1a1a1a; }
.signal-badge {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.5px;
  background: #fff3e0;
  color: #e65100;
  border: 1px solid #ffcc80;
  border-radius: 4px;
  padding: 0.15rem 0.5rem;
}
.signal-prob { font-size: 0.88rem; color: #555; font-weight: 600; }
.signal-meta { font-size: 0.78rem; color: #aaa; margin-top: 0.3rem; }
.signal-meta span { margin-right: 1rem; }

/* ── Indicator card ── */
.ind-card {
  background: #dedede;
  border: 1px solid #c0c0c0;
  border-radius: 8px;
  padding: 0.8rem 1rem;
  margin-bottom: 0.5rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.ind-label { font-size: 0.75rem; color: #aaa; font-weight: 500; }
.ind-value  { font-size: 0.95rem; font-weight: 700; color: #1a1a1a; }

/* ── Confidence card ── */
.conf-card {
  background: #dedede;
  border: 1px solid #c0c0c0;
  border-radius: 8px;
  padding: 1.2rem;
  text-align: center;
  margin-top: 0.5rem;
}
.conf-value { font-size: 2rem; font-weight: 800; color: #1a1a1a; letter-spacing: -1px; margin-bottom: 0.2rem; }
.conf-label { font-size: 0.78rem; color: #aaa; font-weight: 500; }
.conf-badge {
  display: inline-block;
  font-size: 0.72rem;
  font-weight: 700;
  padding: 0.2rem 0.7rem;
  border-radius: 4px;
  margin-top: 0.5rem;
  letter-spacing: 0.3px;
}
.badge-buy  { background: #fff3e0; color: #e65100; border: 1px solid #ffcc80; }
.badge-none { background: #e8e5e0; color: #999;    border: 1px solid #d0ccc6; }

/* ── Performance cards ── */
.perf-card {
  background: #dedede;
  border: 1px solid #c0c0c0;
  border-radius: 8px;
  padding: 1.4rem 1rem;
  text-align: center;
}
.perf-value { font-size: 2rem; font-weight: 800; color: #1a1a1a; letter-spacing: -1px; }
.perf-label { font-size: 0.7rem; color: #aaa; margin-top: 0.4rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; }

/* ── Dataframe overrides ── */
[data-testid="stDataFrame"] { border: 1px solid #d8d4ce !important; border-radius: 8px !important; }

/* ── Watchlist remove button ── */
div[data-testid="stButton"] button[kind="secondary"] {
  background: none !important;
  border: 1px solid #d0ccc6 !important;
  color: #aaa !important;
  font-size: 0.75rem !important;
  padding: 0.25rem 0.5rem !important;
  border-radius: 4px !important;
}
div[data-testid="stButton"] button[kind="secondary"]:hover {
  border-color: #1a1a1a !important;
  color: #1a1a1a !important;
}

/* ── Footer ── */
.footer {
  margin-top: 4rem;
  padding-top: 1.2rem;
  border-top: 1px solid #bababa;
  font-size: 0.75rem;
  color: #bbb;
  text-align: center;
}
</style>
"""
