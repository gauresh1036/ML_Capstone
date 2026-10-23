"""Visual design: CSS, reusable HTML components and the Plotly chart style."""
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

PRIMARY = '#4F46E5'
ACCENT = '#06B6D4'
GOOD, WARN, BAD = '#10B981', '#F59E0B', '#EF4444'
INK, MUTED = '#0F172A', '#64748B'
# Colour-blind-safe qualitative palette (Okabe–Ito based)
SEQ = ['#4F46E5', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#0072B2', '#56B4E9', '#F0E442', '#999999', '#000000']
TRACK_COLORS = {'regression': '#4F46E5', 'classification': '#0891B2', 'clustering': '#7C3AED'}

pio.templates['capstone'] = go.layout.Template(
    layout=dict(font=dict(family='Inter, Segoe UI, sans-serif', color=INK, size=13),
                colorway=SEQ, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='#E2E8F0', zerolinecolor='#CBD5E1'),
                yaxis=dict(gridcolor='#E2E8F0', zerolinecolor='#CBD5E1'),
                title=dict(font=dict(size=16, color=INK), x=0.0, xanchor='left'),
                legend=dict(bgcolor='rgba(255,255,255,0.6)'),
                margin=dict(l=10, r=10, t=50, b=10)))
pio.templates.default = 'plotly_white+capstone'
px.defaults.color_discrete_sequence = SEQ

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Inter', 'Segoe UI', sans-serif; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1350px; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0F172A 0%, #1E1B4B 100%); }
section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
section[data-testid="stSidebar"] a[aria-current="page"] { background: rgba(99,102,241,.35) !important; border-radius: 10px; }
.hero { border-radius: 22px; padding: 34px 38px; color: white; margin-bottom: 22px;
        background: radial-gradient(1200px 400px at 10% -20%, rgba(255,255,255,.18), transparent),
                    linear-gradient(120deg, var(--c1) 0%, var(--c2) 100%);
        box-shadow: 0 20px 45px -20px rgba(79,70,229,.55); }
.hero h1 { color: white; font-size: 2.35rem; font-weight: 800; margin: 6px 0 6px 0; line-height: 1.15; }
.hero p { color: rgba(255,255,255,.88); font-size: 1.04rem; margin: 0; max-width: 900px; }
.hero .tag { display: inline-block; font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
             background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35); padding: 4px 12px; border-radius: 999px; }
.hero .chips { margin-top: 16px; }
.hero .chip { display: inline-block; margin: 4px 8px 0 0; padding: 6px 12px; border-radius: 10px; font-size: .85rem;
              background: rgba(15,23,42,.25); border: 1px solid rgba(255,255,255,.22); }
.card { background: white; border: 1px solid #E2E8F0; border-radius: 16px; padding: 18px 20px; height: 100%;
        box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px -16px rgba(15,23,42,.25); }
.card .label { color: #64748B; font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }
.card .value { color: #0F172A; font-size: 1.85rem; font-weight: 800; margin: 2px 0; }
.card .sub { color: #64748B; font-size: .85rem; }
.card .bar { height: 4px; border-radius: 4px; margin-bottom: 12px; background: var(--c); }
.track { background: white; border: 1px solid #E2E8F0; border-radius: 18px; padding: 22px; height: 100%;
         box-shadow: 0 10px 30px -18px rgba(15,23,42,.35); border-top: 5px solid var(--c); }
.track h3 { margin: 0 0 4px 0; font-size: 1.2rem; }
.track .q { color: #475569; font-size: .92rem; margin-bottom: 12px; }
.track .kv { display: flex; justify-content: space-between; border-bottom: 1px dashed #E2E8F0; padding: 6px 0; font-size: .9rem; }
.track .kv b { color: #0F172A; }
.section { margin: 26px 0 10px 0; }
.section h2 { font-size: 1.35rem; font-weight: 750; margin: 0; color: #0F172A; }
.section p { color: #64748B; margin: 2px 0 0 0; font-size: .93rem; }
.step { text-align: center; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 14px; padding: 14px 8px; height: 100%; }
.step .n { display: inline-block; width: 28px; height: 28px; line-height: 28px; border-radius: 50%; color: white;
           background: #4F46E5; font-weight: 700; font-size: .85rem; }
.step .t { font-weight: 650; font-size: .88rem; margin-top: 6px; color: #0F172A; }
.step .d { color: #64748B; font-size: .76rem; margin-top: 2px; }
.result { border-radius: 18px; padding: 24px; text-align: center; color: white;
          background: linear-gradient(135deg, var(--c1), var(--c2)); box-shadow: 0 16px 36px -18px rgba(15,23,42,.6); }
.result .big { font-size: 2.6rem; font-weight: 800; line-height: 1.1; }
.result .small { opacity: .9; font-size: .92rem; }
.pill { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .75rem; font-weight: 700;
        background: #EEF2FF; color: #4338CA; margin-right: 6px; }
div[data-testid="stTabs"] button[role="tab"] { font-weight: 600; padding: 10px 16px; }
div[data-testid="stMetric"] { background: white; border: 1px solid #E2E8F0; border-radius: 14px; padding: 12px 16px; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(tag, title, subtitle, chips=(), c1='#4F46E5', c2='#7C3AED'):
    chip_html = ''.join(f'<span class="chip">{c}</span>' for c in chips)
    st.markdown(f"""<div class="hero" style="--c1:{c1};--c2:{c2}"><span class="tag">{tag}</span>
<h1>{title}</h1><p>{subtitle}</p><div class="chips">{chip_html}</div></div>""", unsafe_allow_html=True)


def cards(items, color=PRIMARY):
    """items: list of (label, value, sub)."""
    cols = st.columns(len(items))
    for col, (label, value, sub) in zip(cols, items):
        col.markdown(f"""<div class="card"><div class="bar" style="--c:{color}"></div><div class="label">{label}</div>
<div class="value">{value}</div><div class="sub">{sub}</div></div>""", unsafe_allow_html=True)


def section(title, subtitle=''):
    st.markdown(f'<div class="section"><h2>{title}</h2><p>{subtitle}</p></div>', unsafe_allow_html=True)


def result_box(big, small, c1=PRIMARY, c2='#7C3AED'):
    st.markdown(f'<div class="result" style="--c1:{c1};--c2:{c2}"><div class="small">{small[0]}</div>'
                f'<div class="big">{big}</div><div class="small">{small[1]}</div></div>', unsafe_allow_html=True)


def steps(items):
    cols = st.columns(len(items))
    for i, (col, (t, d)) in enumerate(zip(cols, items), 1):
        col.markdown(f'<div class="step"><span class="n">{i}</span><div class="t">{t}</div><div class="d">{d}</div></div>',
                     unsafe_allow_html=True)


def chart(fig, height=420):
    fig.update_layout(height=height)
    st.plotly_chart(fig, width='stretch', config={'displaylogo': False})
