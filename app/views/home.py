import pandas as pd
import plotly.express as px
import streamlit as st

import data
import theme as T


def render():
    T.inject_css()
    reg, cls, clu = data.results('regression'), data.results('classification'), data.results('clustering')
    km = next(v for k, v in clu['metrics'].items() if k.startswith('K-Means'))
    T.hero('23CSE301 Machine Learning · Capstone · Review 2',
           'Three problems. 22 algorithms. One reproducible pipeline.',
           'An end-to-end machine-learning project covering regression, classification and clustering. Explore the data, '
           'compare every model, and test the trained models live — every number comes from the executed notebooks.',
           ['Python · scikit-learn · Streamlit', 'random_state = 42', 'leak-free pipelines', '5-fold cross-validation'])

    c1, c2, c3 = st.columns(3)
    tracks = [
        (c1, T.TRACK_COLORS['regression'], '🏡 Regression', 'What is this house worth?', 'Ames Housing · 2,930 sales',
         [('Algorithms', '10'), ('Best model', reg['best_model']), ('Test R²', f"{reg['test_metrics']['R2']:.3f}"),
          ('RMSE', f"${reg['test_metrics']['RMSE']:,.0f}")]),
        (c2, T.TRACK_COLORS['classification'], '🛒 Classification', 'Will this shopper buy?',
         'Online Shoppers · 12,330 sessions',
         [('Algorithms', '10 (Part A + B)'), ('Best model', cls['best_model']),
          ('Weighted F1', f"{cls['test_metrics']['F1 (w)']:.3f}"), ('ROC-AUC', f"{cls['test_metrics']['ROC-AUC']:.3f}")]),
        (c3, T.TRACK_COLORS['clustering'], '👥 Clustering', 'Who are our customers?',
         'Online Retail · 541,909 invoice lines',
         [('Algorithms', 'K-Means, Agglomerative'), ('Segments', str(clu['k'])),
          ('Silhouette', f"{km['Silhouette']:.3f}"), ('Customers', f"{clu['n_customers']:,}")]),
    ]
    for col, color, title, q, ds, kv in tracks:
        rows = ''.join(f'<div class="kv"><span>{k}</span><b>{v}</b></div>' for k, v in kv)
        col.markdown(f'<div class="track" style="--c:{color}"><h3>{title}</h3><div class="q"><b>{q}</b><br>{ds}</div>'
                     f'{rows}</div>', unsafe_allow_html=True)
    st.write('')
    b1, b2, b3 = st.columns(3)
    b1.page_link('views/regression.py', label='Open the regression track →', icon='🏡')
    b2.page_link('views/classification.py', label='Open the classification track →', icon='🛒')
    b3.page_link('views/clustering.py', label='Open the clustering track →', icon='👥')

    T.section('One pipeline for every track', 'The same disciplined workflow, from raw file to deployed model')
    T.steps([('Load & audit', 'shape, types, NaN'), ('EDA', 'plots + observations'), ('Clean', 'justified rules'),
             ('Engineer', 'domain features'), ('Preprocess', 'fit on train only'), ('Baseline', 'library defaults'),
             ('Tune', 'GridSearchCV 5-fold'), ('Compare', 'one table'), ('Validate', 'CV of top-2'),
             ('Deploy', 'saved pipeline → GUI')])

    T.section('All 20 supervised models at a glance')
    a, b = st.columns(2)
    with a:
        r = pd.DataFrame(reg['comparison']).sort_values('R2')
        fig = px.bar(r, x='R2', y='Model', orientation='h', title='Regression — test R²',
                     color_discrete_sequence=[T.TRACK_COLORS['regression']], text='R2')
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_xaxes(range=[0.8, 0.96])
        T.chart(fig, 420)
    with b:
        c = pd.DataFrame(cls['comparison']).sort_values('F1 (w)')
        fig = px.bar(c, x='F1 (w)', y='Model', orientation='h', title='Classification — test weighted F1',
                     color_discrete_sequence=[T.TRACK_COLORS['classification']], text='F1 (w)')
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_xaxes(range=[0.8, 0.92])
        T.chart(fig, 420)
    st.caption('Use the sidebar to open a track. Every track has: Overview · Explore the data · All 10 models · '
               'Model lab (test any model with your own inputs) · Evaluate a model (live on the test split) · Figures.')


render()
