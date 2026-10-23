import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)

import data
import theme as T
from features import engineer_session_features

C1, C2 = '#0891B2', '#4F46E5'
METRICS = ['Accuracy', 'Precision (w)', 'Recall (w)', 'F1 (w)', 'ROC-AUC']


def render():
    T.inject_css()
    res = data.results('classification')
    df = data.shoppers_raw()
    T.hero('Track 2 · Classification · Part A + Part B',
           'Will this shopper buy? 🛒',
           'Ten classifiers predict whether an online browsing session ends in a purchase, using 12,330 sessions from '
           'the UCI Online Shoppers Purchasing Intention dataset.',
           [f"Best: {res['best_model']}", f"Weighted F1 {res['test_metrics']['F1 (w)']:.3f}",
            f"ROC-AUC {res['test_metrics']['ROC-AUC']:.3f}", '10 algorithms · GridSearchCV · 5-fold CV'], C1, C2)

    tabs = st.tabs(['📌 Overview', '🔍 Explore the data', '🏆 All 10 models', '🧪 Model lab', '🎯 Evaluate a model',
                    '🖼️ Notebook figures'])
    with tabs[0]:
        overview(res, df)
    with tabs[1]:
        eda(df)
    with tabs[2]:
        models(res)
    with tabs[3]:
        lab(res)
    with tabs[4]:
        evaluate()
    with tabs[5]:
        gallery()


# ----------------------------------------------------------------------------------------------- overview
def overview(res, df):
    T.cards([('Sessions', f'{len(df):,}', f"{res['n_rows_after_cleaning']:,} after removing 125 duplicates"),
             ('Purchase rate', f"{res['purchase_rate']:.1%}", 'imbalanced target → weighted F1 + ROC-AUC'),
             ('Best weighted F1', f"{res['test_metrics']['F1 (w)']:.4f}", res['best_model']),
             ('Best ROC-AUC', f"{res['test_metrics']['ROC-AUC']:.4f}",
              f"CV F1 {res['cv_top2'][res['best_model']]['F1 (w) mean']:.4f} ± {res['cv_top2'][res['best_model']]['F1 (w) std']:.4f}")],
            C1)
    T.section('The pipeline', 'Identical for all ten models, so every score is directly comparable')
    T.steps([('Audit', '18 cols, 0 missing'), ('EDA', 'skew, imbalance, correlation'),
             ('Clean', '125 duplicates removed'), ('Engineer', '5 new features'),
             ('Encode & scale', 'log1p · one-hot · ordinal'), ('Split', 'stratified 80:20'),
             ('Tune', 'GridSearchCV, 5-fold'), ('Select', 'CV of top-2')])
    T.section('Leaderboard', 'Tuned models on the same 2,441-session test split')
    comp = pd.DataFrame(res['comparison'])
    fig = px.bar(comp.sort_values('F1 (w)'), x='F1 (w)', y='Model', color='Part', orientation='h', text='F1 (w)',
                 color_discrete_map={'A': '#94A3B8', 'B': C1}, title='Weighted F1 — Part A (grey) vs Part B (teal)')
    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig.update_xaxes(range=[comp['F1 (w)'].min() - 0.02, comp['F1 (w)'].max() + 0.01])
    T.chart(fig, 430)
    c1, c2 = st.columns(2)
    c1.success(f"**Why {res['best_model']} wins:** best test F1 and ROC-AUC *and* the best 5-fold cross-validated F1 "
               '— not a lucky split. Its test score lies inside its CV range.')
    c2.info(f"**Business threshold:** lowering the decision threshold to **{res['decision_threshold']:.2f}** "
            '(chosen on out-of-fold training predictions) catches 74 % of buyers instead of 62 %.')


# ----------------------------------------------------------------------------------------------- EDA
def eda(df):
    plot_df = df.assign(Class=df['Revenue'].map({False: 'No purchase', True: 'Purchase'}))
    colors = {'No purchase': '#94A3B8', 'Purchase': C1}
    T.section('Dataset at a glance')
    c1, c2 = st.columns([3, 2])
    with c1:
        st.dataframe(df.head(200), height=320, width='stretch')
    with c2:
        counts = plot_df['Class'].value_counts()
        fig = px.pie(values=counts.values, names=counts.index, hole=0.6, color=counts.index, color_discrete_map=colors,
                     title='Class balance of Revenue')
        fig.update_traces(textinfo='percent+label')
        T.chart(fig, 320)
    with st.expander('Audit: data types, missing values, unique values, summary statistics'):
        audit = pd.DataFrame({'dtype': df.dtypes.astype(str), 'missing': df.isna().sum(), 'unique': df.nunique()})
        a, b = st.columns([1, 2])
        a.dataframe(audit, width='stretch')
        b.dataframe(data.numeric_summary(df.select_dtypes('number')), width='stretch')

    num_cols = [c for c in df.select_dtypes('number').columns if c not in data.NOMINAL_CODES]
    cat_cols = ['Month', 'VisitorType', 'Weekend'] + data.NOMINAL_CODES

    T.section('Feature explorer — numerical', 'Pick a feature to see how buyers and non-buyers differ')
    c1, c2, c3 = st.columns([2, 1, 1])
    feat = c1.selectbox('Feature', num_cols, index=num_cols.index('PageValues'), key='cls_num')
    logx = c2.toggle('Log scale', value=True, key='cls_log')
    norm = c3.toggle('Normalise per class', value=True, key='cls_norm')
    x = np.log1p(plot_df[feat]) if logx else plot_df[feat]
    a, b = st.columns([3, 2])
    with a:
        fig = px.histogram(plot_df.assign(x=x), x='x', color='Class', barmode='overlay', nbins=50, opacity=0.65,
                           histnorm='percent' if norm else None, color_discrete_map=colors,
                           title=f'Distribution of {feat}' + (' (log1p)' if logx else ''))
        fig.update_xaxes(title=f'log(1 + {feat})' if logx else feat)
        T.chart(fig, 380)
    with b:
        fig = px.box(plot_df.assign(x=x), x='Class', y='x', color='Class', color_discrete_map=colors, points=False,
                     title=f'{feat} by class')
        fig.update_yaxes(title=feat)
        T.chart(fig, 380)
    bins = pd.qcut(df[feat].rank(method='first'), 10, labels=False)
    trend = df.groupby(bins).agg(value=(feat, 'mean'), rate=('Revenue', 'mean')).reset_index(drop=True)
    st.caption(f"Purchase rate by decile of **{feat}**: lowest decile {trend['rate'].iloc[0]:.1%} → "
               f"highest decile {trend['rate'].iloc[-1]:.1%} · correlation with Revenue r = "
               f"{df[feat].corr(df['Revenue'].astype(int)):.3f}")

    T.section('Feature explorer — categorical', 'Session volume (bars) and conversion rate (line)')
    cat = st.selectbox('Feature', cat_cols, key='cls_cat')
    g = df.groupby(cat)['Revenue'].agg(['size', 'mean']).reset_index()
    if cat == 'Month':
        g['order'] = g['Month'].map({m: i for i, m in enumerate(data.MONTH_ORDER)})
        g = g.sort_values('order')
    g[cat] = g[cat].astype(str)
    fig = go.Figure()
    fig.add_bar(x=g[cat], y=g['size'], name='Sessions', marker_color='#C7D2FE')
    fig.add_scatter(x=g[cat], y=g['mean'] * 100, name='Purchase rate (%)', yaxis='y2', mode='lines+markers',
                    line=dict(color=C1, width=3))
    fig.update_layout(title=f'{cat}: sessions and purchase rate', yaxis=dict(title='Sessions'),
                      yaxis2=dict(title='Purchase rate (%)', overlaying='y', side='right', showgrid=False),
                      xaxis=dict(title=cat, type='category'), legend=dict(orientation='h', y=1.12))
    T.chart(fig, 380)

    T.section('Correlation heatmap')
    corr = df[num_cols].assign(Revenue=df['Revenue'].astype(int)).corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1, aspect='auto',
                    title='Pearson correlation (numerical features + Revenue)')
    T.chart(fig, 560)


# ----------------------------------------------------------------------------------------------- 10 models
def models(res):
    comp = pd.DataFrame(res['comparison'])
    comp.index = np.arange(1, len(comp) + 1)
    T.section('Consolidated comparison — all 10 algorithms',
              'Review 2 deliverable D6 · same test split · Precision / Recall / F1 are weighted averages')
    cols = ['Model', 'Part'] + METRICS + ['Purchase Recall', 'Purchase F1']
    st.dataframe(comp[cols].style.format({m: '{:.4f}' for m in cols[2:]})
                 .background_gradient(subset=cols[2:], cmap='GnBu'), width='stretch', height=420)
    chosen = st.multiselect('Metrics to plot', METRICS, default=['Accuracy', 'F1 (w)', 'ROC-AUC'])
    if chosen:
        long = comp.melt(id_vars='Model', value_vars=chosen, var_name='Metric', value_name='Score')
        fig = px.bar(long, x='Model', y='Score', color='Metric', barmode='group', title='Metric comparison')
        fig.update_yaxes(range=[max(0, long['Score'].min() - 0.05), 1])
        T.chart(fig, 430)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(comp, x='F1 (w)', y='ROC-AUC', text='Model', color='Part', size=[14] * len(comp),
                         color_discrete_map={'A': '#94A3B8', 'B': C1}, title='Weighted F1 vs ROC-AUC')
        fig.update_traces(textposition='top center', textfont_size=10)
        T.chart(fig, 430)
    with c2:
        g = pd.DataFrame(res['tuning_gains'])
        long = g.melt(id_vars='Model', value_vars=['F1 (w) baseline', 'F1 (w) tuned'], var_name='Stage', value_name='F1')
        fig = px.bar(long, x='F1', y='Model', color='Stage', barmode='group', orientation='h',
                     color_discrete_sequence=['#CBD5E1', C2], title='Effect of tuning (baseline → GridSearchCV)')
        fig.update_xaxes(range=[long['F1'].min() - 0.02, long['F1'].max() + 0.01])
        T.chart(fig, 430)
    T.section('Best hyper-parameters (5-fold GridSearchCV on the training split)')
    g = pd.DataFrame(res['tuning_gains'])[['Model', 'Part', 'Best parameters', 'Δ F1 (w)', 'Δ ROC-AUC']]
    st.dataframe(g.style.format({'Δ F1 (w)': '{:+.4f}', 'Δ ROC-AUC': '{:+.4f}'}), width='stretch', hide_index=True)
    T.section('Final selection — 5-fold cross-validation of the top-2')
    cv = pd.DataFrame(res['cv_top2']).T
    st.dataframe(cv.style.format('{:.4f}'), width='stretch')


# ----------------------------------------------------------------------------------------------- model lab
PRESETS = {
    'Custom': {},
    '🔥 High-intent shopper': dict(PageValues=40.0, ProductRelated=45, ProductRelated_Duration=1800.0, ExitRates=0.012,
                                  BounceRates=0.002, Month='Nov', VisitorType='New_Visitor', Administrative=4,
                                  Administrative_Duration=120.0),
    '👀 Casual browser': dict(PageValues=0.0, ProductRelated=15, ProductRelated_Duration=450.0, ExitRates=0.035,
                             BounceRates=0.01, Month='May', VisitorType='Returning_Visitor'),
    '💨 Quick bounce': dict(PageValues=0.0, ProductRelated=1, ProductRelated_Duration=0.0, ExitRates=0.2,
                           BounceRates=0.2, Month='Mar', VisitorType='Returning_Visitor', Administrative=0,
                           Administrative_Duration=0.0),
}


def lab(res):
    allm = data.model('classification_all_models.pkl')
    names = list(allm)
    schema = res['input_schema']
    rng = schema['numeric_ranges']
    T.section('Model lab', 'Describe a browsing session, choose any of the 10 trained models, and compare all of them')
    c1, c2 = st.columns([2, 3])
    model_name = c1.selectbox('Model', names, index=names.index(res['best_model']))
    preset = c2.radio('Start from', list(PRESETS), horizontal=True)
    p = PRESETS[preset]

    def v(col, default=None):
        return p.get(col, rng[col][2] if default is None else default)

    with st.form('cls_lab'):
        a, b, c = st.columns(3)
        with a:
            st.markdown('**Pages visited**')
            adm = st.number_input('Administrative pages', 0, 30, int(v('Administrative')))
            adm_d = st.number_input('Administrative duration (s)', 0.0, 4000.0, float(v('Administrative_Duration')))
            inf = st.number_input('Informational pages', 0, 25, int(v('Informational')))
            inf_d = st.number_input('Informational duration (s)', 0.0, 3000.0, float(v('Informational_Duration')))
            prod = st.number_input('Product pages', 0, 710, int(v('ProductRelated')))
            prod_d = st.number_input('Product-page duration (s)', 0.0, 65000.0, float(v('ProductRelated_Duration')))
        with b:
            st.markdown('**Engagement**')
            pv = st.number_input('Page value', 0.0, 362.0, float(v('PageValues', 0.0)),
                                 help='Average Google-Analytics value of the pages visited — the strongest signal.')
            bounce = st.slider('Bounce rate', 0.0, 0.2, float(v('BounceRates')), 0.001, format='%.3f')
            exit_r = st.slider('Exit rate', 0.0, 0.2, float(v('ExitRates')), 0.001, format='%.3f')
            special = st.select_slider('Closeness to a special day', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 0.0)
        with c:
            st.markdown('**Context**')
            months = schema['months']
            month = st.selectbox('Month', months, months.index(p.get('Month', 'May')))
            vts = schema['visitor_types']
            vt = st.selectbox('Visitor type', vts, vts.index(p.get('VisitorType', 'Returning_Visitor')))
            weekend = st.toggle('Weekend session')
            codes = schema['codes']
            os_ = st.selectbox('Operating system code', codes['OperatingSystems'], codes['OperatingSystems'].index(2))
            br = st.selectbox('Browser code', codes['Browser'], codes['Browser'].index(2))
            reg = st.selectbox('Region code', codes['Region'], 0)
            tt = st.selectbox('Traffic source code', codes['TrafficType'], codes['TrafficType'].index(2))
        go_ = st.form_submit_button('⚡ Predict with all 10 models', type='primary', width='stretch')

    if not go_ and preset == 'Custom':
        st.info('Fill in the session (or pick a preset) and press **Predict**.')
        return
    X = engineer_session_features(pd.DataFrame([{
        'Administrative': adm, 'Administrative_Duration': adm_d, 'Informational': inf, 'Informational_Duration': inf_d,
        'ProductRelated': prod, 'ProductRelated_Duration': prod_d, 'BounceRates': bounce, 'ExitRates': exit_r,
        'PageValues': pv, 'SpecialDay': special, 'Month': month, 'OperatingSystems': os_, 'Browser': br,
        'Region': reg, 'TrafficType': tt, 'VisitorType': vt, 'Weekend': weekend}]))
    rows = []
    for name, pipe in allm.items():
        pred = int(pipe.predict(X)[0])
        prob = float(pipe.predict_proba(X)[0, 1]) if data.has_proba(name) else np.nan
        rows.append({'Model': name, 'Prediction': 'Purchase' if pred else 'No purchase', 'Probability': prob})
    out = pd.DataFrame(rows)
    sel = out.set_index('Model').loc[model_name]

    a, b = st.columns([2, 3])
    with a:
        buy = sel['Prediction'] == 'Purchase'
        prob_txt = f"{sel['Probability']:.1%}" if not np.isnan(sel['Probability']) else ('BUY' if buy else 'NO BUY')
        T.result_box(prob_txt, (f'{model_name} · purchase probability',
                                '✅ Likely to purchase' if buy else '❌ Unlikely to purchase'),
                     *(('#059669', '#10B981') if buy else ('#B91C1C', '#EF4444')))
        votes = (out['Prediction'] == 'Purchase').sum()
        st.metric('Models voting "Purchase"', f'{votes} / 10')
        if np.isnan(sel['Probability']):
            st.caption('SVC was trained without probability calibration, so it gives a class decision only.')
    with b:
        o = out.dropna(subset=['Probability']).sort_values('Probability')
        fig = px.bar(o, x='Probability', y='Model', orientation='h', text='Probability',
                     color=o['Model'] == model_name, color_discrete_map={True: C1, False: '#CBD5E1'},
                     title='Purchase probability from every model')
        fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig.add_vline(x=0.5, line_dash='dash', line_color=T.MUTED, annotation_text='0.5')
        fig.add_vline(x=res['decision_threshold'], line_dash='dot', line_color=T.WARN,
                      annotation_text=f"{res['decision_threshold']:.2f}")
        fig.update_layout(showlegend=False)
        fig.update_xaxes(range=[0, 1.12], tickformat='.0%')
        T.chart(fig, 380)
    tips = []
    if pv == 0:
        tips.append('**No page value** — the visitor has not reached pages that usually precede checkout; only ~4 % of '
                    'such sessions buy.')
    if exit_r > 0.05:
        tips.append('**High exit rate** — visitors who leave pages quickly rarely buy.')
    if month in ('Nov', 'Dec'):
        tips.append('**Q4 session** — November has the highest conversion rate in the data (~25 %).')
    for t in tips:
        st.info(t)


# ----------------------------------------------------------------------------------------------- evaluate
def evaluate():
    scores, y = data.classification_test_scores()
    names = list(scores)
    res = data.results('classification')
    T.section('Evaluate any model on the test split', 'Computed live from the saved pipeline — 2,441 unseen sessions')
    c1, c2 = st.columns([2, 3])
    name = c1.selectbox('Model', names, index=names.index(res['best_model']), key='eval_model')
    s = scores[name]
    proba = data.has_proba(name)
    if proba:
        thr = c2.slider('Decision threshold', 0.05, 0.95, 0.50, 0.01,
                        help='Probability above which a session is classified as "Purchase".')
    else:
        thr = 0.0
        c2.info('SVC outputs a decision score; its threshold is fixed at 0 (equivalent to the default).')
    pred = (s >= thr).astype(int)
    T.cards([('Accuracy', f'{accuracy_score(y, pred):.4f}', 'share of correct predictions'),
             ('Weighted F1', f"{f1_score(y, pred, average='weighted'):.4f}", 'mandatory ranking metric'),
             ('Purchase recall', f'{recall_score(y, pred):.3f}', 'buyers caught'),
             ('Purchase precision', f'{precision_score(y, pred, zero_division=0):.3f}', 'alarms that were real'),
             ('ROC-AUC', f'{roc_auc_score(y, s):.4f}', 'threshold-free ranking')], C1)
    a, b, c = st.columns([1.1, 1.4, 1.4])
    with a:
        cm = confusion_matrix(y, pred)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', x=['No purchase', 'Purchase'],
                        y=['No purchase', 'Purchase'], labels=dict(x='Predicted', y='Actual', color='Sessions'),
                        title='Confusion matrix')
        fig.update_layout(coloraxis_showscale=False)
        T.chart(fig, 380)
    with b:
        fig = go.Figure()
        for n, sc in scores.items():
            fpr, tpr, _ = roc_curve(y, sc)
            fig.add_scatter(x=fpr, y=tpr, mode='lines', name=f'{n} ({roc_auc_score(y, sc):.3f})',
                            line=dict(width=3.5 if n == name else 1, color=C1 if n == name else '#CBD5E1'))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='#64748B'), name='random')
        fig.update_layout(title='ROC curve (selected model highlighted)', xaxis_title='False positive rate',
                          yaxis_title='True positive rate', showlegend=False)
        T.chart(fig, 380)
    with c:
        prec, rec, _ = precision_recall_curve(y, s)
        fig = go.Figure(go.Scatter(x=rec, y=prec, mode='lines', line=dict(color=C1, width=3), name=name))
        fig.add_hline(y=y.mean(), line_dash='dash', line_color='#64748B', annotation_text='no-skill')
        fig.update_layout(title=f'Precision–Recall (AP = {average_precision_score(y, s):.3f})', xaxis_title='Recall',
                          yaxis_title='Precision', showlegend=False)
        T.chart(fig, 380)
    if proba:
        grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
        curve = pd.DataFrame({'threshold': grid,
                              'Purchase precision': [precision_score(y, s >= t, zero_division=0) for t in grid],
                              'Purchase recall': [recall_score(y, s >= t) for t in grid],
                              'Purchase F1': [f1_score(y, s >= t) for t in grid]})
        fig = px.line(curve, x='threshold', y=['Purchase precision', 'Purchase recall', 'Purchase F1'],
                      title='How the threshold trades precision for recall')
        fig.add_vline(x=thr, line_dash='dash', line_color=T.INK, annotation_text=f'current {thr:.2f}')
        fig.update_yaxes(title='Score')
        T.chart(fig, 360)


# ----------------------------------------------------------------------------------------------- gallery
FIGS = [('cls_comparison_heatmap.png', 'All 10 classifiers × 5 metrics'),
        ('cls_cm_all.png', 'Confusion matrices — all 10 models'),
        ('cls_roc_pr_curves.png', 'ROC and Precision–Recall curves'),
        ('cls_tuning_gain.png', 'Weighted F1 before and after tuning'),
        ('cls_decision_tree.png', 'Decision tree (first three levels)'),
        ('cls_rf_tuning_importance.png', 'Random Forest — n_estimators and feature importance'),
        ('cls_gb_grid.png', 'Gradient Boosting tuning grid'),
        ('cls_mlp_tuning.png', 'MLP architecture search and learning curve'),
        ('cls_cv_top2.png', '5-fold CV of the top-2 models'),
        ('cls_learning_curve.png', 'Learning curve of the final model'),
        ('cls_permutation_importance.png', 'Permutation importance of the final model'),
        ('cls_threshold.png', 'Decision-threshold selection')]


def gallery():
    T.section('Figures from notebooks/classification.ipynb', 'Static exports of the notebook plots')
    cols = st.columns(2)
    for i, (f, cap) in enumerate(FIGS):
        p = data.figure_path(f)
        if p:
            cols[i % 2].image(p, caption=cap, width='stretch')


render()
