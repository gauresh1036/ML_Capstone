import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data
import theme as T

C1, C2 = '#7C3AED', '#DB2777'
SEG_COLORS = {'Champions': '#7C3AED', 'Loyal Customers': '#0891B2', 'At-Risk': '#F59E0B',
              'New / Promising': '#10B981', 'Lost / Hibernating': '#94A3B8'}
ACTIONS = {
    'Champions': 'Reward and retain — VIP tier, early access to new ranges, account manager for big wholesalers; avoid unnecessary discounts.',
    'Loyal Customers': 'Grow the basket — bundles, volume pricing, personalised cross-sell from past categories.',
    'At-Risk': 'Win back before they lapse — personalised reminder and an offer on their usual products; ask for feedback.',
    'New / Promising': 'Secure the second order — onboarding emails, time-limited second-order incentive, best-sellers.',
    'Lost / Hibernating': 'Reactivate cheaply or let go — seasonal newsletter before Q4 only; do not spend heavily.',
}
RFM = ['Recency', 'Frequency', 'Monetary']


def render():
    T.inject_css()
    res = data.results('clustering')
    km = next(v for k, v in res['metrics'].items() if k.startswith('K-Means'))
    T.hero('Track 3 · Clustering (unsupervised)', 'Who are our customers? 👥',
           f"K-Means and Agglomerative clustering segment {res['n_customers']:,} customers of a UK online gift retailer "
           f"by Recency, Frequency and Monetary value — {res['n_transactions_raw']:,} invoice lines, no labels.",
           [f"K-Means k = {res['k']}", f"Silhouette {km['Silhouette']:.3f}", f"Davies–Bouldin {km['Davies-Bouldin']:.3f}",
            f"PCA keeps {sum(res['pca_explained_variance']):.1%} of variance"], C1, C2)
    tabs = st.tabs(['📌 Overview', '🔍 Explore the data', '📐 Choosing k', '⚖️ K-Means vs Agglomerative',
                    '🧩 Segments', '🧭 Segment finder', '🖼️ Notebook figures'])
    with tabs[0]:
        overview(res)
    with tabs[1]:
        eda()
    with tabs[2]:
        choose_k(res)
    with tabs[3]:
        algorithms(res)
    with tabs[4]:
        segments(res)
    with tabs[5]:
        finder(res)
    with tabs[6]:
        gallery()


def overview(res):
    seg = pd.DataFrame(res['segments'])
    champ = seg.set_index('Segment').loc['Champions']
    T.cards([('Invoice lines', f"{res['n_transactions_raw']:,}", f"{res['n_transactions_clean']:,} after cleaning"),
             ('Customers', f"{res['n_customers']:,}", 'one RFM row each'),
             ('Segments', str(res['k']), 'elbow + Davies–Bouldin + business check'),
             ('Champions', f"{champ['Customer_%']:.1f}% → {champ['Revenue_%']:.1f}%", 'of customers → of revenue')], C1)
    T.section('The pipeline')
    T.steps([('Audit', '541k lines, 25 % no ID'), ('Clean', 'cancellations, fees, dups'), ('RFM', 'per customer'),
             ('log1p', 'tame heavy skew'), ('Scale', 'StandardScaler'), ('Choose k', 'elbow & metrics'),
             ('Cluster', 'K-Means + Ward'), ('Profile', 'name & act')])
    T.section('Evaluation (mandatory metrics)')
    m = pd.DataFrame(res['metrics']).T
    st.dataframe(m.style.format({'Silhouette': '{:.4f}', 'Davies-Bouldin': '{:.4f}', 'Calinski-Harabasz': '{:,.1f}'})
                 .highlight_max(subset=['Silhouette', 'Calinski-Harabasz'], color='#DDD6FE')
                 .highlight_min(subset=['Davies-Bouldin'], color='#DDD6FE'), width='stretch')
    st.caption('Silhouette ↑ (cohesion vs separation) · Davies–Bouldin ↓ (cluster similarity) · '
               'Calinski–Harabasz ↑ (between / within dispersion). K-Means wins on all three.')
    fig = px.treemap(seg, path=['Segment'], values='Monetary_total', color='Segment', color_discrete_map=SEG_COLORS,
                     title='Where the revenue comes from (area = total spend)')
    fig.update_traces(texttemplate='%{label}<br>%{percentRoot:.1%} of revenue')
    T.chart(fig, 380)


def eda():
    e = data.clustering_eda()
    T.section('Data quality and cleaning')
    a, b = st.columns(2)
    with a:
        q = pd.DataFrame(e['quality'])
        fig = px.bar(q, x='rows', y='Issue', orientation='h', text='rows', title='Raw-data issues (invoice lines)',
                     color_discrete_sequence=[C2])
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        T.chart(fig, 330)
    with b:
        log = pd.DataFrame(e['cleaning_log'])
        fig = go.Figure(go.Funnel(y=[s[:45] for s in log['step']], x=log['rows'], marker_color=C1,
                                  textinfo='value+percent initial'))
        fig.update_layout(title='Cleaning funnel')
        T.chart(fig, 330)
    T.section('When and where customers buy')
    m = pd.DataFrame(e['monthly'])
    fig = go.Figure()
    fig.add_bar(x=m['Month'], y=m['InvoiceNo'], name='Invoices', marker_color='#DDD6FE')
    fig.add_scatter(x=m['Month'], y=m['Revenue'], name='Revenue (£)', yaxis='y2', mode='lines+markers',
                    line=dict(color=C1, width=3))
    fig.update_layout(title='Monthly invoices and revenue (Dec-11 is partial)', yaxis_title='Invoices',
                      yaxis2=dict(title='Revenue (£)', overlaying='y', side='right', showgrid=False),
                      legend=dict(orientation='h', y=1.12))
    T.chart(fig, 380)
    a, b = st.columns(2)
    with a:
        c = pd.DataFrame(e['countries']).head(10)
        fig = px.bar(c, x='revenue', y='Country', orientation='h', log_x=True, title='Revenue by country (log scale)',
                     color_discrete_sequence=[C1])
        fig.update_layout(yaxis=dict(autorange='reversed'), xaxis_title='Revenue (£, log)')
        T.chart(fig, 380)
    with b:
        h = e['weekday_hour']
        fig = px.imshow(np.array(h['values']), x=[f'{c}:00' for c in h['columns']], y=h['index'], aspect='auto',
                        color_continuous_scale='Purples', title='Invoices by weekday × hour (no Saturdays — B2B)')
        T.chart(fig, 380)
    T.section('RFM features', 'Raw values are extremely skewed; the log transform makes them usable for K-Means')
    cust = data.customers()
    c1, c2 = st.columns([1, 3])
    feat = c1.radio('Feature', RFM)
    logx = c1.toggle('log1p', value=True, key='clu_log')
    vals = np.log1p(cust[feat]) if logx else cust[feat]
    fig = px.histogram(vals, nbins=60, title=f'{feat} distribution (skew {vals.skew():.2f})',
                       color_discrete_sequence=[C1])
    fig.update_layout(showlegend=False, xaxis_title=f'log(1 + {feat})' if logx else feat)
    with c2:
        T.chart(fig, 340)


def choose_k(res):
    sw = pd.DataFrame(res['k_selection'])
    T.section('Elbow curve and metric sweep (k = 2 … 10)')
    a, b = st.columns(2)
    with a:
        fig = px.line(sw, x='k', y='Inertia', markers=True, title='Elbow curve — inertia vs k',
                      color_discrete_sequence=[C1])
        fig.add_vline(x=res['k'], line_dash='dash', annotation_text=f"elbow k = {res['k']}")
        T.chart(fig, 360)
    with b:
        metric = st.radio('Metric', ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz'], horizontal=True)
        fig = px.line(sw, x='k', y=metric, markers=True, title=f'{metric} vs k', color_discrete_sequence=[C2])
        fig.add_vline(x=res['k'], line_dash='dash')
        T.chart(fig, 310)
    st.dataframe(sw.set_index('k').style.format({'Inertia': '{:,.1f}', 'Silhouette': '{:.4f}',
                                                  'Davies-Bouldin': '{:.4f}', 'Calinski-Harabasz': '{:,.1f}',
                                                  'Smallest cluster %': '{:.1f}'}), width='stretch')
    st.info('**Why k = 5:** it is the knee of the inertia curve, a local Davies–Bouldin minimum, and the first k that '
            'separates the Champions (8 % of customers, 54 % of revenue) from ordinary loyal buyers. Silhouette peaks '
            'at k = 2, but "good vs rest" is not actionable.')


def algorithms(res):
    cust = data.customers()
    c1, c2, c3 = st.columns(3)
    algo = c1.radio('Algorithm', ['KMeans', 'Agglomerative'], horizontal=True,
                    format_func=lambda a: 'K-Means' if a == 'KMeans' else 'Agglomerative (Ward)')
    view = c2.radio('Projection', ['PCA (2D)', 't-SNE (2D)', 'RFM (3D)'], horizontal=True)
    sample = c3.slider('Customers shown', 500, len(cust), min(2500, len(cust)), 500)
    d = cust.sample(sample, random_state=0).assign(Cluster=lambda x: 'Cluster ' + x[algo].astype(str))
    hover = {'CustomerID': True, 'Recency': True, 'Frequency': True, 'Monetary': ':.0f'}
    if view == 'PCA (2D)':
        ev = res['pca_explained_variance']
        fig = px.scatter(d, x='PC1', y='PC2', color='Cluster', hover_data=hover, opacity=0.7,
                         labels={'PC1': f'PC1 ({ev[0]:.1%})', 'PC2': f'PC2 ({ev[1]:.1%})'},
                         category_orders={'Cluster': sorted(d['Cluster'].unique())})
    elif view == 't-SNE (2D)':
        fig = px.scatter(d, x='TSNE1', y='TSNE2', color='Cluster', hover_data=hover, opacity=0.7,
                         category_orders={'Cluster': sorted(d['Cluster'].unique())})
    else:
        fig = px.scatter_3d(d, x='Recency', y='Frequency', z='Monetary', color='Cluster', log_x=True, log_y=True,
                            log_z=True, opacity=0.7, hover_data=hover,
                            category_orders={'Cluster': sorted(d['Cluster'].unique())})
        fig.update_traces(marker_size=3)
    fig.update_layout(title=f"{'K-Means' if algo == 'KMeans' else 'Agglomerative (Ward)'} clusters — {view}")
    T.chart(fig, 560)
    a, b = st.columns(2)
    with a:
        T.section('Do the two algorithms agree?', f"Adjusted Rand Index = {res['agreement']['ARI']:.3f}")
        ct = pd.crosstab(cust['KMeans'], cust['Agglomerative'])
        fig = px.imshow(ct, text_auto=True, color_continuous_scale='Purples', aspect='auto',
                        labels=dict(x='Agglomerative cluster', y='K-Means cluster', color='Customers'))
        T.chart(fig, 380)
    with b:
        T.section('Linkage strategies (k = 5)', 'Good scores can hide useless, lopsided clusters')
        lk = pd.DataFrame(res['linkage_comparison']).set_index('Linkage')
        st.dataframe(lk.style.format({'Silhouette': '{:.4f}', 'Davies-Bouldin': '{:.4f}',
                                      'Calinski-Harabasz': '{:,.1f}', 'Largest cluster %': '{:.1f}'}), width='stretch')
        st.caption('Single and average linkage put 62–100 % of customers into one cluster (chaining). Ward is the only '
                   'balanced linkage, so it is the one compared with K-Means.')
    p = data.figure_path('clu_dendrogram_ward.png')
    if p:
        st.image(p, caption='Dendrogram (Ward) with the cut that gives 5 clusters', width='stretch')


def segments(res):
    seg = pd.DataFrame(res['segments']).sort_values('Revenue_%', ascending=False)
    T.section('Five customer segments', 'Medians in original units · K-Means, k = 5')
    cols = st.columns(len(seg))
    for col, (_, r) in zip(cols, seg.iterrows()):
        col.markdown(f"""<div class="track" style="--c:{SEG_COLORS.get(r['Segment'], C1)}">
<h3>{r['Segment']}</h3><div class="q">{int(r['Customers']):,} customers · {r['Customer_%']:.1f}%</div>
<div class="kv"><span>Revenue share</span><b>{r['Revenue_%']:.1f}%</b></div>
<div class="kv"><span>Last purchase</span><b>{r['Recency_median']:.0f} days</b></div>
<div class="kv"><span>Orders</span><b>{r['Frequency_median']:.0f}</b></div>
<div class="kv"><span>Spend</span><b>£{r['Monetary_median']:,.0f}</b></div>
<div class="kv"><span>Tenure</span><b>{r['Tenure_median']:.0f} days</b></div></div>""", unsafe_allow_html=True)
    a, b = st.columns(2)
    cust = data.customers()
    with a:
        z = np.log1p(cust[RFM])
        z = (z - z.mean()) / z.std()
        snake = z.assign(Segment=cust['Segment']).groupby('Segment')[RFM].mean().reset_index().melt(
            id_vars='Segment', var_name='Feature', value_name='z')
        fig = px.line(snake, x='Feature', y='z', color='Segment', markers=True, color_discrete_map=SEG_COLORS,
                      title='Snake plot — standardised log-RFM')
        fig.add_hline(y=0, line_color='#94A3B8')
        T.chart(fig, 400)
    with b:
        long = seg.melt(id_vars='Segment', value_vars=['Customer_%', 'Revenue_%'], var_name='Share', value_name='%')
        long['Share'] = long['Share'].map({'Customer_%': '% of customers', 'Revenue_%': '% of revenue'})
        fig = px.bar(long, x='Segment', y='%', color='Share', barmode='group', title='Customers vs revenue',
                     color_discrete_sequence=['#CBD5E1', C1])
        T.chart(fig, 400)
    feat = st.radio('Compare segments on', RFM, horizontal=True)
    fig = px.box(cust, x='Segment', y=feat, color='Segment', color_discrete_map=SEG_COLORS, log_y=True, points=False,
                 title=f'{feat} by segment (log scale)')
    fig.update_layout(showlegend=False)
    T.chart(fig, 380)
    st.dataframe(pd.DataFrame({'Segment': list(ACTIONS), 'Recommended action': list(ACTIONS.values())}),
                 width='stretch', hide_index=True)


def finder(res):
    art = data.model('clustering_kmeans.pkl')
    pipe, pca, names = art['pipeline'], art['pca'], art['segment_names']
    T.section('Segment finder', 'Enter a customer’s RFM values — the saved K-Means pipeline assigns a segment')
    presets = {'Custom': (30, 4, 1500.0), '👑 VIP wholesaler': (5, 25, 9000.0), '🌱 New customer': (20, 1, 300.0),
               '⚠️ Lapsing regular': (120, 4, 1200.0), '💤 Long gone': (300, 1, 150.0)}
    pick = st.radio('Preset', list(presets), horizontal=True)
    r0, f0, m0 = presets[pick]
    with st.form('seg'):
        a, b, c = st.columns(3)
        r = a.number_input('Recency — days since last purchase', 1, 400, r0)
        f = b.number_input('Frequency — number of orders', 1, 250, f0)
        m = c.number_input('Monetary — total spend (£)', 1.0, 300000.0, m0, 50.0)
        st.form_submit_button('🧭 Find segment', type='primary', width='stretch')
    x = pd.DataFrame([[r, f, m]], columns=art['features'])
    cid = int(pipe.predict(x)[0])
    seg = names[cid]
    pc = pca.transform(pipe[:-1].transform(x))[0]
    color = SEG_COLORS.get(seg, C1)
    a, b = st.columns([2, 3])
    with a:
        T.result_box(seg, ('This customer belongs to', f'cluster {cid}'), color, '#1E1B4B')
        st.markdown(f'**Recommended action:** {ACTIONS.get(seg, "")}')
        prof = pd.DataFrame(res['segments']).set_index('Segment').loc[seg]
        st.caption(f"Typical {seg}: {prof['Recency_median']:.0f} days since last order, "
                   f"{prof['Frequency_median']:.0f} orders, £{prof['Monetary_median']:,.0f} spent.")
    with b:
        cust = data.customers().sample(2000, random_state=0)
        fig = px.scatter(cust, x='PC1', y='PC2', color='Segment', opacity=0.35, color_discrete_map=SEG_COLORS,
                         title='Position on the customer map (PCA)')
        fig.add_scatter(x=[pc[0]], y=[pc[1]], mode='markers', name='This customer',
                        marker=dict(symbol='star', size=24, color='black', line=dict(color='white', width=2)))
        T.chart(fig, 430)


FIGS = [('clu_elbow_metrics.png', 'Elbow curve and metrics vs k'), ('clu_silhouette_kmeans.png', 'Silhouette plot'),
        ('clu_dendrogram_linkages.png', 'Dendrograms for four linkages'), ('clu_pca_clusters.png', 'PCA projection'),
        ('clu_tsne_clusters.png', 't-SNE projection'), ('clu_metrics_comparison.png', 'Metric comparison'),
        ('clu_segment_profiles.png', 'Segment profiles'), ('clu_rfm_score_check.png', 'Post-hoc RFM score check'),
        ('clu_rfm_distributions.png', 'RFM before / after log'), ('clu_agreement.png', 'Algorithm agreement')]


def gallery():
    T.section('Figures from notebooks/clustering.ipynb')
    cols = st.columns(2)
    for i, (f, cap) in enumerate(FIGS):
        p = data.figure_path(f)
        if p:
            cols[i % 2].image(p, caption=cap, width='stretch')


render()
