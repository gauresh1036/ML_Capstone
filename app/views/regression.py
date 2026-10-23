import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import data
import theme as T
from features import engineer_housing_features

C1, C2 = '#4F46E5', '#7C3AED'
QUAL = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
QUAL_LABEL = {'Po': 'Poor', 'Fa': 'Fair', 'TA': 'Typical', 'Gd': 'Good', 'Ex': 'Excellent'}


def render():
    T.inject_css()
    res = data.results('regression')
    df = data.housing_raw()
    T.hero('Track 1 · Regression', 'What is this house worth? 🏡',
           'Ten regressors predict the sale price of homes in Ames, Iowa from 79 property attributes '
           '(De Cock, 2011). The models learn log(price), so their errors are relative.',
           [f"Best: {res['best_model']}", f"R² {res['test_metrics']['R2']:.3f}",
            f"RMSE ${res['test_metrics']['RMSE']:,.0f}", f"MAE ${res['test_metrics']['MAE']:,.0f}"], C1, C2)
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
        evaluate(res)
    with tabs[5]:
        gallery()


def overview(res, df):
    cv = res['cv_top2'][res['best_model']]
    T.cards([('Houses', f'{len(df):,}', f"{res['n_rows_after_cleaning']:,} after removing 29 extreme sales"),
             ('Median price', f"${res['median_price']:,.0f}", 'right-skewed target → log transform'),
             ('Best test R²', f"{res['test_metrics']['R2']:.4f}", res['best_model']),
             ('5-fold CV R²', f"{cv['CV R² mean']:.4f}", f"± {cv['CV R² std']:.4f} across folds")], C1)
    T.section('The pipeline')
    T.steps([('Audit', '82 cols, 27 with NaN'), ('EDA', 'skew, correlation, outliers'),
             ('Clean', 'NaN = "None"; 29 outliers'), ('Engineer', 'TotalSF, HouseAge, …'),
             ('Encode', 'ordinal + one-hot'), ('Scale', 'StandardScaler (train)'),
             ('Tune', 'GridSearchCV, 5-fold'), ('Select', 'CV of top-2')])
    comp = pd.DataFrame(res['comparison'])
    T.section('Leaderboard', f"Tuned models on the same {res['n_test']}-house test split")
    a, b = st.columns(2)
    with a:
        fig = px.bar(comp.sort_values('R2'), x='R2', y='Model', orientation='h', text='R2', title='Test R²',
                     color=comp.sort_values('R2')['Model'] == res['best_model'],
                     color_discrete_map={True: C1, False: '#A5B4FC'})
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(showlegend=False)
        fig.update_xaxes(range=[0.8, 0.95])
        T.chart(fig, 420)
    with b:
        fig = px.bar(comp.sort_values('RMSE', ascending=False), x=['RMSE', 'MAE'], y='Model', orientation='h',
                     barmode='group', title='Test error (USD)', color_discrete_sequence=['#F97316', '#FDBA74'])
        fig.update_xaxes(title='USD')
        T.chart(fig, 420)
    st.success('**Key finding:** after modelling log(price), regularised linear models (ElasticNet, Lasso, linear SVR) '
               'match Gradient Boosting. In log space the price is close to additive in size, quality, age and location.')


def eda(df):
    T.section('Dataset at a glance')
    st.dataframe(df.head(200), height=300, width='stretch')
    a, b = st.columns(2)
    with a:
        miss = df.isna().mean().mul(100).sort_values(ascending=False)
        miss = miss[miss > 0].reset_index()
        miss.columns = ['Column', 'Missing %']
        fig = px.bar(miss, x='Missing %', y='Column', orientation='h', title='Missing values (27 columns)',
                     color='Missing %', color_continuous_scale='Purples')
        fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange='reversed'))
        T.chart(fig, 520)
    with b:
        logy = st.toggle('Show log(1 + SalePrice)', value=False)
        y = np.log1p(df['SalePrice']) if logy else df['SalePrice']
        fig = px.histogram(y, nbins=60, title=f"SalePrice distribution (skew = {y.skew():.2f})",
                           color_discrete_sequence=[C1])
        fig.update_layout(showlegend=False, xaxis_title='log(1 + price)' if logy else 'Sale price (USD)')
        T.chart(fig, 470)
    T.section('Feature vs price', 'Choose any numerical feature')
    num = [c for c in df.select_dtypes('number').columns if c not in ('Order', 'PID', 'SalePrice')]
    c1, c2 = st.columns([2, 1])
    feat = c1.selectbox('Feature', num, index=num.index('Gr Liv Area'))
    color = c2.selectbox('Colour by', ['None', 'Overall Qual', 'Neighborhood', 'Bldg Type', 'House Style'])
    fig = px.scatter(df, x=feat, y='SalePrice', color=None if color == 'None' else color, opacity=0.6,
                     title=f"{feat} vs SalePrice (r = {df[feat].corr(df['SalePrice']):.2f})",
                     color_continuous_scale='Viridis')
    ok = df[[feat, 'SalePrice']].dropna()
    xs = np.linspace(ok[feat].min(), ok[feat].max(), 50)
    fig.add_scatter(x=xs, y=np.polyval(np.polyfit(ok[feat], ok['SalePrice'], 1), xs), mode='lines',
                    name='linear trend', line=dict(color=T.BAD, width=3, dash='dash'))
    T.chart(fig, 460)
    T.section('Location matters')
    order = df.groupby('Neighborhood')['SalePrice'].median().sort_values(ascending=False).index
    fig = px.box(df, x='Neighborhood', y='SalePrice', category_orders={'Neighborhood': list(order)}, points=False,
                 color_discrete_sequence=[C1], title='SalePrice by neighbourhood (sorted by median)')
    T.chart(fig, 430)
    T.section('Correlation heatmap', 'The 15 numerical features most correlated with SalePrice')
    corr = df[num + ['SalePrice']].corr()
    top = corr['SalePrice'].abs().sort_values(ascending=False).head(16).index
    fig = px.imshow(corr.loc[top, top].round(2), text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                    aspect='auto')
    T.chart(fig, 600)


def models(res):
    comp = pd.DataFrame(res['comparison'])
    comp.index = np.arange(1, len(comp) + 1)
    g = pd.DataFrame(res['tuning_gains']).set_index('Model')
    comp['R² baseline'] = comp['Model'].map(g['R2 baseline'])
    comp['Best parameters'] = comp['Model'].map(g['Best parameters'])
    T.section('All 10 regressors — ranked by test R²', 'Same preprocessing, same split, log target for every model')
    st.dataframe(comp.style.format({'R2': '{:.4f}', 'R² baseline': '{:.4f}', 'RMSE': '${:,.0f}', 'MAE': '${:,.0f}'})
                 .background_gradient(subset=['R2'], cmap='Purples'), width='stretch', height=420)
    long = comp.melt(id_vars='Model', value_vars=['R² baseline', 'R2'], var_name='Stage', value_name='R²')
    long['Stage'] = long['Stage'].map({'R² baseline': 'Baseline (defaults)', 'R2': 'Tuned'})
    long['R²'] = long['R²'].clip(lower=0)
    fig = px.bar(long, x='Model', y='R²', color='Stage', barmode='group', title='Effect of GridSearchCV tuning',
                 color_discrete_sequence=['#CBD5E1', C1])
    T.chart(fig, 420)
    st.caption('Default Lasso / ElasticNet predict a constant (R² ≈ −0.04, shown as 0): their default penalty is far too '
               'large for a log-scale target. Tuning makes them top-3 models.')
    T.section('5-fold cross-validation of the top-2')
    cv = pd.DataFrame(res['cv_top2']).T
    st.dataframe(cv, width='stretch')


def lab(res):
    allm = data.model('regression_all_models.pkl')
    names = list(allm)
    d = res['feature_defaults']
    T.section('Model lab', 'Describe a house, choose any of the 10 models, and see what every model predicts. '
                           'Fields not shown keep the value of a typical Ames house.')
    model_name = st.selectbox('Model', names, index=names.index(res['best_model']))
    with st.form('reg_lab'):
        a, b, c = st.columns(3)
        with a:
            st.markdown('**Size**')
            first = st.number_input('1st floor (sq ft)', 300, 5000, int(d['1st Flr SF']), 50)
            second = st.number_input('2nd floor (sq ft)', 0, 3000, int(d['2nd Flr SF']), 50)
            bsmt = st.number_input('Basement (sq ft)', 0, 4000, int(d['Total Bsmt SF']), 50)
            lot = st.number_input('Lot area (sq ft)', 1000, 100000, int(d['Lot Area']), 250)
            cars = st.slider('Garage capacity (cars)', 0, 4, int(d['Garage Cars']))
        with b:
            st.markdown('**Quality & age**')
            qual = st.slider('Overall quality (1–10)', 1, 10, int(d['Overall Qual']))
            cond = st.slider('Overall condition (1–10)', 1, 10, int(d['Overall Cond']))
            kitchen = st.select_slider('Kitchen quality', QUAL, d['Kitchen Qual'], format_func=QUAL_LABEL.get)
            exter = st.select_slider('Exterior quality', QUAL, d['Exter Qual'], format_func=QUAL_LABEL.get)
            built = st.slider('Year built', 1872, 2010, int(d['Year Built']))
            remod = st.slider('Year remodelled', 1950, 2010, max(int(d['Year Remod/Add']), 1950))
        with c:
            st.markdown('**Location & amenities**')
            hoods = res['neighborhoods']
            hood = st.selectbox('Neighbourhood', hoods, hoods.index(d['Neighborhood']))
            full = st.slider('Full bathrooms', 0, 4, int(d['Full Bath']))
            half = st.slider('Half bathrooms', 0, 2, int(d['Half Bath']))
            fire = st.slider('Fireplaces', 0, 3, int(d['Fireplaces']))
            air = st.radio('Central air', ['Y', 'N'], horizontal=True)
        go_ = st.form_submit_button('⚡ Estimate with all 10 models', type='primary', width='stretch')
    if not go_:
        st.info('Adjust the house and press **Estimate**.')
        return
    row = dict(d)
    row.update({'1st Flr SF': first, '2nd Flr SF': second, 'Gr Liv Area': first + second, 'Total Bsmt SF': bsmt,
                'Lot Area': lot, 'Garage Cars': cars, 'Garage Area': cars * 260.0, 'Overall Qual': qual,
                'Overall Cond': cond, 'Kitchen Qual': kitchen, 'Exter Qual': exter, 'Year Built': built,
                'Year Remod/Add': max(remod, built), 'Neighborhood': hood, 'Full Bath': full, 'Half Bath': half,
                'Fireplaces': fire, 'Central Air': air,
                'Garage Type': row['Garage Type'] if cars else None, 'Garage Finish': row['Garage Finish'] if cars else None})
    X = engineer_housing_features(pd.DataFrame([row]))
    preds = pd.DataFrame([{'Model': n, 'Price': float(p.predict(X)[0])} for n, p in allm.items()])
    price = preds.set_index('Model').loc[model_name, 'Price']
    rmse = {r['Model']: r['RMSE'] for r in res['comparison']}[model_name]
    a, b = st.columns([2, 3])
    with a:
        T.result_box(f'${price:,.0f}', (f'{model_name} · estimated sale price',
                                        f'typical error ± ${rmse:,.0f} (test RMSE)'), C1, C2)
        st.metric('Spread across the 10 models', f"${preds['Price'].min():,.0f} – ${preds['Price'].max():,.0f}")
        st.metric('vs Ames median', f'${price - res["median_price"]:+,.0f}')
    with b:
        o = preds.sort_values('Price')
        fig = px.bar(o, x='Price', y='Model', orientation='h', text='Price', title='Estimate from every model',
                     color=o['Model'] == model_name, color_discrete_map={True: C1, False: '#C7D2FE'})
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig.update_layout(showlegend=False)
        fig.update_xaxes(range=[0, o['Price'].max() * 1.22], title='USD')
        T.chart(fig, 400)


def evaluate(res):
    preds, y = data.regression_test_predictions()
    names = list(preds)
    T.section('Evaluate any model on the test split', f'Computed live — {len(y)} houses never seen in training')
    name = st.selectbox('Model', names, index=names.index(res['best_model']), key='reg_eval')
    p = preds[name]
    err = np.abs(y - p) / y * 100
    T.cards([('R²', f'{r2_score(y, p):.4f}', 'share of price variance explained'),
             ('RMSE', f'${np.sqrt(mean_squared_error(y, p)):,.0f}', 'penalises large misses'),
             ('MAE', f'${mean_absolute_error(y, p):,.0f}', 'typical miss in dollars'),
             ('Median % error', f'{np.median(err):.1f}%', f'90 % of houses within ±{np.percentile(err, 90):.1f}%')], C1)
    a, b = st.columns(2)
    with a:
        fig = px.scatter(x=y, y=p, opacity=0.6, labels={'x': 'Actual price (USD)', 'y': 'Predicted price (USD)'},
                         title='Predicted vs actual', color_discrete_sequence=[C1])
        lim = [min(y.min(), p.min()), max(y.max(), p.max())]
        fig.add_scatter(x=lim, y=lim, mode='lines', line=dict(dash='dash', color=T.BAD), name='perfect')
        fig.update_layout(showlegend=False)
        T.chart(fig, 430)
    with b:
        fig = px.scatter(x=p, y=y - p, opacity=0.6, labels={'x': 'Predicted price (USD)', 'y': 'Residual (USD)'},
                         title='Residuals vs predicted', color_discrete_sequence=['#F97316'])
        fig.add_hline(y=0, line_dash='dash', line_color=T.BAD)
        T.chart(fig, 430)
    band = pd.cut(y, [0, 120000, 160000, 220000, 1e7], labels=['< $120k', '$120–160k', '$160–220k', '> $220k'])
    by = pd.DataFrame({'band': band, 'err': err}).groupby('band', observed=True)['err'].median().reset_index()
    fig = px.bar(by, x='band', y='err', text='err', title='Median % error by price band',
                 color_discrete_sequence=[C2], labels={'band': 'Actual price band', 'err': 'Median absolute % error'})
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    T.chart(fig, 340)


FIGS = [('reg_model_comparison.png', 'Tuned vs baseline R², RMSE and MAE'),
        ('reg_residual_diagnostics.png', 'Residual diagnostics of the best model'),
        ('reg_feature_importance.png', 'Gradient Boosting and permutation importance'),
        ('reg_lasso_sparsity.png', 'Lasso sparsity vs alpha'),
        ('reg_elasticnet_grid.png', 'ElasticNet tuning grid'),
        ('reg_polynomial_degrees.png', 'Polynomial degree comparison'),
        ('reg_decision_tree.png', 'Decision tree depth and importance'),
        ('reg_gradient_boosting_grid.png', 'Gradient Boosting tuning grid'),
        ('reg_svr_grid.png', 'SVR: kernel and C'),
        ('reg_knn_curve.png', 'KNN: k'),
        ('reg_cv_top2.png', '5-fold CV of the top-2'),
        ('reg_correlation_heatmap.png', 'Correlation heatmap')]


def gallery():
    T.section('Figures from notebooks/regression.ipynb')
    cols = st.columns(2)
    for i, (f, cap) in enumerate(FIGS):
        p = data.figure_path(f)
        if p:
            cols[i % 2].image(p, caption=cap, width='stretch')


render()
