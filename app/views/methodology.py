import streamlit as st

import data
import theme as T


def render():
    T.inject_css()
    T.hero('Methodology & project', 'How the work was done',
           'Design decisions, evaluation protocol, rubric coverage and reproducibility.',
           ['23CSE301 · AY 2026–27', 'B.Tech CSE · III Year'], '#0F172A', '#4F46E5')
    a, b = st.columns(2)
    with a:
        T.section('Evaluation protocol')
        st.markdown("""
- **One split per track** (80 : 20, `random_state = 42`; stratified for classification). Every model sees the same
  test rows.
- **No leakage:** imputers, encoders and scalers sit in a `ColumnTransformer` fitted on the training split only.
  Cross-validation re-fits the whole pipeline inside every fold.
- **Honest tuning:** `GridSearchCV` with 5-fold CV on the training split only. The test split is used once, for the
  final comparison.
- **Baseline vs tuned** scores are reported for every algorithm, so the improvement from tuning is documented.
- **Top-2 models** of each supervised track are re-checked with 5-fold cross-validation before the final choice.
""")
        T.section('Key preprocessing decisions')
        st.markdown("""
| Track | Decision | Why |
|---|---|---|
| Regression | model `log(1 + price)` | right-skewed target; fixed SVR (R² −0.04 → 0.93) |
| Regression | NaN → `'None'` for pool/garage/basement | missing = feature absent |
| Classification | `log1p` + scaling of counts/durations | skew 2–7.6 |
| Classification | one-hot `Browser`, `Region`, … | nominal codes, no order |
| Classification | `HasPageValue` feature | 56 % vs 4 % conversion |
| Clustering | remove cancellations, fees, anonymous lines | RFM must describe real purchases |
| Clustering | `log1p` + scaling of RFM | Monetary skew 19.6 → 0.4 |
""")
    with b:
        T.section('Rubric coverage — Review 2')
        st.markdown("""
| Criterion | Where to find it |
|---|---|
| A1 — 5 Part-B classifiers | Classification → All 10 models |
| A2 — table of all 10 classifiers | Classification → All 10 models |
| A3 — final model + tuning gain | Classification → Overview / All 10 models |
| B1 — K-Means + Agglomerative | Clustering → K-Means vs Agglomerative |
| B2 — Silhouette, DB, CH, Elbow, Dendrogram | Clustering → Overview / Choosing k |
| B3 — PCA for both, t-SNE | Clustering → K-Means vs Agglomerative |
| C1–C3 — code, README, GitHub | `notebooks/`, `README.md`, commit history |
| Bonus — GUI + deployment | this app |
""")
        T.section('Reproduce everything')
        st.code('pip install -r requirements.txt\n'
                'python -m nbconvert --to notebook --execute --inplace notebooks/regression.ipynb\n'
                'python -m nbconvert --to notebook --execute --inplace notebooks/classification.ipynb\n'
                'python -m nbconvert --to notebook --execute --inplace notebooks/clustering.ipynb\n'
                'python -m pytest tests -q\n'
                'streamlit run app/app.py', language='bash')
        st.caption('The notebooks write models/*.pkl and reports/*.json; this app only reads them, so it always shows the '
                   'notebooks\' results.')
    T.section('Datasets & references')
    st.markdown("""
1. De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing Data.* Journal of Statistics Education 19(3).
2. Sakar, C.O. *et al.* (2019). *Real-time prediction of online shoppers' purchasing intention.* Neural Computing and Applications 31.
3. Chen, D., Sain, S.L. & Guo, K. (2012). *RFM model-based customer segmentation using data mining.* JDMCSM 19.
4. Pedregosa, F. *et al.* (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12.
""")


render()
