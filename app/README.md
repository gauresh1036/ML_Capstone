# Interactive GUI — Streamlit (Review 2 bonus)

A full interactive showcase of the project:

| Page | Tabs |
|------|------|
| **Home** | project overview, the three tracks, pipeline, all 20 supervised models at a glance |
| **Regression — house prices** | Overview · Explore the data · All 10 models · Model lab · Evaluate a model · Notebook figures |
| **Classification — purchase intent** | Overview · Explore the data · All 10 models · Model lab · Evaluate a model (threshold slider) · Notebook figures |
| **Clustering — customer segments** | Overview · Explore the data · Choosing k · K-Means vs Agglomerative (PCA / t-SNE / 3-D) · Segments · Segment finder · Notebook figures |
| **Methodology** | evaluation protocol, key decisions, rubric coverage, how to reproduce |

**Model lab** lets you enter your own house / session and get predictions from the model you choose *and* from all 10
models side by side. **Evaluate a model** recomputes the confusion matrix, ROC/PR curves or predicted-vs-actual plots
live on the notebooks' test split for any of the 10 models.

Code layout: `app.py` (navigation) · `views/` (one file per page) · `data.py` (cached loaders and test-split
reconstruction) · `theme.py` (design system) · `features.py` (feature engineering mirrored from the notebooks).

The app contains **no hard-coded results**. Every number is read from the JSON files the notebooks write, and every
prediction uses the exact pipelines the notebooks save. `features.py` mirrors the notebooks' feature-engineering
functions, and `tests/test_models.py` checks that the two copies stay identical and that the saved models reproduce
the notebook test scores.

## Run locally

```bash
pip install -r requirements.txt        # from the repository root
streamlit run app/app.py
```

Then open <http://localhost:8501>.

**Live app:** https://gauresh1036.streamlit.app

## Deploy publicly (Streamlit Community Cloud — free)

1. Push the repository to GitHub (public repository).
2. Sign in at <https://share.streamlit.io> with GitHub → **Create app** → **Deploy a public app from GitHub**.
3. Repository: `<your-user>/ML_Capstone`, branch `main`, main file path **`app/app.py`**.
4. Under **Advanced settings** choose **Python 3.13**. The dependencies in `app/requirements.txt` are installed
   automatically.
5. Click **Deploy**. Add the public URL to the main README.

> The model files were pickled with scikit-learn 1.8.0; keep that version pinned or the pickles will not load.
