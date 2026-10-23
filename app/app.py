"""ML Capstone (23CSE301) — interactive showcase of the three tracks.

Run from the repository root:
    streamlit run app/app.py

Every number is read from reports/*.json and every prediction comes from models/*.pkl,
both written by the notebooks — nothing is hard-coded.
"""
import streamlit as st

st.set_page_config(page_title='ML Capstone · 23CSE301', page_icon='📊', layout='wide',
                   initial_sidebar_state='expanded')

pages = st.navigation({
    'Project': [st.Page('views/home.py', title='Home', icon='🏠', default=True)],
    'Tracks': [st.Page('views/regression.py', title='Regression — house prices', icon='🏡', url_path='regression'),
               st.Page('views/classification.py', title='Classification — purchase intent', icon='🛒',
                       url_path='classification'),
               st.Page('views/clustering.py', title='Clustering — customer segments', icon='👥', url_path='clustering')],
    'About': [st.Page('views/methodology.py', title='Methodology', icon='📘', url_path='methodology')],
})
with st.sidebar:
    st.markdown('### 📊 ML Capstone')
    st.caption('23CSE301 Machine Learning · Review 2')
pages.run()
