import streamlit as st
from churn_page import show_churn_page
from cluster_page import show_cluster_page


page = st.sidebar.selectbox("Churn Predict Or Cluster Predict", ("Churn", "Segmentation"))

if page == "Churn": 
    show_churn_page()
else:
    show_cluster_page()