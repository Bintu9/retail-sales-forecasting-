import streamlit as st
import pandas as pd
import pickle
import datetime
import matplotlib.pyplot as plt

model_path = r"C:\Users\Bintu\Downloads\model_sales.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Retail Sales Forecasting", layout="wide")
st.title("📈 Retail Sales Forecasting Dashboard")

uploaded_file = st.file_uploader("Upload your sales CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data['Date'] = pd.to_datetime(data['Date'])
    st.dataframe(data.head())
