import pandas as pd
import streamlit as st

# Verisetini GitHub üzerinden yükleme
DATA_URL = "https://raw.githubusercontent.com/donay/inka/main/movies_short.xlsx"

@st.cache_data
def load_data():
    return pd.read_excel(DATA_URL)

df = load_data()
st.write("Veri başarıyla yüklendi!")
st.dataframe(df.head())


