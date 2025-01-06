import pandas as pd
import streamlit as st

# Verisetinin GitHub URL'si
DATA_URL = "https://raw.githubusercontent.com/donayy/inka/main/movies_short.csv"



@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

df = load_data()
st.write("Veri başarıyla yüklendi!")
st.dataframe(df.head())



