import pandas as pd
import streamlit as st

# Verisetinin GitHub URL'si
DATA_URL = "https://raw.githubusercontent.com/<kullanıcı_adınız>/<depo_adınız>/main/movies_new.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

df = load_data()
st.write("Veri başarıyla yüklendi!")
st.dataframe(df.head())



