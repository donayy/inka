import pandas as pd
import streamlit as st

# Verisetinin GitHub URL'si
DATA_URL = "https://raw.githubusercontent.com/donayy/inka/refs/heads/main/movie_short_f.csv"

@st.cache_data
def load_data():
    # Hatalı satırları atlamak için on_bad_lines kullanıyoruz
    return pd.read_csv(DATA_URL, on_bad_lines="skip")

try:
    df = load_data()
    st.write("Veri başarıyla yüklendi!")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"Veri yüklenirken bir hata oluştu: {e}")

