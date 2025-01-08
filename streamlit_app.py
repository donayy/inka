import pandas as pd
import streamlit as st

# Verisetinin GitHub URL'si
DATA_URL = "https://raw.githubusercontent.com/donayy/inka/main/movies_short.csv"

@st.cache_data
def load_data(url):
    try:
        data = pd.read_csv(url, encoding="utf-8", error_bad_lines=False, warn_bad_lines=True)  # Hatalı satırları atlar
        return data
    except Exception as e:
        st.error(f"Veri yüklenirken bir hata oluştu: {e}")
        return None

st.title("Inka ve Chill")

st.write("Veri yükleniyor...")
df = load_data(DATA_URL)

if df is not None:
    st.write("Veri başarıyla yüklendi!")
    st.dataframe(df.head())
else:
    st.error("Veri yüklenemedi, lütfen bağlantıyı kontrol edin!")
