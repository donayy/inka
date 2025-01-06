import pandas as pd
import streamlit as st

# Verisetinin GitHub URL'si
DATA_URL = "https://raw.githubusercontent.com/donayy/inka/main/movies_short.csv"

# Veriyi yükleyen fonksiyon
@st.cache_data
def load_data(url):
    try:
        data = pd.read_csv(url)
        return data
    except Exception as e:
        st.error(f"Veri yüklenirken bir hata oluştu: {e}")
        return None

# Veriyi yükle
df = load_data(DATA_URL)

# Eğer veri başarıyla yüklendiyse göster
if df is not None:
    st.write("Veri başarıyla yüklendi!")
    st.dataframe(df.head())
else:
    st.error("Veri yüklenemedi, lütfen bağlantıyı kontrol edin!")




