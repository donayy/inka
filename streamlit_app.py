import pandas as pd
import numpy as np
import streamlit as st

# Verisetinin GitHub URL'si
DATA_URL = "https://raw.githubusercontent.com/donayy/inka/refs/heads/main/movie_short_f.csv"

@st.cache_data
def load_data():
    # Hatalı satırları atlamak için on_bad_lines kullanıyoruz
    return pd.read_csv(DATA_URL, on_bad_lines="skip")

# Simple Recommender fonksiyonu
def simple_recommender_tmdb(df, percentile=0.95):
    vote_counts = df[df['tmdb_vote_count'].notnull()]['tmdb_vote_count'].astype('int')
    vote_averages = df[df['tmdb_vote_average'].notnull()]['tmdb_vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').\
        apply(lambda x: str(x).split('-')[0] if pd.notnull(x) else np.nan)
    qualified = df[(df['tmdb_vote_count'] >= m) & (df['tmdb_vote_count'].notnull()) & (
        df['tmdb_vote_average'].notnull())][
        ['title', 'year', 'tmdb_vote_count', 'tmdb_vote_average', 'popularity']]
    qualified['tmdb_vote_count'] = qualified['tmdb_vote_count'].astype('int')
    qualified['tmdb_vote_average'] = qualified['tmdb_vote_average'].astype('int')
    qualified['wr'] = qualified.apply(lambda x: (x['tmdb_vote_count'] / (x['tmdb_vote_count'] + m) *
                                                 x['tmdb_vote_average']) + (m / (m + x['tmdb_vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified[['title', 'year', 'wr']].reset_index(drop=True)

# Uygulama başlıyor
st.title("Inka & Chill 🎥")
st.write("TMDB tabanlı basit tavsiye sistemi.")

try:
    df = load_data()
    st.success("Veri başarıyla yüklendi!")
    st.dataframe(df.head())
    
    # Tavsiye sistemi düğmesi
    if st.button("Tavsiye Al"):
        st.write("Tavsiye edilen filmler:")
        recommendations = simple_recommender_tmdb(df)
        st.table(recommendations)
except Exception as e:
    st.error(f"Veri yüklenirken bir hata oluştu: {e}")


