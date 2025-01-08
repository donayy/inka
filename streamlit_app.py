import pandas as pd
import numpy as np
import streamlit as st
from rapidfuzz import fuzz, process  # Daha hızlı ve güvenilir eşleşme için rapidfuzz kullanıyoruz

# Verisetinin GitHub URL'si
DATA_URL = "https://raw.githubusercontent.com/donayy/inka/refs/heads/main/movie_short_f.csv"

@st.cache_data
def load_data():
    # Hatalı satırları atlamak için on_bad_lines kullanıyoruz
    return pd.read_csv(DATA_URL, on_bad_lines="skip")

# Simple recommender fonksiyonu
def simple_recommender_tmdb(df, percentile=0.95):
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][
        ['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(
        lambda x: (x['vote_count'] / (x['vote_count'] + m) * x['vote_average']) + 
                  (m / (m + x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False)
    qualified = qualified.head(10)[['title', 'vote_average', 'wr']]
    
    return qualified.reset_index(drop=True)

# Genre-based recommender fonksiyonu
def genre_based_recommender_tmbd_f(df, genre, percentile=0.90):
    genre = genre.lower()

    # Genre sütununu normalize ediyoruz
    df['genres'] = df['genres'].apply(lambda x: str(x) if isinstance(x, str) else '')
    df['genres'] = df['genres'].apply(lambda x: [g.strip().lower() for g in x.split(',')] if x else [])

    # En yakın eşleşmeyi buluyoruz
    all_genres = df['genres'].explode().unique()
    closest_match = process.extractOne(genre, all_genres, scorer=fuzz.ratio)[0]

    # Eşleşen türdeki filmleri filtreliyoruz
    df_filtered = df[df['genres'].apply(lambda x: closest_match in x)]

    # tmdb_vote_count ve tmdb_vote_average için filtreleme
    vote_counts = df_filtered[df_filtered['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df_filtered[df_filtered['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)

    # Weighted rating hesaplama
    qualified = df_filtered[(df_filtered['vote_count'] >= m) &
                            (df_filtered['vote_count'].notnull()) &
                            (df_filtered['vote_average'].notnull())][
        ['title', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(
        lambda x: (x['vote_count'] / (x['vote_count'] + m) * x['vote_average']) + 
                  (m / (m + x['vote_count']) * C),
        axis=1)

    qualified = qualified.drop_duplicates(subset='title')
    qualified = qualified.sort_values('wr', ascending=False).head(10)

    return qualified[['title', 'vote_average', 'wr']].reset_index(drop=True)

# Uygulama başlıyor
st.title("Inka ve Chill 🎥")
st.write("TMDB tabanlı önerici sistem.")

try:
    df = load_data()

    # Simple Recommender Başlangıç
    if st.button("En İyi 10 Film (Simple Recommender)"):
        recommendations_simple = simple_recommender_tmdb(df)
        st.write("En İyi 10 Film:")
        st.table(recommendations_simple)

    # Genre-Based Recommender Başlangıç
    genre_input = st.text_input("Bir tür girin (örneğin, Action, Drama, Comedy):")
    if genre_input:
        recommendations_genre = genre_based_recommender_tmbd_f(df, genre_input)
        if not recommendations_genre.empty:
            st.write(f"{genre_input.capitalize()} türündeki öneriler:")
            st.table(recommendations_genre)
        else:
            st.write("Bu türde yeterli film bulunamadı.")

except Exception as e:
    st.error(f"Veri yüklenirken bir hata oluştu: {e}")

