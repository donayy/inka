import pandas as pd
import numpy as np
import streamlit as st
from rapidfuzz import fuzz, process  # For better fuzzy matching
import difflib

# Verisetinin GitHub URL'si
DATA_URL = "https://raw.githubusercontent.com/donayy/inka/refs/heads/main/movie_short_f.csv"

@st.cache_data
def load_data():
    # Veriyi yükle
    df = pd.read_csv(DATA_URL, on_bad_lines="skip")
    # genres ve keywords sütunlarını normalize et
    df['genres'] = df['genres'].fillna('').apply(lambda x: x.split(',') if isinstance(x, str) else [])
    df['keywords'] = df['keywords'].fillna('').apply(lambda x: x.split(',') if isinstance(x, str) else [])
    return df

# Simple recommender function
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
    return qualified.head(10)[['title', 'vote_average', 'wr']].reset_index(drop=True)

# Keyword-based recommender function
def keyword_based_recommender(keyword, dataframe, top_n=10):
    keyword = keyword.lower()
    # 'overview' ve 'keywords' sütunlarında arama yap
    filtered_df = dataframe[
        dataframe['overview'].str.lower().str.contains(keyword, na=False) |
        dataframe['keywords'].str.lower().str.contains(keyword, na=False)
    ]
    # Sonuçları popülerlik veya oy ortalamasına göre sıralayarak getir
    filtered_df = filtered_df.sort_values(by='popularity', ascending=False)
    return filtered_df.head(top_n)[['title', 'overview']]

# Other recommender functions...
# (Genre-Based, Director-Based, Cast-Based, Content-Based recommender functions go here)

# Streamlit App
st.title("Inka & Chill 🎥")
st.write("Ne izlesek?")

try:
    df = load_data()

    # Simple Recommender
    if st.button("En Beğenilen 10 Film"):
        recommendations_simple = simple_recommender_tmdb(df)
        st.table(recommendations_simple)

    # Keyword-Based Recommender
    keyword_input = st.text_input("Bir kelime veya tema girin (örneğin, Christmas):")
    if keyword_input:
        recommendations_keyword = keyword_based_recommender(keyword_input, df)
        if not recommendations_keyword.empty:
            st.write(f"'{keyword_input}' ile ilgili öneriler:")
            st.table(recommendations_keyword)
        else:
            st.write(f"'{keyword_input}' ile ilgili yeterli film bulunamadı.")

    # Other recommenders (Genre-Based, Director-Based, Cast-Based, Content-Based)
    # Example usage:
    # genre_input = st.text_input(...)
    # recommendations_genre = genre_based_recommender_tmbd_f(...)

except Exception as e:
    st.error(f"Bir hata oluştu: {e}")
