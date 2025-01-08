import pandas as pd
import numpy as np
import streamlit as st
import difflib
from rapidfuzz import fuzz, process  # For better fuzzy matching
from typing import List

# Verisetinin GitHub URL'si
DATA_URL = "https://raw.githubusercontent.com/donayy/inka/refs/heads/main/movie_short_f.csv"

@st.cache_data
def load_data():
    # Hatalı satırları atlamak için on_bad_lines kullanıyoruz
    return pd.read_csv(DATA_URL, on_bad_lines="skip")

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
    qualified = qualified.head(10)[['title', 'vote_average', 'wr']]
    
    return qualified.reset_index(drop=True)

# Genre-based recommender function
def genre_based_recommender_tmbd_f(df, genre, percentile=0.90):
    genre = genre.lower()

    # Normalize the genre column
    df['genres'] = df['genres'].apply(lambda x: str(x) if isinstance(x, str) else '')
    df['genres'] = df['genres'].apply(lambda x: [g.strip().lower() for g in x.split(',')] if x else [])

    # Find the closest match for the genre
    all_genres = df['genres'].explode().unique()
    closest_match = process.extractOne(genre, all_genres, scorer=fuzz.ratio)[0]

    # Filter films by matching genre
    df_filtered = df[df['genres'].apply(lambda x: closest_match in x)]

    # Filter by tmdb_vote_count and tmdb_vote_average
    vote_counts = df_filtered[df_filtered['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df_filtered[df_filtered['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)

    # Calculate weighted rating
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

# Content-based recommender function
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def content_based_recommender(title, dataframe, top_n=10):
    dataframe['genres'] = dataframe['genres'].fillna('').astype(str)
    dataframe['keywords'] = dataframe['keywords'].fillna('').astype(str)

    # Target movie details
    target_movie = dataframe[dataframe['title'] == title]
    if target_movie.empty:
        return []
    
    target_movie = target_movie.iloc[0]
    target_genres = set(target_movie['genres'].split())
    target_keywords = set(target_movie['keywords'].split())

    # Calculate similarity scores
    scores = []
    for _, row in dataframe.iterrows():
        if row['title'] != title:  # Exclude the movie itself
            genres = set(row['genres'].split())
            keywords = set(row['keywords'].split())
            genre_score = jaccard_similarity(target_genres, genres)
            keyword_score = jaccard_similarity(target_keywords, keywords)
            total_score = genre_score * 0.7 + keyword_score * 0.3  # Weighted
            scores.append((row['title'], total_score))

    # Sort by similarity score and return top_n movies
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [title for title, score in scores[:top_n]]

# Uygulama başlıyor
st.title("Inka & Chill 🎥")
st.write("Ne izlesek?")

try:
    df = load_data()

    # Simple Recommender Başlangıç
    st.write("Öncelikle şunları önerebilirim:")
    if st.button("En Beğenilen 10 Film"):
        recommendations_simple = simple_recommender_tmdb(df)
        st.table(recommendations_simple)

    # Genre-Based Recommender Başlangıç
    genre_input = st.text_input("Dilerseniz türe göre arama yapalım. Bir tür girin (örneğin, Action, Drama, Comedy):")
    if genre_input:
        recommendations_genre = genre_based_recommender_tmbd_f(df, genre_input)
        if not recommendations_genre.empty:
            st.write(f"{genre_input.capitalize()} türündeki öneriler:")
            st.table(recommendations_genre)
        else:
            st.write("Bu türde yeterli film bulunamadı.")
    
    # Content-Based Recommender Başlangıç
    movie_input = st.text_input("Bir film ismi girin, benzerlerini önereyim (örneğin, Inception):")
    if movie_input:
        recommendations_content = content_based_recommender(movie_input, df)
        if recommendations_content:
            st.write(f"Benzer filmler:")
            st.table(recommendations_content)
        else:
            st.write(f"'{movie_input}' adlı filme benzer bir film bulunamadı veya film mevcut değil.")

except Exception as e:
    st.error(f"Veri yüklenirken bir hata oluştu: {e}")
