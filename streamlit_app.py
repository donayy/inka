import pandas as pd
import numpy as np
import streamlit as st
from rapidfuzz import fuzz, process  # For better fuzzy matching
import difflib

# Verisetinin GitHub URL'si
DATA_URL = "https://raw.githubusercontent.com/donayy/inka/refs/heads/main/movie_short_f.csv"

@st.cache_data
def load_data():
    # Load data from the provided URL
    df = pd.read_csv(DATA_URL, on_bad_lines="skip")
    
    # Normalize the 'genres' and 'keywords' columns to be lists
    df['genres'] = df['genres'].fillna('').apply(lambda x: x.split(',') if isinstance(x, str) else [])
    df['keywords'] = df['keywords'].fillna('').apply(lambda x: x.split(',') if isinstance(x, str) else [])
    
    # Ensure 'overview' and 'keywords' are strings for string operations
    df['overview'] = df['overview'].fillna('').astype(str)
    df['keywords'] = df['keywords'].fillna('').astype(str)
    
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

# Genre-based recommender function
def genre_based_recommender_tmbd_f(df, genre, percentile=0.90):
    genre = genre.lower()
    df['genres'] = df['genres'].apply(lambda x: str(x) if isinstance(x, str) else '')
    df['genres'] = df['genres'].apply(lambda x: [g.strip().lower() for g in x.split(',')] if x else [])
    all_genres = df['genres'].explode().unique()
    closest_match = process.extractOne(genre, all_genres, scorer=fuzz.ratio)[0]
    df_filtered = df[df['genres'].apply(lambda x: closest_match in x)]
    vote_counts = df_filtered[df_filtered['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df_filtered[df_filtered['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
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
    return qualified.sort_values('wr', ascending=False).head(10)[['title', 'vote_average', 'wr']].reset_index(drop=True)

# Director-based recommender function
def director_based_recommender_tmdb_f(director, dataframe, percentile=0.90):
    director_choices = dataframe['directors'].dropna().unique()
    closest_match = difflib.get_close_matches(director, director_choices, n=1, cutoff=0.8)
    if not closest_match:
        return f"Warning: {director} isimli bir yönetmen bulunamadı."
    closest_match = closest_match[0]
    df = dataframe[dataframe['directors'] == closest_match]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & 
                   (df['vote_average'].notnull())][['title', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(
        lambda x: (x['vote_count'] / (x['vote_count'] + m) * x['vote_average']) + (m / (m + x['vote_count']) * C),
        axis=1)
    qualified = qualified.drop_duplicates(subset='title')
    return qualified.sort_values('wr', ascending=False).head(10)[['title', 'vote_average', 'wr']].reset_index(drop=True)

# Mood-based recommender function
mood_to_genre = {
    "happy": ["comedy", "family"],
    "sad": ["drama", "romance"],
    "adventurous": ["action", "adventure"],
    "scary": ["horror", "thriller"]
}

def mood_based_recommender(mood, dataframe, top_n=10):
    genres = mood_to_genre.get(mood.lower(), [])
    if not genres:
        return f"No genres found for mood: {mood}"
    filtered_df = dataframe[dataframe['genres'].apply(lambda x: any(g in genres for g in x))]
    filtered_df = filtered_df.sort_values(by='popularity', ascending=False)
    return filtered_df.head(top_n)[['title', 'genres']].reset_index(drop=True)

# Streamlit App
st.title("Inka & Chill 🎥")
st.write("Ne izlesek?")

try:
    df = load_data()

    # Mood-Based Recommender
    mood_input = st.text_input("Bir ruh hali girin (örneğin, happy, sad):")
    if mood_input:
        recommendations_mood = mood_based_recommender(mood_input, df)
        if isinstance(recommendations_mood, pd.DataFrame):
            st.write(f"'{mood_input.capitalize()}' modundaysanız şunları öneririz:")
            st.table(recommendations_mood)
        else:
            st.write(recommendations_mood)

    # Other recommenders...
except Exception as e:
    st.error(f"Bir hata oluştu: {e}")
