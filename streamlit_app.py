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

# Cast-based recommender function
def preprocess_cast_column(df):
    cast_columns = df['cast'].str.split(',', expand=True)
    df = pd.concat([df, cast_columns], axis=1)
    return df

def cast_based_recommender_tmdb_f(df, cast_name, percentile=0.90):
    df = preprocess_cast_column(df)
    cast_columns = df.columns[5:]
    df_cast = df[df[cast_columns].apply(lambda x: x.str.contains(cast_name, na=False).any(), axis=1)]
    if df_cast.empty:
        return f"{cast_name} için film bulunamadı."
    vote_counts = df_cast[df_cast['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df_cast[df_cast['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    qualified = df_cast[(df_cast['vote_count'] >= m) &
                        (df_cast['vote_count'].notnull()) &
                        (df_cast['vote_average'].notnull())][
        ['title', 'vote_count', 'vote_average', 'popularity']]
    qualified['wr'] = qualified.apply(
        lambda x: (x['vote_count'] / (x['vote_count'] + m) * x['vote_average']) + (
                    m / (m + x['vote_count']) * C),
        axis=1)
    qualified = qualified.drop_duplicates(subset='title')
    return qualified.sort_values('wr', ascending=False).head(10)[['title', 'vote_average', 'wr']].reset_index(drop=True)

# Content-based recommender using Jaccard similarity
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def jaccard_based_recommender(title, dataframe, top_n=10):
    # İlgili filmin bilgilerini al
    target_movie = dataframe[dataframe['title'] == title]
    if target_movie.empty:
        return []

    target_movie = target_movie.iloc[0]
    
    # genres ve keywords sütunlarını listeye dönüştür
    target_genres = set(target_movie['genres']) if isinstance(target_movie['genres'], list) else set(str(target_movie['genres']).split(','))
    target_keywords = set(target_movie['keywords']) if isinstance(target_movie['keywords'], list) else set(str(target_movie['keywords']).split(','))

    # Benzerlik skorlarını hesapla
    scores = []
    for _, row in dataframe.iterrows():
        if row['title'] != title:  # Kendini dışla
            genres = set(row['genres']) if isinstance(row['genres'], list) else set(str(row['genres']).split(','))
            keywords = set(row['keywords']) if isinstance(row['keywords'], list) else set(str(row['keywords']).split(','))
            genre_score = jaccard_similarity(target_genres, genres)
            keyword_score = jaccard_similarity(target_keywords, keywords)
            total_score = genre_score * 0.7 + keyword_score * 0.3  # Ağırlıklandırma
            scores.append((row['title'], total_score))

    # Skorlara göre sırala ve top_n sonuçları getir
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [title for title, score in scores[:top_n]]

# Streamlit App
st.title("Inka & Chill 🎥")
st.write("Ne izlesek?")

try:
    df = load_data()

    # Simple Recommender
    if st.button("En Beğenilen 10 Film"):
        recommendations_simple = simple_recommender_tmdb(df)
        st.table(recommendations_simple)

    # Genre-Based Recommender
    genre_input = st.text_input("Bir tür girin (örneğin, Action, Drama):")
    if genre_input:
        recommendations_genre = genre_based_recommender_tmbd_f(df, genre_input)
        if not recommendations_genre.empty:
            st.write(f"{genre_input.capitalize()} türündeki öneriler:")
            st.table(recommendations_genre)
        else:
            st.write("Bu türde yeterli film bulunamadı.")

    # Director-Based Recommender
    director_input = st.text_input("Bir yönetmen ismi girin (örneğin, Christopher Nolan):")
    if director_input:
        recommendations_director = director_based_recommender_tmdb_f(director_input, df)
        if isinstance(recommendations_director, pd.DataFrame):
            st.write(f"{director_input.capitalize()} yönetimindeki öneriler:")
            st.table(recommendations_director)
        else:
            st.write(recommendations_director)

    # Cast-Based Recommender
    cast_input = st.text_input("Bir oyuncu ismi girin (örneğin, Christian Bale):")
    if cast_input:
        recommendations_cast = cast_based_recommender_tmdb_f(df, cast_input)
        if not recommendations_cast.empty:
            st.write(f"{cast_input.capitalize()} oyuncusunun yer aldığı öneriler:")
            st.table(recommendations_cast)
        else:
            st.write("Bu oyuncunun yer aldığı yeterli film bulunamadı.")

    # Content-Based Recommender
    content_input = st.text_input("Bir film ismi girin (örneğin, Inception):")
    if content_input:
        recommendations_content = jaccard_based_recommender(content_input, df)
        if recommendations_content:
            st.write(f"'{content_input}' filmini sevdiyseniz şunları öneririz:")
            st.write(recommendations_content)
        else:
            st.write("Bu filmle ilgili yeterli veri bulunamadı.")
except Exception as e:
    st.error(f"Bir hata oluştu: {e}")
