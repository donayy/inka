import pandas as pd
import numpy as np
import streamlit as st
import difflib
from rapidfuzz import fuzz, process  # For better fuzzy matching

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

# Director-based recommender function
def director_based_recommender_tmdb_f(director, dataframe, percentile=0.90):
    director_choices = dataframe['directors'].dropna().unique()

    # Find closest matching director name
    closest_match = difflib.get_close_matches(director, director_choices, n=1, cutoff=0.8)

    # If no match is found, print a warning
    if not closest_match:
        print(f"Warning: No close match found for director: {director}")
        return pd.Series([])

    # Filter by the closest match
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
    qualified = qualified.sort_values('wr', ascending=False).head(10)

    return qualified[['title', 'vote_average', 'wr']].reset_index(drop=True)

# Cast column preprocessing function
def preprocess_cast_column(df):
    # Split the cast column by commas and create new columns for each actor
    cast_columns = df['cast'].str.split(',', expand=True)
    df = pd.concat([df, cast_columns], axis=1)
    return df

# Cast-based recommender function
def cast_based_recommender_tmdb_f(df, cast_name, percentile=0.90):
    # Preprocess the cast column
    df = preprocess_cast_column(df)

    # Find films with the given cast member
    cast_columns = df.columns[5:]  # New cast columns (excluding the first 5 columns)
    df_cast = df[df[cast_columns].apply(lambda x: x.str.contains(cast_name, na=False).any(), axis=1)]

    # If no films found for the cast member
    if df_cast.empty:
        return f"{cast_name} için film bulunamadı."

    # Filter films by tmdb_vote_count and tmdb_vote_average
    vote_counts = df_cast[df_cast['tmdb_vote_count'].notnull()]['tmdb_vote_count'].astype('int')
    vote_averages = df_cast[df_cast['tmdb_vote_average'].notnull()]['tmdb_vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)

    qualified = df_cast[(df_cast['tmdb_vote_count'] >= m) &
                        (df_cast['tmdb_vote_count'].notnull()) &
                        (df_cast['tmdb_vote_average'].notnull())][
        ['title', 'tmdb_vote_count', 'tmdb_vote_average', 'popularity']]

    # Calculate weighted rating (wr)
    qualified['wr'] = qualified.apply(
        lambda x: (x['tmdb_vote_count'] / (x['tmdb_vote_count'] + m) * x['tmdb_vote_average']) + (
                    m / (m + x['tmdb_vote_count']) * C),
        axis=1)

    # Sort the results and return the top 10 films
    qualified = qualified.drop_duplicates(subset='title')
    qualified = qualified.sort_values('wr', ascending=False).head(10)["title"]

    return qualified.reset_index(drop=True)

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

    # Director-Based Recommender Başlangıç
    director_input = st.text_input("Bir yönetmen ismi girin (örneğin, Christopher Nolan):")
    if director_input:
        recommendations_director = director_based_recommender_tmdb_f(director_input, df)
        if not recommendations_director.empty:
            st.write(f"{director_input.capitalize()} tarafından yönetilen öneriler:")
            st.table(recommendations_director)
        else:
            st.write("Bu yönetmenin yönetmenliğinde yeterli film bulunamadı.")
    
    # Cast-Based Recommender Başlangıç
    cast_input = st.text_input("Bir oyuncu ismi girin (örneğin, Christian Bale):")
    if cast_input:
        recommendations_cast = cast_based_recommender_tmdb_f(df, cast_input)
        if isinstance(recommendations_cast, pd.Series):
            st.write(f"{cast_input.capitalize()} oyuncusunun yer aldığı öneriler:")
            st.table(recommendations_cast)
        else:
            st.write(recommendations_cast)

except Exception as e:
    st.error(f"Veri yüklenirken bir hata oluştu: {e}")

