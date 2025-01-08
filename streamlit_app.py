import pandas as pd
import numpy as np
import streamlit as st
from rapidfuzz import fuzz, process  # For better fuzzy matching
import difflib

# Verisetinin GitHub URL'si
DATA_URL = "https://raw.githubusercontent.com/donayy/inka/refs/heads/main/movies_dataset.csv"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"  # Base URL for TMDB poster images


@st.cache_data
@st.cache_data
def load_data():
    # Veriyi yükle
    df = pd.read_csv(DATA_URL, on_bad_lines="skip")
    
    # genres kolonunu normalize et
    if 'genres' in df.columns:
        df['genres'] = df['genres'].fillna('').apply(lambda x: x.split(',') if isinstance(x, str) else [])
    else:
        st.error("'genres' kolonu bulunamadı. Lütfen verinizi kontrol edin.")
        return pd.DataFrame()  # Boş bir dataframe döndür
    
    # keywords ve overview kolonlarını normalize et
    df['keywords'] = df['keywords'].fillna('').apply(lambda x: x.split(',') if isinstance(x, str) else [])
    df['overview'] = df['overview'].fillna('').astype(str)
    df['keywords'] = df['keywords'].fillna('').astype(str)
    
    # Poster URL'si oluştur
    if 'backdrop_path' in df.columns:
        df['poster_url'] = df['backdrop_path'].apply(lambda x: f"{POSTER_BASE_URL}{x}" if pd.notnull(x) else None)
    
    return df


# Simple recommender function
def simple_recommender_tmdb(df, percentile=0.95):
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][
        ['title', 'year', 'vote_count', 'vote_average', 'popularity', 'poster_url']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(
        lambda x: (x['vote_count'] / (x['vote_count'] + m) * x['vote_average']) + 
                  (m / (m + x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False)
    return qualified.head(10)[['title', 'vote_average', 'wr', 'poster_url']].reset_index(drop=True)


# Genre-based recommender function
def genre_based_recommender_tmbd_f(df, genre, percentile=0.90):
    genre = genre.lower()

    # Normalize the genres column
    df['genres'] = df['genres'].apply(lambda x: x if isinstance(x, list) else str(x).split(','))

    # Get all unique genres
    all_genres = df['genres'].explode().unique()
    if not all_genres.size:
        return "No genres available in the dataset."

    # Find the closest matching genre
    closest_match = process.extractOne(genre, all_genres, scorer=fuzz.ratio)
    if closest_match:
        closest_match = closest_match[0]
    else:
        return f"No matching genres found for: {genre}"

    # Filter movies by the matched genre
    df_filtered = df[df['genres'].apply(lambda x: closest_match in x)]
    if df_filtered.empty:
        return f"No movies found for the genre: {closest_match}"

    # Calculate weighted rating
    vote_counts = df_filtered[df_filtered['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df_filtered[df_filtered['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)

    qualified = df_filtered[(df_filtered['vote_count'] >= m) &
                            (df_filtered['vote_count'].notnull()) &
                            (df_filtered['vote_average'].notnull())][
        ['title', 'vote_count', 'vote_average', 'popularity']]
    if qualified.empty:
        return f"No qualified movies found for the genre: {closest_match}"

    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(
        lambda x: (x['vote_count'] / (x['vote_count'] + m) * x['vote_average']) +
                  (m / (m + x['vote_count']) * C),
        axis=1
    )

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

# Keyword-based recommender function
def keyword_based_recommender(keyword, dataframe, top_n=10):
    keyword = keyword.lower()
    
    # Ensure the columns are strings
    dataframe['overview'] = dataframe['overview'].astype(str)
    dataframe['keywords'] = dataframe['keywords'].astype(str)
    
    # Filter the dataframe
    filtered_df = dataframe[
        dataframe['overview'].str.lower().str.contains(keyword, na=False) |
        dataframe['keywords'].str.lower().str.contains(keyword, na=False)
    ]
    filtered_df = filtered_df.sort_values(by='popularity', ascending=False)
    return filtered_df.head(top_n)[['title']]


# Content-based recommender using Jaccard similarity
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def jaccard_based_recommender(title, dataframe, top_n=10):
    target_movie = dataframe[dataframe['title'] == title]
    if target_movie.empty:
        return []

    target_movie = target_movie.iloc[0]
    target_genres = set(target_movie['genres']) if isinstance(target_movie['genres'], list) else set(str(target_movie['genres']).split(','))
    target_keywords = set(target_movie['keywords']) if isinstance(target_movie['keywords'], list) else set(str(target_movie['keywords']).split(','))

    scores = []
    for _, row in dataframe.iterrows():
        if row['title'] != title:
            genres = set(row['genres']) if isinstance(row['genres'], list) else set(str(row['genres']).split(','))
            keywords = set(row['keywords']) if isinstance(row['keywords'], list) else set(str(row['keywords']).split(','))
            genre_score = jaccard_similarity(target_genres, genres)
            keyword_score = jaccard_similarity(target_keywords, keywords)
            total_score = genre_score * 0.7 + keyword_score * 0.3
            scores.append((row['title'], total_score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [title for title, score in scores[:top_n]]

# Mood-based recommender function
mood_to_genre = {
    "happy": ["comedy", "family"],
    "sad": ["drama", "romance"],
    "adventurous": ["action", "adventure"],
    "scary": ["horror", "thriller"]
}

def mood_based_recommender(mood, dataframe, top_n=10):
    # Get genres related to the mood
    genres = mood_to_genre.get(mood.lower(), [])
    if not genres:
        return f"No genres found for mood: {mood}"
    
    # Ensure 'genres' is processed as a list
    dataframe['genres'] = dataframe['genres'].apply(lambda x: x if isinstance(x, list) else str(x).split(','))
    
    # Filter movies where any genre matches the mood genres
    filtered_df = dataframe[dataframe['genres'].apply(lambda x: any(g.strip().lower() in genres for g in x))]
    
    # Sort by popularity and return the top results
    filtered_df = filtered_df.sort_values(by='popularity', ascending=False)
    return filtered_df.head(top_n)[['title', 'genres']].reset_index(drop=True)
    
# Streamlit App
st.title("Inka & Chill 🎥")
st.write("Ne izlesek?")

try:
    df = load_data()

    # Simple Recommender
    if st.button("En Beğenilen 10 Film"):
        recommendations_simple = simple_recommender_tmdb(df)
        for _, row in recommendations_simple.iterrows():
            st.write(f"**{row['title']}** (Rating: {row['vote_average']})")
            if row['poster_url']:
                st.image(row['poster_url'], width=150)
            else:
                st.write("Poster bulunamadı.")

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

    # Keyword-Based Recommender
    keyword_input = st.text_input("Bir kelime veya tema girin (örneğin, Christmas):")
        if keyword_input:
            try:
                recommendations_keyword = keyword_based_recommender(keyword_input, df)
            if not recommendations_keyword.empty:
                st.write(f"'{keyword_input}' ile ilgili öneriler:")
                st.table(recommendations_keyword)
            else:
                st.write(f"'{keyword_input}' ile ilgili yeterli film bulunamadı.")
        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")


    # Mood-Based Recommender
    mood_input = st.text_input("Bir ruh hali girin (örneğin, happy, sad):")
    if mood_input:
        recommendations_mood = mood_based_recommender(mood_input, df)
        if isinstance(recommendations_mood, pd.DataFrame):
            st.write(f"'{mood_input.capitalize()}' modundaysanız şunları öneririz:")
            st.table(recommendations_mood)
        else:
            st.write(recommendations_mood)
            
except Exception as e:
    st.error(f"Bir hata oluştu: {e}")
