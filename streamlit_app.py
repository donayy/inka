import pandas as pd
import streamlit as st
from rapidfuzz import fuzz, process  
import difflib

# GitHub URL of dataset
DATA_URL = "https://raw.githubusercontent.com/donayy/inka/refs/heads/main/movies_dataset.csv"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"  


@st.cache_data
def load_data():
    # Load dataset
    df = pd.read_csv(DATA_URL, on_bad_lines="skip")
    
    # Normalize the genres column
    if 'genres' in df.columns:
        df['genres'] = df['genres'].fillna('').apply(lambda x: x.split(',') if isinstance(x, str) else [])
    else:
        st.error("'genres' kolonu bulunamadı. Lütfen verinizi kontrol edin.")
        return pd.DataFrame()  # return an empty dataset
    
    # Normalize the keywords and overview columns
    df['keywords'] = df['keywords'].fillna('').apply(lambda x: x.split(',') if isinstance(x, str) else [])
    df['overview'] = df['overview'].fillna('').astype(str)
    
    # Generate Poster URL
    if 'backdrop_path' in df.columns:
        df['poster_url'] = df['backdrop_path'].apply(lambda x: f"{POSTER_BASE_URL}{x}" if pd.notnull(x) else None)
    
    return df


# Simple recommender function
def simple_recommender_tmdb(df, percentile=0.95):
    num_votes = df[df['numVotes'].notnull()]['numVotes'].astype('int')
    vote_averages = df[df['averageRating'].notnull()]['averageRating'].astype('float')
    C = vote_averages.mean()
    m = num_votes.quantile(percentile)
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    qualified = df[(df['numVotes'] >= m) & (df['averageRating'].notnull())][
        ['title', 'year', 'numVotes', 'averageRating', 'poster_url']]
    qualified['wr'] = qualified.apply(
        lambda x: (x['numVotes'] / (x['numVotes'] + m) * x['averageRating']) + 
                  (m / (m + x['numVotes']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False)
    return qualified[['title', 'averageRating', 'poster_url']].reset_index(drop=True)


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
    numVotess = df_filtered[df_filtered['numVotes'].notnull()]['numVotes'].astype('int')
    vote_averages = df_filtered[df_filtered['averageRating'].notnull()]['averageRating'].astype('int')
    C = vote_averages.mean()
    m = numVotess.quantile(percentile)

    qualified = df_filtered[(df_filtered['numVotes'] >= m) &
                            (df_filtered['numVotes'].notnull()) &
                            (df_filtered['averageRating'].notnull())][
        ['title', 'numVotes', 'averageRating', 'popularity']]
    if qualified.empty:
        return f"No qualified movies found for the genre: {closest_match}"

    qualified['numVotes'] = qualified['numVotes'].astype('int')
    qualified['averageRating'] = qualified['averageRating'].astype('int')
    qualified['wr'] = qualified.apply(
        lambda x: (x['numVotes'] / (x['numVotes'] + m) * x['averageRating']) +
                  (m / (m + x['numVotes']) * C),
        axis=1
    )

    qualified = qualified.drop_duplicates(subset='title')
    return qualified.sort_values('wr', ascending=False).head(10)[['title', 'averageRating']].reset_index(drop=True)


# Director-based recommender function
def director_based_recommender_tmdb_f(director, dataframe, percentile=0.90):
    director_choices = dataframe['directors'].dropna().unique()
    closest_match = difflib.get_close_matches(director, director_choices, n=1, cutoff=0.8)
    if not closest_match:
        return f"Warning: {director} isimli bir yönetmen bulunamadı."
    closest_match = closest_match[0]
    df = dataframe[dataframe['directors'] == closest_match]
    numVotess = df[df['numVotes'].notnull()]['numVotes'].astype('int')
    vote_averages = df[df['averageRating'].notnull()]['averageRating'].astype('int')
    C = vote_averages.mean()
    m = numVotess.quantile(percentile)
    qualified = df[(df['numVotes'] >= m) & (df['numVotes'].notnull()) & 
                   (df['averageRating'].notnull())][['title', 'numVotes', 'averageRating', 'popularity']]
    qualified['numVotes'] = qualified['numVotes'].astype('int')
    qualified['averageRating'] = qualified['averageRating'].astype('int')
    qualified['wr'] = qualified.apply(
        lambda x: (x['numVotes'] / (x['numVotes'] + m) * x['averageRating']) + (m / (m + x['numVotes']) * C),
        axis=1)
    qualified = qualified.drop_duplicates(subset='title')
    return qualified.sort_values('wr', ascending=False).head(10)[['title', 'averageRating']].reset_index(drop=True)

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
    numVotess = df_cast[df_cast['numVotes'].notnull()]['numVotes'].astype('int')
    vote_averages = df_cast[df_cast['averageRating'].notnull()]['averageRating'].astype('int')
    C = vote_averages.mean()
    m = numVotess.quantile(percentile)
    qualified = df_cast[(df_cast['numVotes'] >= m) &
                        (df_cast['numVotes'].notnull()) &
                        (df_cast['averageRating'].notnull())][
        ['title', 'numVotes', 'averageRating', 'popularity']]
    qualified['wr'] = qualified.apply(
        lambda x: (x['numVotes'] / (x['numVotes'] + m) * x['averageRating']) + (
                    m / (m + x['numVotes']) * C),
        axis=1)
    qualified = qualified.drop_duplicates(subset='title')
    return qualified.sort_values('wr', ascending=False).head(10)[['title', 'averageRating']].reset_index(drop=True)

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

    return filtered_df.head(top_n)[['title']].reset_index(drop=True)



# Content-based recommender using Jaccard similarity
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def jaccard_based_recommender(title, dataframe, top_n=10):
    target_movie = dataframe[dataframe['title'] == title]
    if target_movie.empty:
        return pd.DataFrame(columns=['title', 'score'])

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
            scores.append({'title': row['title'], 'score': total_score})

    sorted_scores = sorted(scores, key=lambda x: x['score'], reverse=True)[:top_n]

    return pd.DataFrame(sorted_scores).reset_index(drop=True)

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
    return filtered_df.head(top_n)[['title']].reset_index(drop=True)
    


# Streamlit App
st.title("Inka & Chill 🎥")
st.write("Ne izlesek?")

try:
    df = load_data()

    st.sidebar.title("Navigasyon")
    page = st.sidebar.radio(
        "Gitmek istediğiniz sayfayı seçin:",
        ("Simple Recommender", "Genre-Based Recommender", "Director-Based Recommender", 
         "Cast-Based Recommender", "Content-Based Recommender", "Keyword-Based Recommender",
         "Mood-Based Recommender")
    )

    if page == "Simple Recommender":
        st.title("Simple Recommender")
        if st.button("En Beğenilen 10 Film"):
            recommendations_simple = simple_recommender_tmdb(df)
            for _, row in recommendations_simple.iterrows():
                st.write(f"**{row['title']}** (Rating: {row['averageRating']})")
                if row['poster_url']:
                    st.image(row['poster_url'], width=150)
                else:
                    st.write("Poster bulunamadı.")



    elif page == "Genre-Based Recommender":
        st.title("Genre-Based Recommender")
        genre = st.text_input("Bir tür girin (örneğin, Action):")
        if genre:
            recommendations = genre_based_recommender_tmbd_f(df, genre)
            if not recommendations.empty:
                st.table(recommendations)
            else:
                st.write(f"'{genre}' türünde yeterli film bulunamadı.")

    elif page == "Director-Based Recommender":
        st.title("Director-Based Recommender")
        director = st.text_input("Bir yönetmen ismi girin (örneğin, Christopher Nolan):")
        if director:
            recommendations = director_based_recommender_tmdb_f(director, df)
            if isinstance(recommendations, pd.DataFrame):
                st.table(recommendations)
            else:
                st.write(recommendations)
                
    elif page == "Cast-Based Recommender":
        st.title("Cast-Based Recommender")
        cast_name = st.text_input("Bir oyuncu ismi girin (örneğin, Christian Bale):")
        if cast_name:
            recommendations = cast_based_recommender_tmdb_f(df, cast_name)
            st.table(recommendations)

    
    elif page == "Content-Based Recommender":
        st.title("Content-Based Recommender")
        content_input = st.text_input("Bir film ismi girin (örneğin, Inception):")
        if st.button("Filmleri Getir"):
            if content_input:
                try:
                    recommendations_content = jaccard_based_recommender(content_input, df)
                    if not recommendations_content.empty:
                        st.write(f"'{content_input}' filmini sevdiyseniz şunları öneririz:")
                        st.table(recommendations_content)
                    else:
                        st.error(f"'{content_input}' ile ilgili yeterli veri bulunamadı.", icon="⚠️")
                except Exception as e:
                    st.error(f"Bir hata oluştu: {str(e)}", icon="🚨")
            else:
                st.warning("Lütfen bir film ismi girin.", icon="❗")



    elif page == "Keyword-Based Recommender":
        st.title("Keyword-Based Recommender")
        keyword = st.text_input("Bir kelime girin (örneğin, Christmas):")
        if keyword:
            recommendations = keyword_based_recommender(keyword, df)
            st.table(recommendations)

    elif page == "Mood-Based Recommender":
        st.title("Mood-Based Recommender")
        mood = st.text_input("Bir ruh hali girin (örneğin, happy, sad):")
        if mood:
            recommendations = mood_based_recommender(mood, df)
            st.table(recommendations)

except Exception as e:
    st.error








       




except Exception as e:
    st.error(f"Bir hata oluştu: {e}")
