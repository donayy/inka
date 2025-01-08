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
    return qualified.head(10)[['title', 'averageRating', 'poster_url']].reset_index(drop=True)



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
    
def get_genre_suggestions(partial_input, all_genres):
    partial_input = partial_input.lower()
    suggestions = [genre for genre in all_genres if partial_input in genre]
    return suggestions

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

def content_based_recommender(title, dataframe, top_n=10):
    target_movie = dataframe[dataframe['title'] == title]
    if target_movie.empty:
        return pd.DataFrame(columns=['Film Adı', 'IMDB Rating'])

    target_movie = target_movie.iloc[0]

    target_genres = set(target_movie['genres']) if isinstance(target_movie['genres'], list) else set(str(target_movie['genres']).split(','))
    target_keywords = set(target_movie['keywords']) if isinstance(target_movie['keywords'], list) else set(str(target_movie['keywords']).split(','))

    recommendations = []

    for _, row in dataframe.iterrows():
        if row['title'] != title:
            genres = set(row['genres']) if isinstance(row['genres'], list) else set(str(row['genres']).split(','))
            keywords = set(row['keywords']) if isinstance(row['keywords'], list) else set(str(row['keywords']).split(','))
            
            genre_score = jaccard_similarity(target_genres, genres)
            keyword_score = jaccard_similarity(target_keywords, keywords)
            total_score = genre_score * 0.7 + keyword_score * 0.3

            if pd.isna(row['averageRating']):
                continue

            recommendations.append({'Film Adı': row['title'], 'IMDB Rating': row['averageRating'], 'Total Score': total_score})

    sorted_recommendations = sorted(recommendations, key=lambda x: x['Total Score'], reverse=True)[:top_n]

    return pd.DataFrame(sorted_recommendations).drop(columns=['Total Score']).reset_index(drop=True)



# Mood-based recommender function
mood_to_genre = {
    "happy": ["comedy", "family", "musical"],
    "sad": ["drama", "romance", "biography"],
    "adventurous": ["action", "adventure", "fantasy", "sci-fi"],
    "scary": ["horror", "thriller", "mystery"],
    "excited": ["action", "adventure", "sci-fi", "crime"],
    "relaxed": ["comedy", "family", "romance", "animation"],
    "curious": ["mystery", "documentary", "crime"],
    "nostalgic": ["classic", "musical", "family", "romance"],
    "inspired": ["biography", "history", "sport", "drama"],
    "romantic": ["romance", "comedy", "drama"],
    "thoughtful": ["drama", "mystery", "sci-fi"],
    "funny": ["comedy", "animation", "family"],
    "dark": ["thriller", "horror", "crime", "drama"],
    "uplifting": ["family", "musical", "adventure", "comedy"],
    "tense": ["thriller", "crime", "mystery", "horror"],
    "magical": ["fantasy", "sci-fi", "animation", "adventure"]
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

url = "https://img.freepik.com/free-photo/copy-space-popcorn-table_23-2148470198.jpg?t=st=1736371692~exp=1736375292~hmac=99acfcb457dcd82a820f5d9089e5a7c1890d658a9f5861b02dbae8fc5b704cfa&w=1800"
st.markdown(
    f"""
    <style>
    body {{
        background-image: url("{url}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    body {
        background-color: #121212; /* Koyu Gri */
        color: #FFFFFF; /* Beyaz */
    }

    h1, h2, h3, h4, h5, h6 {
        color: #FFD700; /* Altın Sarısı */
    }

    [data-testid="stSidebar"] {
        background-color: #1C1C1C; /* Sidebar için koyu gri */
        color: #FFFFFF; /* Sidebar metin rengi beyaz */
    }

    .stButton>button {
        background-color: #FFD700; /* Buton için Altın Sarısı */
        color: #121212; /* Metin için koyu gri */
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    h1 {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .center-title {
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 40px;
        font-weight: bold;
        margin-top: -50px;
    }
    .page-title {
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 24px;
        font-weight: normal;
        margin-bottom: 20px;
    }
    </style>
    <div class="center-title">Inka & Chill 🎥 </div>
    <div class="page-title">Ne izlesek?</div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* Sidebar Arka Plan Rengi ve Yazı Stilleri */
    [data-testid="stSidebar"] {
        background-color: #333333; /* Koyu Antrasit */
        color: #FFFFFF; /* Beyaz */
    }
    /* Sidebar Yazı Font Boyutu ve Renk */
    [data-testid="stSidebar"] .css-1v3fvcr {
        font-size: 18px; /* Daha büyük font */
        color: #FFFFFF; /* Beyaz */
    }
    /* Sidebar'daki başlık yazıları */
    [data-testid="stSidebar"] h2 {
        font-size: 22px; /* Başlık boyutunu artır */
        color: #FFD700; /* Altın sarısı */
    }
    /* Sidebar'daki alt yazılar */
    [data-testid="stSidebar"] p {
        font-size: 16px; /* Alt yazı boyutu */
        color: #FFFFFF; /* Beyaz */
    }
    </style>
    """,
    unsafe_allow_html=True
)



try:
    df = load_data()

    st.sidebar.title("Menu")
    page = st.sidebar.radio(
        "Gitmek istediğiniz sayfayı seçin:",
        ("En Beğenilen Filmler", "Tür Bazlı Tavsiye", "Yönetmen Bazlı Tavsiye", 
         "Oyuncu Bazlı Tavsiye", "Filme Göre Tavsiye", "İçerik Bazlı Tavsiye",
         "Ruh Haline Göre Tavsiye", "Hakkında")
    )

    if page == "En Beğenilen Filmler":
        st.write("En Beğenilen Filmler")
        if st.button("En beğenilen 10 film için tıklayın"):
            recommendations_simple = simple_recommender_tmdb(df)
            for _, row in recommendations_simple.iterrows():
                st.write(f"**{row['title']}** (IMDB Rating: {row['averageRating']})")
                if row['poster_url']:
                    st.image(row['poster_url'], width=500)
                else:
                    st.write("Poster bulunamadı.")

    elif page == "Tür Bazlı Tavsiye":
        st.write("Tür Bazlı Tavsiye")

        df['genres'] = df['genres'].apply(lambda x: x if isinstance(x, list) else str(x).split(','))
        all_genres = sorted(set(genre.strip().lower() for genres in df['genres'] for genre in genres))

        genre_input = st.text_input("Bir tür girin (örneğin, Action):")

        if genre_input:

            suggestions = get_genre_suggestions(genre_input, all_genres)

            if suggestions:
                st.write("Öneriler:")
                for suggestion in suggestions[:5]:  
                    st.write(f"- {suggestion.capitalize()}")

                closest_match = suggestions[0]
                recommendations = genre_based_recommender_tmbd_f(df, closest_match)

                if not recommendations.empty:
                    st.write(f"'{closest_match.capitalize()}' türündeki öneriler:")
                    st.table(recommendations)
                else:
                    st.write(f"'{closest_match}' türünde yeterli film bulunamadı.")
            else:
                st.write(f"'{genre_input}' ile başlayan tür bulunamadı. Lütfen başka bir tür deneyin.")
        else:
            st.write("Tür için bir şeyler yazmaya başlayın...")
        
    
    elif page == "Yönetmen Bazlı Tavsiye":
        st.write("Yönetmen Bazlı Tavsiye")
        director = st.text_input("Bir yönetmen ismi girin (örneğin, Christopher Nolan):")
        if director:
            recommendations = director_based_recommender_tmdb_f(director, df)
            if isinstance(recommendations, pd.DataFrame):
                st.table(recommendations)
            else:
                st.write(recommendations)
                
    elif page == "Oyuncu Bazlı Tavsiye":
        st.write("Oyuncu Bazlı Tavsiye")
        cast_name = st.text_input("Bir oyuncu ismi girin (örneğin, Christian Bale):")
        if cast_name:
            recommendations = cast_based_recommender_tmdb_f(df, cast_name)
            st.table(recommendations)

    
    elif page == "Filme Göre Tavsiye":
        st.write("Filme Göre Tavsiye")
        movie_title = st.text_input("Bir film ismi girin (örneğin, Inception):")
        if movie_title:
            recommendations = content_based_recommender(movie_title, df)
            st.table(recommendations)


    elif page == "İçerik Bazlı Tavsiye":
        st.write("İçerik Bazlı Tavsiye")
        keyword = st.text_input("Bir kelime girin (örneğin, Christmas):")
        if keyword:
            recommendations = keyword_based_recommender(keyword, df)
            st.table(recommendations)

    elif page == "Ruh Haline Göre Tavsiye":
        st.write("Ruh Hali Bazlı Tavsiye")
        mood = st.text_input("Bir ruh hali girin (örneğin, happy, sad):")
        if mood:
            recommendations = mood_based_recommender(mood, df)
            st.table(recommendations)

    elif page == "Hakkında":
        st.title("Hakkında")
        st.write("""
        **I.N.K.A.** (Intelligent Network for Kinematic Advice), kullanıcıların ruh haline, tercih ettikleri türlere, 
        yönetmenlere veya anahtar kelimelere göre film önerileri sunan yenilikçi bir film tavsiye sistemidir. 
        
        Bu sistem, geniş bir veri kümesi üzerinde çalışarak hem popüler hem de kişiselleştirilmiş önerilerde bulunur. 
        I.N.K.A., akıllı algoritmaları sayesinde izleme deneyiminizi en üst düzeye çıkarmayı hedefler.
        
        **Özellikler:**
        - Tür, yönetmen, oyuncu ve ruh hali bazlı öneriler.
        - Kullanıcı dostu arayüz ve hızlı sonuçlar.
        - Kapsamlı film veri tabanı.
        - IMDB derecelendirmeleri ve popülerlik bazlı sıralama.
        
        **Amacımız:** I.N.K.A. ile herkes için doğru filmi bulmak ve keyifli bir sinema deneyimi yaşatmak!""")

except Exception as e:
    st.error








       




except Exception as e:
    st.error(f"Bir hata oluştu: {e}")
