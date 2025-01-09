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
def genre_based_recommender(df, genre, percentile=0.90):
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
    num_votes = df_filtered[df_filtered['numVotes'].notnull()]['numVotes'].astype('int')
    vote_averages = df_filtered[df_filtered['averageRating'].notnull()]['averageRating'].astype('float')
    C = vote_averages.mean()
    m = num_votes.quantile(percentile)

    qualified = df_filtered[(df_filtered['numVotes'] >= m) &
                            (df_filtered['numVotes'].notnull()) &
                            (df_filtered['averageRating'].notnull())][
        ['title', 'numVotes', 'averageRating', 'poster_url']]
    
    if qualified.empty:
        return f"No qualified movies found for the genre: {closest_match}"

    qualified['numVotes'] = qualified['numVotes'].astype('int')
    qualified['averageRating'] = qualified['averageRating'].astype('float')
    qualified['wr'] = qualified.apply(
        lambda x: (x['numVotes'] / (x['numVotes'] + m) * x['averageRating']) +
                  (m / (m + x['numVotes']) * C),
        axis=1
    )

    # Sort by weighted rating and drop duplicates
    qualified = qualified.drop_duplicates(subset='title')
    qualified = qualified.sort_values('wr', ascending=False)

    # Return the top 10 movies
    return qualified.head(10).reset_index(drop=True)

    
def get_genre_suggestions(partial_input, all_genres):
    partial_input = partial_input.lower()
    suggestions = [genre for genre in all_genres if partial_input in genre]
    return suggestions

# Director-based recommender function
def director_based_recommender(director, dataframe, percentile=0.90):
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
                   (df['averageRating'].notnull())][['title', 'numVotes', 'averageRating', 'popularity', 'poster_url']]
    qualified['numVotes'] = qualified['numVotes'].astype('int')
    qualified['averageRating'] = qualified['averageRating'].astype('int')
    qualified['wr'] = qualified.apply(
        lambda x: (x['numVotes'] / (x['numVotes'] + m) * x['averageRating']) + (m / (m + x['numVotes']) * C),
        axis=1)
    qualified = qualified.drop_duplicates(subset='title')
    return qualified.sort_values('wr', ascending=False).head(10)[['title', 'averageRating', 'poster_url']].reset_index(drop=True)


def get_director_suggestions(partial_input, all_directors):
    partial_input = partial_input.lower()
    suggestions = [director for director in all_directors if partial_input in director]
    return suggestions



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
        ['title', 'numVotes', 'averageRating', 'popularity', 'poster_url']]
    qualified['wr'] = qualified.apply(
        lambda x: (x['numVotes'] / (x['numVotes'] + m) * x['averageRating']) + (
                    m / (m + x['numVotes']) * C),
        axis=1)
    qualified = qualified.drop_duplicates(subset='title')
    return qualified.sort_values('wr', ascending=False).head(10)[['title', 'averageRating', 'poster_url']].reset_index(drop=True)


# Content-based recommender using Jaccard similarity
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def content_based_recommender(title, dataframe, top_n=10):
    if 'title' not in dataframe.columns:
        raise ValueError("'title' sütunu veri çerçevesinde bulunamadı.")

    target_movie = dataframe[dataframe['title'] == title]
    if target_movie.empty:
        return pd.DataFrame(columns=['Film Adı', 'IMDB Rating', 'Poster URL'])

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

            poster_url = row['poster_url'] if 'poster_url' in row and pd.notna(row['poster_url']) else 'Poster bulunamadı'

            recommendations.append({
                'Film Adı': row['title'], 
                'IMDB Rating': row['averageRating'], 
                'Poster URL': poster_url, 
                'Total Score': total_score
            })

    sorted_recommendations = sorted(recommendations, key=lambda x: x['Total Score'], reverse=True)[:top_n]

    return pd.DataFrame(sorted_recommendations).drop(columns=['Total Score']).reset_index(drop=True)



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

    return filtered_df.head(top_n)[['title', 'poster_url']].reset_index(drop=True)

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
    return filtered_df.head(top_n)[['title', 'poster_url']].reset_index(drop=True)
    


# Streamlit App

st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("https://img.freepik.com/free-photo/cinema-elements-arrangement-with-copy-space_23-2148425062.jpg?t=st=1736373050~exp=1736376650~hmac=d2741d5aa576f25d9c80a0d8bb936113b25da687c8450b39196be204aa838ed6&w=1060");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: scroll;
        background-position: center 70%;
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
    <div class="center-title">I.N.K.A. & Chill 🎥 </div>
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

st.markdown(
    """
    <style>
    /* Diğer CSS ayarlarınız burada */
    .stButton>button {
        background-color: #FFD700; /* Buton için Altın Sarısı */
        color: #000000; /* Yazı için Siyah */
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
        font-weight: bold;
        font-size: 16px; /* Yazı boyutu */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* Text Input Kutusunun Yazı Boyutunu ve Stilini Değiştirme */
    .stTextInput > div > input {
        font-size: 18px; /* Yazı boyutunu artırın */
        color: #000000; /* Siyah yazı rengi */
        padding: 10px; /* Kutunun iç boşluğu */
        border-radius: 5px; /* Köşeleri yuvarlatma */
        border: 2px solid #FFD700; /* Altın sarısı çerçeve */
    }
    </style>
    """,
    unsafe_allow_html=True
)

try:
    df = load_data()

    st.sidebar.title("Film Öneri Seçenekleri")
    page = st.sidebar.radio(
    "Hangi türde öneri istiyorsunuz?", 
    options=["Popüler Filmler", "Türe Göre Öneriler", "Yönetmen Seçimine Göre",
             "Oyuncu Seçimine Göre", "Film Benzeri Öneriler", "Anahtar Kelimelere Göre",
             "Ruh Hali Önerileri", "Hakkımızda"])


    if page == "Popüler Filmler":
        if st.button("En beğenilen 10 film için tıklayın"):
            recommendations_simple = simple_recommender_tmdb(df)
            for _, row in recommendations_simple.iterrows():
                st.write(f"**{row['title']}** (IMDB Rating: {row['averageRating']})")
                if row['poster_url']:
                    st.image(row['poster_url'], width=500)
                else:
                    st.write("Poster bulunamadı.")

    elif page == "Türe Göre Öneriler":
        df['genres'] = df['genres'].apply(lambda x: x if isinstance(x, list) else str(x).split(','))
        all_genres = sorted(set(genre.strip().lower() for genres in df['genres'] for genre in genres))
        genre_input = st.text_input("Bir tür girin (örneğin, Action):")
        if genre_input:
            suggestions = get_genre_suggestions(genre_input, all_genres)
            if suggestions:
                st.write("Türler:")
                for suggestion in suggestions[:5]:  
                    st.write(f"- {suggestion.capitalize()}")
                closest_match = suggestions[0]
                recommendations = genre_based_recommender_tmbd_f(df, closest_match)
                if isinstance(recommendations, pd.DataFrame):
                    st.write(f"'{closest_match.capitalize()}' türündeki öneriler:")
                    for _, row in recommendations.iterrows():
                        st.write(f"**{row['title']}** (IMDB Rating: {row['averageRating']:.1f})")
                        if row['poster_url']:
                            st.image(row['poster_url'], width=500)
                        else:
                            st.write("Poster bulunamadı.")
                else:
                    st.write(recommendations)
            else:
                st.write(f"'{genre_input}' ile başlayan tür bulunamadı. Lütfen başka bir tür deneyin.")
        else:
            st.write("Tür için bir şeyler yazmaya başlayın...")


    elif page == "Yönetmen Seçimine Göre":
        director_input = st.text_input("Bir yönetmen ismi girin (örneğin, Christopher Nolan):")
        if director_input:
            recommendations = director_based_recommender(director_input, df)
            if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
                st.write(f"'{director_input}' yönetmeninden öneriler:")
                for _, row in recommendations.iterrows():
                    st.write(f"**{row['title']}** (IMDB Rating: {row['averageRating']:.1f})")
                    if row['poster_url']:
                        st.image(row['poster_url'], width=500)
                    else:
                        st.write("Poster bulunamadı.")
            else:
                st.write(recommendations)
    

    elif page == "Oyuncu Seçimine Göre":
        cast_name = st.text_input("Bir oyuncu ismi girin (örneğin, Christian Bale):")

        if cast_name:
            recommendations = cast_based_recommender_tmdb_f(df, cast_name)

            if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
                st.write(f"'{cast_name}' oyuncusunun yer aldığı filmler:")
                for _, row in recommendations.iterrows():
                    st.write(f"**{row['title']}** (IMDB Rating: {row['averageRating']:.1f})")
                    if row['poster_url']:
                        st.image(row['poster_url'], width=500)
                    else:
                        st.write("Poster bulunamadı.")
            else:
                st.write(recommendations)


    
    elif page == "Film Benzeri Öneriler":
        movie_title = st.text_input("Bir film ismi girin (örneğin, Inception):")

        if movie_title:
            try:
                recommendations = content_based_recommender(movie_title, df)
    
                if recommendations.empty:
                    st.write(f"'{movie_title}' ile ilgili öneri bulunamadı.")
                else:
                    st.write(f"'{movie_title}' benzeri filmler:")
                    for _, row in recommendations.iterrows():
                        st.write(f"**{row['Film Adı']}** (IMDB Rating: {row['IMDB Rating']:.1f})")
                        if row['Poster URL'] and row['Poster URL'] != 'Poster bulunamadı':
                            st.image(row['Poster URL'], width=500)
                        else:
                            st.write("Poster bulunamadı.")
            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")


    elif page == "Anahtar Kelimelere Göre":
        keyword = st.text_input("Bir kelime girin (örneğin, Christmas):")
        if keyword:
            recommendations = keyword_based_recommender(keyword, df)
            st.table(recommendations)

    elif page == "Ruh Hali Önerileri":
        mood = st.text_input("Bir ruh hali girin (örneğin, happy, sad):")
        if mood:
            recommendations = mood_based_recommender(mood, df)
            st.table(recommendations)

    elif page == "Hakkımızda":
        st.title("Hakkımızda")
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
    st.error(f"Bir hata oluştu: {e}")
