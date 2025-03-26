import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import joblib

# Streamlit Configuration
st.set_page_config(page_title="Enhanced Movie Recommender", page_icon="🎥", layout="wide")
st.title("🎥 Enhanced Movie Recommender System")
st.write("Discover movies based on your favorite titles and genres!")

# Cache data loading
@st.cache_data
def load_data():
    cols = ['title', 'genres']
    return pd.read_csv('new_movies.csv', usecols=cols)

movies = load_data()

# Convert to lowercase for consistency
movies['title'] = movies['title'].str.lower()
movies['genres'] = movies['genres'].str.lower()

# Load or compute similarity matrix
try:
    similarity_matrix = joblib.load('similarity_matrix.pkl')
except:
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(movies['title'] + ' ' + movies['genres'])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    joblib.dump(similarity_matrix, 'similarity_matrix.pkl')

# Session state initialization
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = ""
if "search_completed" not in st.session_state:
    st.session_state.search_completed = False
if "possible_matches" not in st.session_state:
    st.session_state.possible_matches = []

def recommend_movies(title, num_recommendations=5):
    title = title.strip().lower()

    if title not in movies['title'].values:
        possible_matches = movies[movies['title'].str.contains(title, na=False)]['title'].tolist()
        if possible_matches:
            return possible_matches
        else:
            return "Movie not found in the database."

    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]
    movie_indices = [i[0] for i in sim_scores]

    recommendations = movies['title'].iloc[movie_indices].str.title().tolist()
    return recommendations

with st.form("movie_form", clear_on_submit=False):
    movie_title = st.text_input("Enter a movie title:", value=st.session_state.selected_movie)
    submitted = st.form_submit_button("Search")

if submitted and movie_title:
    st.session_state.search_completed = True
    st.session_state.selected_movie = movie_title
    st.session_state.possible_matches = recommend_movies(movie_title)

if isinstance(st.session_state.possible_matches, list) and all(isinstance(item, str) for item in st.session_state.possible_matches):
    selected_match = st.selectbox("Did you mean one of these?", ["Select a movie"] + st.session_state.possible_matches)

    if selected_match != "Select a movie":
        st.session_state.selected_movie = selected_match
        recommendations = recommend_movies(selected_match)

        st.subheader(f"Movies similar to '{selected_match.title()}':")
        for i, movie in enumerate(recommendations, 1):
            st.write(f"{i}. {movie}")

elif st.session_state.search_completed and not isinstance(st.session_state.possible_matches, list):
    st.warning(st.session_state.possible_matches)

if st.session_state.search_completed:
    if st.button("New Search"):
        for key in ["selected_movie", "search_completed", "possible_matches"]:
            st.session_state[key] = ""
