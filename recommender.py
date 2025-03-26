import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load dataset
movies = pd.read_csv('new_movies.csv')

# Preprocessing: Convert to lowercase and fill NaNs
movies['title'] = movies['title'].str.lower().fillna('')
movies['genres'] = movies['genres'].str.lower().fillna('')

# Combine title and genres for TF-IDF
movies['features'] = movies['title'] + ' ' + movies['genres']

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['features'])

# Compute cosine similarity
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Streamlit UI Configuration
st.set_page_config(page_title="Movie Recommender System", page_icon="🎬", layout="wide")
st.title("🎬 Enhanced Movie Recommender System")
st.write("Get personalized movie recommendations based on title and genres!")

# Initialize session state
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

# Streamlit input form
with st.form("movie_form", clear_on_submit=False):
    movie_title = st.text_input("Enter a movie title:", value=st.session_state.selected_movie)
    submitted = st.form_submit_button("Recommend")

if submitted and movie_title:
    st.session_state.search_completed = True
    st.session_state.selected_movie = movie_title
    st.session_state.possible_matches = recommend_movies(movie_title)

if isinstance(st.session_state.possible_matches, list):
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
