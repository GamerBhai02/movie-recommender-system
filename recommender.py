import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load dataset
movies = pd.read_csv('movies.csv')

# Preprocessing: Convert to lowercase
movies['title'] = movies['title'].str.lower()

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['title'])

# Compute cosine similarity
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Streamlit UI Configuration
st.set_page_config(page_title="Movie Recommender System", page_icon="🎬", layout="wide")
st.title("🎬 Movie Recommender System")
st.write("Get movie recommendations based on your favorite movie!")

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
        # Get possible matches
        possible_matches = movies[movies['title'].str.contains(title, na=False)]['title'].tolist()
        
        if possible_matches:
            return possible_matches  # Return possible matches to select from
        else:
            return "Movie not found in the database."

    # Find index of the input movie
    idx = movies[movies['title'] == title].index[0]

    # Get similarity scores for all movies
    sim_scores = list(enumerate(similarity_matrix[idx]))

    # Sort by similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top recommendations excluding the input movie itself
    sim_scores = sim_scores[1:num_recommendations + 1]
    movie_indices = [i[0] for i in sim_scores]

    # Return the recommended movie titles
    recommendations = movies['title'].iloc[movie_indices].str.title().tolist()
    return recommendations

# Input field with form for "Enter" key submission
with st.form("movie_form", clear_on_submit=False):
    movie_title = st.text_input("Enter a movie title:", value=st.session_state.selected_movie)
    submitted = st.form_submit_button("Recommend")

if submitted and movie_title:
    st.session_state.search_completed = True
    st.session_state.selected_movie = movie_title
    st.session_state.possible_matches = recommend_movies(movie_title)

# If possible matches are found and a search is completed
if (
    st.session_state.search_completed 
    and isinstance(st.session_state.possible_matches, list) 
    and all(isinstance(item, str) for item in st.session_state.possible_matches)
):
    selected_match = st.selectbox("Did you mean one of these?", ["Select a movie"] + st.session_state.possible_matches)

    if selected_match != "Select a movie":
        st.session_state.selected_movie = selected_match
        recommendations = recommend_movies(selected_match)

        st.subheader(f"Movies similar to '{selected_match.title()}':")
        for i, movie in enumerate(recommendations, 1):
            st.write(f"{i}. {movie}")

# If there are no possible matches and search is complete
elif st.session_state.search_completed and not isinstance(st.session_state.possible_matches, list):
    st.warning(st.session_state.possible_matches)

# Show "New Search" button only after search is completed
if st.session_state.search_completed:
    if st.button("New Search"):
        # Clear session state for a fresh start
        for key in ["selected_movie", "search_completed", "possible_matches"]:
            st.session_state[key] = ""
