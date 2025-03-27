
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from annoy import AnnoyIndex
import streamlit as st
import os

# Streamlit configuration
st.set_page_config(page_title='Fast Movie Recommender System', page_icon='🎬', layout='wide')
st.title('🎬 Fast Movie Recommender System')
st.write('Get quick movie recommendations based on title and genres!')

# Load dataset with caching
@st.cache_data
def load_data():
    return pd.read_csv('new_movies.csv', usecols=['title', 'genres'])

movies = load_data()

# Preprocessing
movies['title'] = movies['title'].str.lower()
movies['genres'] = movies['genres'].fillna('').str.lower()

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
tfidf_matrix = tfidf.fit_transform(movies['title'] + ' ' + movies['genres'])

# Annoy Index Setup
f = tfidf_matrix.shape[1]
index = AnnoyIndex(f, 'angular')
index_path = 'annoy_index.ann'

if not os.path.exists(index_path):
    for i, vector in enumerate(tfidf_matrix.toarray()):
        index.add_item(i, vector)
    index.build(10)
    index.save(index_path)
else:
    index.load(index_path)

# Initialize session state
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = ''
if 'search_completed' not in st.session_state:
    st.session_state.search_completed = False
if 'possible_matches' not in st.session_state:
    st.session_state.possible_matches = []

def recommend_movies(title, num_recommendations=5):
    title = title.strip().lower()

    if title not in movies['title'].values:
        possible_matches = movies[movies['title'].str.contains(title, na=False)]['title'].tolist()
        return possible_matches if possible_matches else 'Movie not found in the database.'

    idx = movies[movies['title'] == title].index[0]
    similar_indices = index.get_nns_by_item(idx, num_recommendations + 1)[1:]

    return movies['title'].iloc[similar_indices].str.title().tolist()

# Input form
with st.form('movie_form', clear_on_submit=False):
    movie_title = st.text_input('Enter a movie title:', value=st.session_state.selected_movie)
    submitted = st.form_submit_button('Recommend')

if submitted and movie_title:
    st.session_state.search_completed = True
    st.session_state.selected_movie = movie_title
    st.session_state.possible_matches = recommend_movies(movie_title)

if isinstance(st.session_state.possible_matches, list) and st.session_state.possible_matches:
    selected_match = st.selectbox('Did you mean one of these?', ['Select a movie'] + st.session_state.possible_matches)
    if selected_match != 'Select a movie':
        st.session_state.selected_movie = selected_match
        recommendations = recommend_movies(selected_match)
        st.subheader(f'Movies similar to "{selected_match.title()}":')
        for i, movie in enumerate(recommendations, 1):
            st.write(f'{i}. {movie}')
elif st.session_state.search_completed:
    st.warning(st.session_state.possible_matches)

if st.session_state.search_completed:
    if st.button('New Search'):
        for key in ['selected_movie', 'search_completed', 'possible_matches']:
            st.session_state[key] = ''
