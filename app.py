import streamlit as st
import pandas as pd
import numpy as np
from gensim.downloader import load
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import os

# Title of the app
st.set_page_config(page_title="Mantra Shashtra", layout="centered")
st.title("Mantra Shasthra")

# Load Word2Vec model
@st.cache_resource
def load_word2vec():
    return load("word2vec-google-news-300")

# Load the dataset
@st.cache_data
def load_data():
    file_path = os.path.join(os.getcwd(), "newdata.xlsx")
    return pd.read_excel(file_path, sheet_name="Sheet1")

# Get sentence vector by averaging Word2Vec vectors
def get_sentence_vector(sentence, model):
    words = [word for word in sentence if word in model]
    if not words:
        return np.random.rand(model.vector_size)
    return np.mean([model[word] for word in words], axis=0)

# Get similar words using Word2Vec
def get_word2vec_synonyms(word, top_n=2):
    try:
        if word in word2vec_model:
            return [w for w, _ in word2vec_model.most_similar(word, topn=top_n)]
    except:
        pass
    return []

# Expand the user query using synonyms
def expand_query(query):
    words = query.lower().split()
    expanded = set(words)
    for word in words:
        expanded.update(get_word2vec_synonyms(word))
    return list(expanded)

# Recommend videos based on a query
def recommend_videos(query, model, word2vec, data, n_recommendations=5):
    query_vector = get_sentence_vector(expand_query(query), word2vec)
    n_recommendations = min(n_recommendations, len(data))
    distances, indices = model.kneighbors([query_vector], n_neighbors=n_recommendations)
    valid_indices = [idx for idx in indices[0] if idx < len(data)]
    recommendations = data.iloc[valid_indices].copy()
    recommendations['Similarity'] = 1 - distances[0][:len(valid_indices)]
    return recommendations[['Keywords', 'Link', 'Similarity']]

# Load everything
with st.spinner("Loading data and model..."):
    word2vec_model = load_word2vec()
    df = load_data()

    # Prepare training data
    keywords = df['Keywords'].fillna('').str.lower().str.split()
    vectors = np.array([get_sentence_vector(sentence, word2vec_model) for sentence in keywords])
    knn_model = NearestNeighbors(n_neighbors=min(5, len(df)), metric='cosine')
    knn_model.fit(vectors)

# Input from user
user_query = st.text_input("Enter a keyword or topic:", placeholder="e.g., health")

if user_query:
    results = recommend_videos(user_query, knn_model, word2vec_model, df)

    if results.empty:
        st.warning("😕 No relevant videos found. Try a different keyword.")
    else:
        st.success(f"Top recommendations for: **{user_query}**")
        for _, row in results.iterrows():
            st.video(row['Link'])
            st.caption(f"🔑 Keywords: {row['Keywords']} | 🔍 Similarity: {row['Similarity']:.2f}")
