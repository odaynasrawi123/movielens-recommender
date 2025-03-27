import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movies
movies = pd.read_csv("data/movies.csv")

# Preprocess genres
movies['processed_genres'] = movies['genres'].str.replace('|', ' ', regex=False).str.lower()

# TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(movies['processed_genres'])

# Ask for user input
user_input = input("Enter the types of movies you like (e.g. action, sci-fi, drama): ").lower()

# Transform user input into TF-IDF vector
user_vec = vectorizer.transform([user_input])

# Compute cosine similarity
similarity_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()

# Get top 5 recommended movie indices
top_indices = similarity_scores.argsort()[::-1][:5]

# Show top 5 recommended movies
print("\n🎬 Top 5 Recommended Movies for You:\n")
for idx in top_indices:
    title = movies.iloc[idx]['title']
    genres = movies.iloc[idx]['genres']
    print(f"{title} ({genres})")
