import pandas as pd
import matplotlib.pyplot as plt


# === Step 1: Load data ===
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")
tags = pd.read_csv("data/tags.csv")
genome_tags = pd.read_csv("data/genome-tags.csv")
genome_scores = pd.read_csv("data/genome-scores.csv")
links = pd.read_csv("data/links.csv")

# === Step 2: Inspect data ===
print("Movies:\n", movies.head(), "\n")
print("Ratings:\n", ratings.head(), "\n")
print("Tags:\n", tags.head(), "\n")
print("Genome Tags:\n", genome_tags.head(), "\n")
print("Genome Scores:\n", genome_scores.head(), "\n")
print("Links:\n", links.head(), "\n")

# Check for missing values
print("Missing values:")
print("Movies:", movies.isnull().sum())
print("Ratings:", ratings.isnull().sum())
print("Tags:", tags.isnull().sum())

# === Step 3: Basic statistics and visualizations ===
# Ratings distribution
ratings['rating'].value_counts().sort_index().plot(kind='bar')
plt.title("Ratings Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# Most rated movies
top_rated = ratings['movieId'].value_counts().head(10).reset_index()
top_rated.columns = ['movieId', 'rating_count']
top_movies = pd.merge(top_rated, movies, on='movieId')
print("Top 10 Most Rated Movies:\n", top_movies[['title', 'rating_count']])
