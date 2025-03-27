import os
import requests
import zipfile

# Folder to store data
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# URL to download (Kaggle or GroupLens)
url = "https://www.kaggle.com/datasets/garymk/movielens-25m-dataset"
zip_path = os.path.join(DATA_DIR, "movielens-25m-dataset")

print("📥 Downloading MovieLens dataset...")
response = requests.get(url, stream=True)
with open(zip_path, 'wb') as f:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            f.write(chunk)

print("📦 Extracting dataset...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR)

print("✅ Dataset ready in 'data/' folder.")
