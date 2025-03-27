# 🎬 MovieLens Recommendation System
This project explores the MovieLens dataset through a series of machine learning approaches including collaborative filtering, content-based filtering, regression, and classification. It also includes a Streamlit app (optional) for interactive recommendations.

## 📁 Project Structure
movie-recommender/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── ratings.csv
│   ├── movies.csv
│   └── ...
├── plots/
│   ├── classification_roc_curve.png
│   ├── classification_pr_curve.png
│   ├── RandomForest_confusion_matrix.png
│   ├── regression_model_results.csv
│   └── ...
├── classification_model.py
├── collaborative_filtering.py
├── content_based_recommender.py
├── eda.py
├── regression_model.py
├── requirements.txt
└── README.md


## 📊 Models Included
### ✅ Content-Based Filtering
Recommends movies based on genre similarity using TF-IDF and cosine similarity.

### ✅ Classification
Predicts whether a user will like a movie (rating ≥ 4) using:
- Random Forest
- Logistic Regression
- ROC and Precision-Recall curves
- Confusion matrix and classification report

### ✅ Regression
Predicts a user's movie rating using:
- Random Forest Regressor
- Linear Regression
- Gradient Boosting
- RMSE, MAE, R² metrics
- Feature importance and prediction plots

## 🚀 How to Run
## 📥 How to Download the Dataset

To avoid including large files in the repo, we provide a script to download the MovieLens dataset.

"bash"
python download_data.py

or 
link to kaggle dataset : https://www.kaggle.com/datasets/garymk/movielens-25m-dataset
### Set up virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

