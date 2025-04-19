import pandas as pd
import gdown
from surprise import SVD, Reader, Dataset
import streamlit as st
from gensim import corpora, models, similarities
from underthesea import word_tokenize
import os
import ast
from prepare_data import load_and_prepare_data

# Gọi hàm để lấy dữ liệu sau khi file đã được tải xong
products, ratings, final_data = load_and_prepare_data()
# ---------------- Tải dữ liệu từ Google Drive ---------------- #

# Kiểm tra nếu file chưa có trong thư mục hiện tại
if not os.path.exists("cleaned_products.csv"):
    print("🔽 Đang tải Products file từ Google Drive...")
    gdown.download("https://drive.google.com/uc?id=16COzK3fj6pHSb1EBpQ6s-VL3KX5s0ufU", "cleaned_products.csv", quiet=False)

if not os.path.exists("cleaned_ratings.csv"):
    print("🔽 Đang tải Ratings file từ Google Drive...")
    gdown.download("https://drive.google.com/uc?id=16x--zf94wa8IH0mnr9TTT8lKUBwrQ9vk", "cleaned_ratings.csv", quiet=False)

# ---------------- Gensim Model ---------------- #

# Đọc dữ liệu từ các tệp CSV
data = pd.read_csv("cleaned_products.csv")
ratings = pd.read_csv("cleaned_ratings.csv")
data['tokens'] = data['tokens'].apply(ast.literal_eval)

# Tải các mô hình Gensim nếu đã có
if not os.path.exists("models/tfidf_dictionary.dict"):
    print("🔽 Tải hoặc tạo các mô hình Gensim...") 
    # Tạo dictionary và mô hình nếu chưa có
    dictionary = corpora.Dictionary(data['tokens'])
    dictionary.save("models/tfidf_dictionary.dict")
    tfidf = models.TfidfModel(dictionary)
    tfidf.save("models/tfidf_model.tfidf")
    index = similarities.Similarity.load("models/tfidf_index.index")
else:
    dictionary = corpora.Dictionary.load("models/tfidf_dictionary.dict")
    tfidf = models.TfidfModel.load("models/tfidf_model.tfidf")
    index = similarities.Similarity.load("models/tfidf_index.index")

def recommend_gensim(product_id, top_n=10):
    if product_id not in data['product_id'].values:
        return f"Sản phẩm với ID '{product_id}' không có trong dữ liệu."

    idx = data[data['product_id'] == product_id].index[0]
    query_doc = data.loc[idx, 'tokens']
    query_bow = dictionary.doc2bow(query_doc)
    query_tfidf = tfidf[query_bow]

    sims = index[query_tfidf]
    sims = list(enumerate(sims))
    sims = sorted(sims, key=lambda x: x[1], reverse=True)
    sims = [sim for sim in sims if sim[0] != idx]
    top_indices = [i[0] for i in sims[:top_n]]

    return data.iloc[top_indices][['product_id', 'product_name', 'price', 'rating', 'image']]

def recommend_for_user_gensim(user_id, top_n=10):
    liked_products = ratings[(ratings['user_id'] == user_id) & (ratings['rating'] >= 4)]
    if liked_products.empty:
        return f"User {user_id} chưa có đánh giá phù hợp để gợi ý."

    liked_ids = liked_products['product_id'].tolist()
    all_recs = pd.DataFrame()

    for pid in liked_ids:
        recs = recommend_gensim(pid)
        if isinstance(recs, pd.DataFrame):
            all_recs = pd.concat([all_recs, recs])

    if all_recs.empty:
        return f"Không có gợi ý phù hợp cho user {user_id}."

    all_recs = all_recs.drop_duplicates(subset='product_id')
    all_recs = all_recs[~all_recs['product_id'].isin(liked_ids)]

    return all_recs.head(top_n)

# ---------------- SVD Model ---------------- #

import joblib

# Load mô hình đã huấn luyện từ file .pkl
@st.cache_resource
def load_svd_model():
    return joblib.load("models/svd_model.pkl")

algo = load_svd_model()

def has_rated(user_id, product_id, ratings_set):
    return (user_id, product_id) in ratings_set

@st.cache_data
def get_svd_recommendations(user_id, n_recommendations=10):
    ratings_set = set(zip(ratings['user_id'], ratings['product_id']))
    product_ids = products['product_id'].unique()
    unrated_products = [pid for pid in product_ids if not has_rated(user_id, pid, ratings_set)]

    predictions = [(pid, algo.predict(user_id, pid).est) for pid in unrated_products]
    top_preds = sorted(predictions, key=lambda x: x[1], reverse=True)[:n_recommendations]
    top_ids = [r[0] for r in top_preds]

    recs = products[products['product_id'].isin(top_ids)].copy()
    recs['predicted_rating'] = recs['product_id'].map(dict(top_preds))

    return recs[['product_id', 'product_name', 'price', 'rating', 'image', 'link', 'description', 'predicted_rating']]