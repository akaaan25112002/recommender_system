import pandas as pd
import gdown
import os
import ast
from surprise import SVD, Reader, Dataset
import streamlit as st
from gensim import corpora, models, similarities
from prepare_data import load_and_prepare_data

# Gọi hàm để lấy dữ liệu đã chuẩn bị
products, ratings, final_data = load_and_prepare_data()

# ---------------- Tải dữ liệu nếu chưa có ---------------- #
import requests

# Hàm tải file từ Google Drive
def download_from_google_drive(file_id, destination):
    URL = f"https://drive.google.com/uc?id={file_id}&export=download"
    session = requests.Session()
    response = session.get(URL, stream=True)
    
    with open(destination, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
    
    print(f"File saved as {destination}")
    
if not os.path.exists("cleaned_products.csv"):
    print("🔽 Đang tải cleaned_products.csv từ Google Drive...")
    gdown.download("https://drive.google.com/uc?id=16COzK3fj6pHSb1EBpQ6s-VL3KX5s0ufU", "cleaned_products.csv", quiet=False)

if not os.path.exists("cleaned_ratings.csv"):
    print("🔽 Đang tải cleaned_ratings.csv từ Google Drive...")
    gdown.download("https://drive.google.com/uc?id=16x--zf94wa8IH0mnr9TTT8lKUBwrQ9vk", "cleaned_ratings.csv", quiet=False)

if not os.path.exists("models/svd_model.pkl"):
    print("🔽 Đang tải mô hình SVD từ Google Drive...")
    os.makedirs("models", exist_ok=True)
    gdown.download("https://drive.google.com/uc?id=1rrchIE01BwYNo0EAfs_hW_WVMv5Sb3eY", "models/svd_model.pkl", quiet=False)

# ---------------- Gensim TF-IDF ---------------- #

# Đọc lại dữ liệu sản phẩm đã lưu
data = pd.read_csv("cleaned_products.csv")
data['tokens'] = data['tokens'].apply(ast.literal_eval)

# Load hoặc tạo dictionary, tfidf và index
if not os.path.exists("models/tfidf_dictionary.dict"):
    print("🔽 Đang tạo Gensim models...")
    dictionary = corpora.Dictionary(data['tokens'])
    dictionary.save("models/tfidf_dictionary.dict")
    tfidf = models.TfidfModel(dictionary)
    tfidf.save("models/tfidf_model.tfidf")
    index = similarities.Similarity(output_prefix="models/tfidf_index", corpus=[tfidf[dictionary.doc2bow(text)] for text in data['tokens']], num_features=len(dictionary))
    index.save("models/tfidf_index.index")
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

# ---------------- SVD Collaborative Filtering ---------------- #

import joblib

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
