import pandas as pd
 from surprise import SVD, Reader, Dataset
 import streamlit as st
 from gensim import corpora, models, similarities
 from prepare_data import products, ratings, final_data
 from underthesea import word_tokenize
 
 # ---------------- Gensim Model ---------------- #
 def build_gensim_index(products):
     # Tokenize dữ liệu
     documents = products['description_clean'].apply(lambda x: word_tokenize(str(x), format="text")).tolist()
     tokenized_docs = [doc.split() for doc in documents]
 
     # Tạo dictionary và TF-IDF model
     dictionary = corpora.Dictionary(tokenized_docs)
     corpus = [dictionary.doc2bow(text) for text in tokenized_docs]
     tfidf_model = models.TfidfModel(corpus)
 
     # Tạo index tương đồng
     index = similarities.Similarity(output_prefix=None, corpus=tfidf_model[corpus], num_features=len(dictionary))
     
     return dictionary, tfidf_model, index
 
 # Khởi tạo dictionary, tfidf, index
 dictionary, tfidf, index = build_gensim_index(products)
 # Load mô hình và dữ liệu Gensim
 dictionary = corpora.Dictionary.load("models/tfidf_dictionary.dict")
 tfidf = models.TfidfModel.load("models/tfidf_model.tfidf")
 index = similarities.Similarity.load("models/tfidf_index.index")
 
 # Gợi ý sản phẩm tương tự một sản phẩm cụ thể
 def recommend_gensim(product_id, top_n=10):
     if product_id not in products['product_id'].values:
         return f"Sản phẩm với ID '{product_id}' không có trong dữ liệu."
 
     idx = products[products['product_id'] == product_id].index[0]
     query_doc = products.loc[idx, 'tokens']
     query_bow = dictionary.doc2bow(query_doc)
     query_tfidf = tfidf[query_bow]
 
     sims = index[query_tfidf]
     sims = list(enumerate(sims))
     sims = sorted(sims, key=lambda x: x[1], reverse=True)
     sims = [sim for sim in sims if sim[0] != idx]
     top_indices = [i[0] for i in sims[:top_n]]
 
     return products.iloc[top_indices][['product_id', 'product_name', 'price', 'rating', 'image', 'link']]
 
 # Gợi ý cho user dựa trên các sản phẩm user thích (Gensim)
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
 @st.cache_resource
 def train_svd_model():
     final_data_clean = final_data.dropna(subset=['user_id', 'product_id', 'rating_x'])
     reader = Reader(rating_scale=(1, 5))
     dataset = Dataset.load_from_df(final_data_clean[['user_id', 'product_id', 'rating_x']], reader)
 
     algo = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
     trainset = dataset.build_full_trainset()
     algo.fit(trainset)
 
     return algo
 
 # Huấn luyện sẵn từ đầu
 algo = train_svd_model()
 
 # Kiểm tra đã đánh giá
 def has_rated(user_id, product_id, ratings_set):
     return (user_id, product_id) in ratings_set
 
 # Gợi ý theo SVD
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
