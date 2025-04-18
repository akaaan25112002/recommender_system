import streamlit as st
import pandas as pd
import numpy as np
from data_loader import load_processed_data, load_ratings
from recommender import recommend_for_user_gensim, get_svd_recommendations
from utils import clean_text_row, stop_words
from underthesea import word_tokenize
import base64
import unicodedata
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# ===== Helper Functions =====

def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        return f"data:image/png;base64,{encoded}"
    except:
        return ""

def preprocess_text(text):
    tokens = word_tokenize(text, format="text").split()
    return clean_text_row(tokens, stop_words)

def remove_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')

# ===== Load Data =====

data, dictionary, tfidf, index = load_processed_data()
ratings = load_ratings()
products = data.dropna(subset=["Content", "product_name"]).copy()

# Tạo thêm cột không dấu để tìm kiếm
products['product_name_clean'] = products['product_name'].apply(lambda x: remove_accents(str(x).lower()))

# ===== Sidebar =====
with st.sidebar:
    st.title("🔍 Shopee Recommender")
    st.markdown("### ℹ️ Thông tin mô hình")
    st.markdown("""
    - **Phương pháp**: Content-Based & Collaborative Filtering  
    - **Mô hình**: Gensim TF-IDF & Surprise SVD  
    - **Nguồn dữ liệu**: Shopee Products & Ratings  
    - **Số lượng sản phẩm**: 49,663  
    - **Tương tác người dùng**: 990,000+
    """)
    st.markdown("---")
    selected_tab = st.radio("Chọn chức năng", ["Sản phẩm nổi bật", "Tìm kiếm sản phẩm", "Gợi ý cho bạn"])
    st.markdown("Made with ❤️ by Anh Khoa & Thiên Bảo")

# ===== Default Image =====
default_image_path = "Data/No_image_available.svg.png"
default_image_url = image_to_base64(default_image_path)

# ===== Tab 1: Sản phẩm nổi bật =====
if selected_tab == "Sản phẩm nổi bật":
    st.header("🔥 Sản phẩm nổi bật theo danh mục")
    sub_categories = ['Tất cả'] + products['sub_category'].dropna().unique().tolist()
    selected_sub_category = st.selectbox("Chọn danh mục con", options=sub_categories)

    if selected_sub_category == "Tất cả":
        top_items = products.sort_values(by='rating', ascending=False).head(15)
    else:
        top_items = products[products['sub_category'] == selected_sub_category].sort_values(by='rating', ascending=False).head(15)

    for _, row in top_items.iterrows():
        image_url = row.image if pd.notna(row.image) and str(row.image).startswith("http") else default_image_url
        st.image(image_url, width=180)
        st.markdown(f"**🛍️ {row.product_name}**")
        st.markdown(f"⭐ {row.rating} | 💵 {int(row.price):,}")
        st.markdown("---")

# ===== Tab 2: Tìm kiếm sản phẩm =====
elif selected_tab == "Tìm kiếm sản phẩm":
    st.header("🔎 Tìm kiếm sản phẩm")
    search_input = st.text_input("Nhập từ khóa tìm kiếm:")
    selected_sub = st.selectbox("Chọn danh mục con", options=["Tất cả"] + list(products['sub_category'].dropna().unique()))

    if search_input:
        query_clean = remove_accents(search_input.lower())
        matched = products[products['product_name_clean'].str.contains(query_clean)]

        if selected_sub != "Tất cả":
            matched = matched[matched['sub_category'] == selected_sub]

        if matched.empty:
            st.warning("Không tìm thấy sản phẩm phù hợp.")
        else:
            chosen = matched.iloc[0]
            st.success(f"🎯 Gợi ý các sản phẩm tương tự với: {chosen.product_name}")
            query_doc = dictionary.doc2bow(preprocess_text(chosen.Content))
            sims = index[tfidf[query_doc]]
            top_indices = np.argsort(sims)[::-1][1:11]

            for idx in top_indices:
                product = products.iloc[idx]
                image_url = product.image if pd.notna(product.image) and str(product.image).startswith("http") else default_image_url
                st.image(image_url, width=180)
                st.markdown(f"**🛍️ {product.product_name}**")
                st.markdown(f"⭐ {product.rating} | 💵 {int(product.price):,}")
                st.markdown("---")

# ===== Tab 3: Gợi ý cho bạn =====
elif selected_tab == "Gợi ý cho bạn":
    st.header("🎁 Gợi ý sản phẩm cho bạn")
    user_id = st.text_input("Nhập ID người dùng:")
    method = st.radio("Chọn phương pháp gợi ý", ["SVD (Collaborative)", "Gensim (Content-Based)"])

    if user_id:
        with st.spinner("Đang tạo gợi ý..."):
            if method == "SVD (Collaborative)":
                recs = get_svd_recommendations(user_id)
            else:
                recs = recommend_for_user_gensim(user_id)

        if isinstance(recs, pd.DataFrame):
            for _, row in recs.iterrows():
                image_url = row.image if pd.notna(row.image) and str(row.image).startswith("http") else default_image_url
                st.image(image_url, width=180)
                st.markdown(f"**🛍️ {row.product_name}**")
                st.markdown(f"⭐ {row.rating} | 💵 {int(row.price):,}")
                if "predicted_rating" in row:
                    st.markdown(f"📊 Dự đoán đánh giá: **{row.predicted_rating:.2f}**")
                st.markdown("---")
        elif isinstance(recs, str):
            st.warning(recs)
        else:
            st.error("Không thể tạo gợi ý. Vui lòng kiểm tra dữ liệu hoặc thử lại sau.")