import streamlit as st
import pandas as pd
import numpy as np
from data_loader import load_processed_data, load_ratings
from recommender import recommend_gensim, recommend_for_user_gensim
from utils import clean_text_row, stop_words
from underthesea import word_tokenize
import streamlit.components.v1 as components
import streamlit as st
import pandas as pd
from recommender import train_svd_model, get_svd_recommendations
from surprise import Reader, Dataset, SVD
import os
import base64


# Load dữ liệu
data, dictionary, tfidf, index = load_processed_data()
ratings = load_ratings()
products = data  # Đặt tên products cho nhất quán với các phần bên dưới

# Hàm xử lý text nhập để tìm kiếm
def preprocess_text(text):
    tokens = word_tokenize(text, format="text").split()
    return clean_text_row(tokens, stop_words)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.title("🔍 Shopee Recommender")
    st.markdown("### ℹ️ Thông tin mô hình")
    st.markdown("""
    - **Phương pháp**: Content-Based Filtering  
    - **Mô hình**: content-based Filtering: Gensim TF-IDF Collaborative Filtering: Surprise SVD
    - **Nguồn dữ liệu**: Shopee Products & Ratings  
    - **Số lượng sản phẩm**: 49,663  
    - **Tương tác người dùng**: 990,000+
    """)
    st.markdown("---")
    selected_tab = st.radio("Chọn chức năng", ["Sản phẩm nổi bật", "Tìm kiếm sản phẩm", "Gợi ý cho bạn"])
    st.markdown("Made with ❤️ by Anh Khoa & Thiên Bảo")

# -------------------- Tab 1: Sản phẩm nổi bật --------------------
# Hàm chuyển ảnh thành base64 (dùng cho ảnh mặc định)
def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        print(f"Lỗi khi đọc ảnh mặc định: {e}")
        return ""

# Tạo đường dẫn an toàn cho ảnh mặc định
current_dir = os.path.dirname(os.path.abspath(__file__))
default_image_path = os.path.join(current_dir, "Data", "No_image_available.svg.png")
default_image_url = image_to_base64(default_image_path)

# === TAB: Sản phẩm nổi bật ===
if selected_tab == "Sản phẩm nổi bật":
    st.header("🔥 Sản phẩm nổi bật theo danh mục")

    sub_categories = ['Tất cả'] + products['sub_category'].unique().tolist()
    selected_sub_category = st.selectbox("Chọn danh mục con", options=sub_categories)

    if selected_sub_category == "Tất cả":
        top_items = products.sort_values(by='rating', ascending=False).head(15)
    else:
        top_items = products[products['sub_category'] == selected_sub_category].sort_values(by='rating', ascending=False).head(15)

    html_content = f"""
    <style>
        .scroll-wrapper {{
            position: relative;
            margin-bottom: 20px;
        }}
        .scroll-container {{
            display: flex;
            overflow-x: auto;
            scroll-behavior: smooth;
            padding: 10px 0;
            white-space: nowrap;
        }}
        .scroll-container::-webkit-scrollbar {{
            height: 8px;
        }}
        .scroll-container::-webkit-scrollbar-thumb {{
            background-color: #888;
            border-radius: 4px;
        }}
        .product-card {{
            flex: 0 0 auto;
            width: 250px;
            background-color: #111;
            color: white;
            border: 1px solid #333;
            border-radius: 10px;
            padding: 15px;
            margin-right: 15px;
            box-shadow: 1px 1px 5px rgba(0,0,0,0.2);
            overflow: hidden;
            white-space: normal;
            box-sizing: border-box;
            height: 380px;
        }}
        .scroll-btn {{
            position: absolute;
            top: 35%;
            background-color: rgba(0,0,0,0.6);
            color: white;
            border: none;
            padding: 8px 12px;
            font-size: 18px;
            border-radius: 5px;
            cursor: pointer;
            z-index: 2;
        }}
        .scroll-btn:hover {{
            background-color: rgba(255,255,255,0.2);
        }}
        .scroll-left {{
            left: 0;
        }}
        .scroll-right {{
            right: 0;
        }}
        .product-card img {{
            width: 100%;
            height: 180px;
            object-fit: cover;
            border-radius: 8px;
        }}
        .product-card h4 {{
            font-size: 16px;
            margin-top: 10px;
        }}
        .product-card p {{
            font-size: 14px;
            margin: 5px 0;
        }}
        .product-card .description {{
            font-size: 12px;
            color: #bbb;
            line-height: 1.5;
            max-height: 100px;
            overflow-y: auto;
            text-overflow: ellipsis;
            white-space: normal;
            width: 100%;
            margin-top: 8px;
        }}
    </style>

    <div class="scroll-wrapper">
        <button class="scroll-btn scroll-left" onclick="document.getElementById('scroll_all').scrollBy({{left: -300, behavior: 'smooth'}})">←</button>
        <button class="scroll-btn scroll-right" onclick="document.getElementById('scroll_all').scrollBy({{left: 300, behavior: 'smooth'}})">→</button>
        <div id="scroll_all" class="scroll-container">
    """

    for _, row in top_items.iterrows():
        # Kiểm tra ảnh hợp lệ
        image_url = row.image if pd.notna(row.image) and row.image.startswith("http") else default_image_url

        html_content += f"""
            <div class="product-card">
                <img src="{image_url}" />
                <h4>🛍️ {row.product_name[:40]}</h4>
                <p>⭐ {row.rating} | 💵 {int(row.price):,}</p>
                <div class="description">{row.description[:150]}...</div>
            </div>
        """

    html_content += "</div></div>"

    components.html(html_content, height=400)


# -------------------- Tab 2: Tìm kiếm sản phẩm --------------------
elif selected_tab == "Tìm kiếm sản phẩm":
    st.header("🔎 Tìm kiếm sản phẩm")
    search_input = st.text_input("Nhập từ khóa tìm kiếm:")

    filter_col1 = st.selectbox("Chọn danh mục con", options=["Tất cả"] + list(products['sub_category'].unique()))

    if search_input:
        matched = products[products['product_name'].str.contains(search_input, case=False)]
        if filter_col1 != "Tất cả":
            matched = matched[matched['sub_category'] == filter_col1]

        if matched.empty:
            st.warning("Không tìm thấy sản phẩm phù hợp.")
        else:
            chosen = matched.iloc[0]
            st.success(f"🎯 Gợi ý các sản phẩm tương tự với: {chosen.product_name}")

            query_doc = dictionary.doc2bow(preprocess_text(chosen.Content))
            sims = index[tfidf[query_doc]]
            top_indices = np.argsort(sims)[::-1][1:11]  # Lấy top 10 sản phẩm

            html_content = f"""
            <style>
                .scroll-wrapper {{
                    position: relative;
                    margin-bottom: 20px;
                }}
                .scroll-container {{
                    display: flex;
                    overflow-x: auto;
                    scroll-behavior: smooth;
                    padding: 10px 0;
                    white-space: nowrap;
                }}
                .scroll-container::-webkit-scrollbar {{
                    height: 8px;
                }}
                .scroll-container::-webkit-scrollbar-thumb {{
                    background-color: #888;
                    border-radius: 4px;
                }}
                .product-card {{
                    flex: 0 0 auto;
                    width: 250px;
                    background-color: #111;
                    color: white;
                    border: 1px solid #333;
                    border-radius: 10px;
                    padding: 15px;
                    margin-right: 15px;
                    box-shadow: 1px 1px 5px rgba(0,0,0,0.2);
                    overflow: hidden;
                    white-space: normal;
                    box-sizing: border-box;
                    height: 380px;
                }}
                .scroll-btn {{
                    position: absolute;
                    top: 35%;
                    background-color: rgba(0,0,0,0.6);
                    color: white;
                    border: none;
                    padding: 8px 12px;
                    font-size: 18px;
                    border-radius: 5px;
                    cursor: pointer;
                    z-index: 2;
                }}
                .scroll-btn:hover {{
                    background-color: rgba(255,255,255,0.2);
                }}
                .scroll-left {{
                    left: 0;
                }}
                .scroll-right {{
                    right: 0;
                }}
                .product-card img {{
                    width: 100%;
                    height: 180px;
                    object-fit: cover;
                    border-radius: 8px;
                }}
                .product-card h4 {{
                    font-size: 16px;
                    margin-top: 10px;
                }}
                .product-card p {{
                    font-size: 14px;
                    margin: 5px 0;
                }}
                .product-card .description {{
                    font-size: 12px;
                    color: #bbb;
                    line-height: 1.5;
                    max-height: 100px;
                    overflow-y: auto;
                    text-overflow: ellipsis;
                    white-space: normal;
                    width: 100%;
                    margin-top: 8px;
                }}
            </style>

            <div class="scroll-wrapper">
                <button class="scroll-btn scroll-left" onclick="document.getElementById('scroll_all').scrollBy({{left: -300, behavior: 'smooth'}})">←</button>
                <button class="scroll-btn scroll-right" onclick="document.getElementById('scroll_all').scrollBy({{left: 300, behavior: 'smooth'}})">→</button>
                <div id="scroll_all" class="scroll-container">
            """

            # Render các sản phẩm
            for idx in top_indices:
                row = products.iloc[idx]

                # Kiểm tra ảnh hợp lệ
                image_url = row.image if pd.notna(row.image) and row.image.startswith("http") else default_image_url

                html_content += f"""
                    <div class="product-card">
                        <img src="{image_url}" />
                        <h4>🛍️ {row.product_name[:40]}</h4>
                        <p>⭐ {row.rating} | 💵 {int(row.price):,}</p>
                        <div class="description">{row.description[:150]}...</div>
                    </div>
                """

            html_content += "</div></div>"
            components.html(html_content, height=400)



# -------------------- Tab 3: Gợi ý cá nhân hóa --------------------
from recommender import train_svd_model, get_svd_recommendations  # Import từ recommender.py
from prepare_data import products, ratings  # Import dữ liệu từ prepare_data
from prepare_data import final_data

if selected_tab == "Gợi ý cho bạn":
    st.header("🎁 Gợi ý cá nhân hóa")
    user_id_input = st.text_input("Nhập mã khách hàng (1 số ví dụ 1,2,3,...):")

    if user_id_input:
        user_id_input = user_id_input.strip()

        if user_id_input.isdigit():
            user_id = int(user_id_input)
            user_rated = final_data[(final_data['user_id'] == user_id) & (final_data['rating_x'] >= 0)]

            if user_rated.empty:
                st.warning("Không tìm thấy lịch sử đánh giá tích cực của người dùng.")
            else:
                # Kết hợp dữ liệu user_rated với products để lấy thông tin product_name
                user_rated_data = final_data[(final_data['user_id'] == user_id) & (final_data['rating_x'] >= 0)]

                # Hiển thị lịch sử đánh giá của người dùng mà không cần cột timestamp
                st.subheader("📊 Lịch sử đánh giá của bạn:")
                st.dataframe(user_rated_data[['user_id', 'product_name', 'rating_x']])

                # Lấy gợi ý sản phẩm từ mô hình SVD
                recommendations = get_svd_recommendations(user_id, n_recommendations=10)

                st.write("Gợi ý sản phẩm từ mô hình SVD:")

                html_content = f"""
                <style>
                    .scroll-wrapper {{
                        position: relative;
                        margin-bottom: 20px;
                    }}
                    .scroll-container {{
                        display: flex;
                        overflow-x: auto;
                        scroll-behavior: smooth;
                        padding: 10px 0;
                        white-space: nowrap;
                    }}
                    .scroll-container::-webkit-scrollbar {{
                        height: 8px;
                    }}
                    .scroll-container::-webkit-scrollbar-thumb {{
                        background-color: #888;
                        border-radius: 4px;
                    }}
                    .product-card {{
                        flex: 0 0 auto;
                        width: 250px;
                        background-color: #111;
                        color: white;
                        border: 1px solid #333;
                        border-radius: 10px;
                        padding: 15px;
                        margin-right: 15px;
                        box-shadow: 1px 1px 5px rgba(0,0,0,0.2);
                        overflow: hidden;
                        white-space: normal;
                        box-sizing: border-box;
                        height: 380px;
                    }}
                    .scroll-btn {{
                        position: absolute;
                        top: 35%;
                        background-color: rgba(0,0,0,0.6);
                        color: white;
                        border: none;
                        padding: 8px 12px;
                        font-size: 18px;
                        border-radius: 5px;
                        cursor: pointer;
                        z-index: 2;
                    }}
                    .scroll-btn:hover {{
                        background-color: rgba(255,255,255,0.2);
                    }}
                    .scroll-left {{
                        left: 0;
                    }}
                    .scroll-right {{
                        right: 0;
                    }}
                    .product-card img {{
                        width: 100%;
                        height: 180px;
                        object-fit: cover;
                        border-radius: 8px;
                    }}
                    .product-card h4 {{
                        font-size: 16px;
                        margin-top: 10px;
                    }}
                    .product-card p {{
                        font-size: 14px;
                        margin: 5px 0;
                    }}
                    .product-card .description {{
                        font-size: 12px;
                        color: #bbb;
                        line-height: 1.5;
                        max-height: 100px;
                        overflow-y: auto;
                        text-overflow: ellipsis;
                        white-space: normal;
                        width: 100%;
                        margin-top: 8px;
                    }}
                </style>

                <div class="scroll-wrapper">
                    <button class="scroll-btn scroll-left" onclick="document.getElementById('scroll_all').scrollBy({{left: -300, behavior: 'smooth'}})">←</button>
                    <button class="scroll-btn scroll-right" onclick="document.getElementById('scroll_all').scrollBy({{left: 300, behavior: 'smooth'}})">→</button>
                    <div id="scroll_all" class="scroll-container">
                """

                for _, row in recommendations.iterrows():
                    image_url = row.image if pd.notna(row.image) and row.image.startswith("http") else default_image_url

                    html_content += f"""
                    <div class="product-card">
                        <img src="{image_url}" />
                        <h4>🛍️ {row.product_name[:40]}</h4>
                        <p>⭐ {row.rating} | 💵 {int(row.price):,}</p>
                        <div class="description">{row.description[:150]}...</div>
                    </div>
                    """

                html_content += "</div></div>"
                components.html(html_content, height=400)
        else:
            st.error("Vui lòng nhập mã khách hàng hợp lệ (dạng số nguyên).")
