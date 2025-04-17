import os
import pandas as pd
from utils import filter_vietnamese_words
from underthesea import word_tokenize
import gdown

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
os.makedirs(DATA_DIR, exist_ok=True)

# File paths
CLEANED_PRODUCTS_FILE = os.path.join(DATA_DIR, 'cleaned_products.csv')
CLEANED_RATINGS_FILE = os.path.join(DATA_DIR, 'cleaned_ratings.csv')
RAW_PRODUCTS_FILE = os.path.join(DATA_DIR, 'Products_ThoiTrangNam_raw.csv')
RAW_RATINGS_FILE = os.path.join(DATA_DIR, 'Products_ThoiTrangNam_rating_raw.csv')

# Google Drive File IDs (tùy bạn thay vào)
CLEANED_PRODUCTS_ID = "1ABCxyz_cleaned_products_ID"
CLEANED_RATINGS_ID = "1DEFxyz_cleaned_ratings_ID"
RAW_PRODUCTS_ID = "1kMQ6Fk__epxgcBQADdGSf2OdUM5YZmgR"
RAW_RATINGS_ID = "10mS7UAzMf-VtHlvgiuSQYJ5L22LAzpdH"

def download_file(file_id, output_path, desc):
    print(f"🔽 Đang tải {desc} từ Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

def load_and_prepare_data():
    # Ưu tiên dùng file đã cleaned nếu có
    if os.path.exists(CLEANED_PRODUCTS_FILE) and os.path.exists(CLEANED_RATINGS_FILE):
        print("✅ Đang load dữ liệu đã được cleaned...")
        products = pd.read_csv(CLEANED_PRODUCTS_FILE)
        ratings = pd.read_csv(CLEANED_RATINGS_FILE)
    else:
        # Nếu chưa có thì xử lý lại từ file raw
        if not os.path.exists(RAW_PRODUCTS_FILE):
            download_file(RAW_PRODUCTS_ID, RAW_PRODUCTS_FILE, "Products raw")
        if not os.path.exists(RAW_RATINGS_FILE):
            download_file(RAW_RATINGS_ID, RAW_RATINGS_FILE, "Ratings raw")

        products = pd.read_csv(RAW_PRODUCTS_FILE)
        ratings = pd.read_csv(RAW_RATINGS_FILE, sep="\t")

        # Xử lý dữ liệu như trước
        products['image'] = products['image'].fillna('No image available')
        products['description'] = products['description'].fillna('No description available')

        ratings = ratings.drop_duplicates()
        ratings_avg = ratings.groupby(['user_id', 'product_id'])['rating'].mean().reset_index()
        ratings_avg['rating'] = ratings_avg['rating'].round(0)
        ratings = ratings_avg[ratings_avg['product_id'].isin(products['product_id'])]

        data = products[['product_id', 'product_name', 'sub_category', 'price', 'rating', 'image', 'description']].copy()
        data['description'] = data['description'].str.replace('Danh Mục\nShopee\nThời Trang Nam\n', '', regex=False)
        data['description'] = data['description'].str.replace('\n', ' ')
        data['description_clean'] = data['description'].apply(filter_vietnamese_words)
        data['Content'] = data['product_name'] + ' ' + data['description_clean'].apply(lambda x: ' '.join(x.split()[:200]))
        data['tokens'] = data['description_clean'].apply(lambda x: word_tokenize(str(x), format="text").split())

        # Save cleaned
        data.to_csv(CLEANED_PRODUCTS_FILE, index=False)
        ratings.to_csv(CLEANED_RATINGS_FILE, index=False)

        products = data  # dùng cleaned version để đồng nhất

    final_data = pd.merge(ratings, products, how='inner', on='product_id')
    print("✅ Dữ liệu đã sẵn sàng!")

    return products, ratings, final_data

if __name__ == "__main__":
    products, ratings, final_data = load_and_prepare_data()