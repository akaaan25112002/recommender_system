import os
import pandas as pd
from utils import filter_vietnamese_words
from underthesea import word_tokenize
import gdown

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
os.makedirs(DATA_DIR, exist_ok=True)

# File paths
PRODUCTS_FILE = os.path.join(DATA_DIR, 'Products_ThoiTrangNam_raw.csv')
RATINGS_FILE = os.path.join(DATA_DIR, 'Products_ThoiTrangNam_rating_raw.csv')

# Google Drive File IDs mới
PRODUCTS_ID = "1kMQ6Fk__epxgcBQADdGSf2OdUM5YZmgR"
RATINGS_ID = "10mS7UAzMf-VtHlvgiuSQYJ5L22LAzpdH"

# Tải file nếu chưa có
if not os.path.exists(PRODUCTS_FILE):
    print("🔽 Đang tải Products file từ Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={PRODUCTS_ID}", PRODUCTS_FILE, quiet=False)

if not os.path.exists(RATINGS_FILE):
    print("🔽 Đang tải Ratings file từ Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={RATINGS_ID}", RATINGS_FILE, quiet=False)

# Load dữ liệu
products = pd.read_csv(PRODUCTS_FILE)
ratings = pd.read_csv(RATINGS_FILE, sep='\t')

# Xử lý NaN
products['image'] = products['image'].fillna('No image available')
products['description'] = products['description'].fillna('No description available')

# Xử lý rating trùng, tính trung bình
ratings = ratings.drop_duplicates()
ratings_avg = ratings.groupby(['user_id', 'product_id'])['rating'].mean().reset_index()
ratings_avg['rating'] = ratings_avg['rating'].round(0)
ratings = ratings_avg[ratings_avg['product_id'].isin(products['product_id'])]

# Tiền xử lý văn bản sản phẩm
data = products[['product_id', 'product_name', 'sub_category', 'price', 'rating', 'image', 'description']].copy()
data['description'] = data['description'].str.replace('Danh Mục\nShopee\nThời Trang Nam\n', '', regex=False)
data['description'] = data['description'].str.replace('\n', ' ')
data['description_clean'] = data['description'].apply(filter_vietnamese_words)
data['Content'] = data['product_name'] + ' ' + data['description_clean'].apply(lambda x: ' '.join(x.split()[:200]))
data['tokens'] = data['description_clean'].apply(lambda x: word_tokenize(str(x), format="text").split())
final_data = pd.merge(ratings, products, how='inner', on='product_id')

# Lưu file đã xử lý
data.to_csv(os.path.join(DATA_DIR, "cleaned_products.csv"), index=False)
ratings.to_csv(os.path.join(DATA_DIR, "cleaned_ratings.csv"), index=False)
print("✅ Xử lý dữ liệu thành công")

