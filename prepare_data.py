# prepare_data.py
import pandas as pd
import os
from utils import filter_vietnamese_words
from underthesea import word_tokenize

# Lấy đường dẫn tuyệt đối tới thư mục hiện tại
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn tuyệt đối tới các file dữ liệu
PRODUCTS_FILE = os.path.join(BASE_DIR, 'Data', 'Products_ThoiTrangNam_raw.csv')
RATINGS_FILE = os.path.join(BASE_DIR, 'Data', 'Products_ThoiTrangNam_rating_raw.csv')

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
final_data = pd.merge(ratings, products, how='inner', on='product_id')
data['tokens'] = data['description_clean'].apply(lambda x: word_tokenize(str(x), format="text").split())
# Lưu lại file đã làm sạch
data_dir = os.path.join(BASE_DIR, "Data")
os.makedirs(data_dir, exist_ok=True)

# Lưu lại file đã làm sạch
data.to_csv(os.path.join(data_dir, "cleaned_products.csv"), index=False)
ratings.to_csv(os.path.join(data_dir, "cleaned_ratings.csv"), index=False)
print("Xử lý thành công")
