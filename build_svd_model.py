import pandas as pd
from surprise import Dataset, Reader, SVD
import joblib

# Giả sử bạn đã load dữ liệu ratings vào final_data
final_data = pd.read_csv("Data/final_data.csv")  # hoặc load cách bạn muốn

final_data_clean = final_data.dropna(subset=['user_id', 'product_id', 'rating_x'])
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(final_data_clean[['user_id', 'product_id', 'rating_x']], reader)

trainset = dataset.build_full_trainset()
algo = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
algo.fit(trainset)

# Lưu mô hình
joblib.dump(algo, "models/svd_model.pkl")
print("✅ Mô hình SVD đã được lưu thành công!")