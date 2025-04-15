# build_model.py
import os
import pandas as pd
from underthesea import word_tokenize
from gensim import corpora, models, similarities
from utils import clean_text_row, stop_words
import pickle

# Đảm bảo path tuyệt đối dựa trên vị trí file hiện tại
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
product_path = os.path.join(BASE_DIR, "Data", "cleaned_products.csv")
model_dir = os.path.join(BASE_DIR, "models")
os.makedirs(model_dir, exist_ok=True)

#index_file = os.path.join(model_dir, "tfidf_index.index")
index_path = os.path.join(model_dir, "sim_index.index")

# Nếu đã build thì bỏ qua
if os.path.exists(index_file):
    print("✅ Đã build rồi. Không cần chạy lại.")
    exit()

# Đọc dữ liệu
data = pd.read_csv(product_path)

# Token hóa
data["Content_wt"] = data["Content"].apply(lambda x: word_tokenize(str(x), format="text"))

# Làm sạch & stop word
content_gem = [[text for text in x.split()] for x in data.Content_wt]
content_gem_re = pd.Series(content_gem).apply(lambda x: clean_text_row(x, stop_words))

# Tạo Dictionary & Corpus
dictionary = corpora.Dictionary(content_gem_re)
corpus = [dictionary.doc2bow(text) for text in content_gem_re]

# TF-IDF & Similarity
tfidf = models.TfidfModel(corpus)
index = similarities.Similarity(output_prefix=os.path.join(model_dir, "sim_index"),
                                corpus=tfidf[corpus],
                                num_features=len(dictionary))

# Lưu dữ liệu
data["tokens"] = content_gem_re
data.to_pickle(os.path.join(model_dir, "processed_data.pkl"))
dictionary.save(os.path.join(model_dir, "tfidf_dictionary.dict"))
tfidf.save(os.path.join(model_dir, "tfidf_model.tfidf"))

print("✅ Đã build model TF-IDF và lưu thành công.")
