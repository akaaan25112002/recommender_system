# build_gensim_model.py
import pandas as pd
import os
from gensim import corpora, models, similarities
import ast

data = pd.read_csv("Data/cleaned_products.csv")
data['tokens'] = data['tokens'].apply(ast.literal_eval)

dictionary = corpora.Dictionary(data['tokens'].tolist())
corpus = [dictionary.doc2bow(text) for text in data['tokens']]
tfidf = models.TfidfModel(corpus)
index = similarities.Similarity(output_prefix="models/tfidf_index", corpus=tfidf[corpus], num_features=len(dictionary))

# Lưu
dictionary.save("models/tfidf_dictionary.dict")
tfidf.save("models/tfidf_model.tfidf")
index.save("models/tfidf_index.index")
print("Đã lưu Gensim model")