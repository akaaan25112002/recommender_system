# data_loader.py
import pandas as pd
from gensim import corpora, models, similarities
import streamlit as st

@st.cache_resource
def load_processed_data():
    data = pd.read_pickle("models/processed_data.pkl")
    dictionary = corpora.Dictionary.load("models/tfidf_dictionary.dict")
    tfidf = models.TfidfModel.load("models/tfidf_model.tfidf")
    index = similarities.Similarity.load("models/tfidf_index.index")
    return data, dictionary, tfidf, index

@st.cache_data
def load_ratings():
    ratings = pd.read_csv("Data/cleaned_ratings.csv")
    return ratings
