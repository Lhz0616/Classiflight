import spacy
import string
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors


# Word2Vec variables
import os
last = os.getcwd().split('\\')[-1]
if last == 'streamlit':
    path_offset = '../'

else:
    path_offset = './'


wv = KeyedVectors.load(f'{path_offset}/models/vectors.bin')
nlp = spacy.load('en_core_web_sm')
stop_words = nlp.Defaults.stop_words
punctuations = string.punctuation


def vectorise(user_input):
    vector_size = wv.vector_size
    wv_res = np.zeros(vector_size)

    ctr = 1
    for w in user_input:
        if w in wv:
            ctr += 1
            wv_res += wv[w]
    wv_res = wv_res/ctr
    return wv_res


def tokenise_sentence(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    doc = nlp(sentence)
    
    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() for word in doc ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens


def generate_fasttext_file(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        for i in range(len(data)):
            f.write("__label__" + str(data.iloc[i]["category"])[5:] + " " + data.iloc[i]["question"] + "\n")


def load_data(data_path):
    df = pd.read_csv(data_path)
    df.columns = ['category', 'question']
    df = df[~df.duplicated(subset='question', keep=False)]
    return df