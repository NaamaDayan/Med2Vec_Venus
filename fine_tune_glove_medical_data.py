
import ast
import subprocess
import sys

# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])
#
subprocess.call(["pip", "install"," --upgrade pip"])

# install('--up')
import ast
import csv
import numpy as np
from mittens import GloVe, Mittens
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import _stop_words
from collections import Counter
import pickle
import os
import nltk
nltk.download('punkt')
import pandas as pd

def glove2dict(glove_filename):
    with open(glove_filename, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:])))
                for line in reader}
    return embed


def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def tokenize_words():

    # use the generated sentences
    visit_data = load_data('./data/decoded_large_data')

    corpus = [code for visit in visit_data for code in visit]
    corpus = [code if code != ['-1'] else "<EOS>" for code in corpus]
    vocab_df = pd.read_csv('encoded_vocab_with_hierarchy_large_data.csv')
    # Create a set of all diagnoses
    vocab_diagnoses = set(diagnosis for diagnosis in vocab_df['Diagnosis'])
    corpus = [d for d in corpus if d == '<EOS>' or d in vocab_diagnoses]

    with open("repo_corpus_large_data.pkl","wb") as f:
        pickle.dump(corpus, f)


def main():
    glove_path = "glove.6B.50d.txt"
    pre_glove = glove2dict(glove_path)
    tokenize_words()

    corpus = load_data("repo_corpus_large_data.pkl")
    vocab = set(corpus)
    print(corpus)
    print(vocab)

    cv = CountVectorizer(ngram_range=(1,1), vocabulary=vocab)
    X = cv.fit_transform(corpus)
    Xc = (X.T * X)
    Xc.setdiag(0)
    coocc_ar = Xc.toarray()

    mittens_model = Mittens(n=50, max_iter=45)
    new_embeddings = mittens_model.fit(
        coocc_ar,
        vocab=vocab,
        initial_embedding_dict= pre_glove)

    newglove = dict(zip(vocab, new_embeddings))
    with open("repo_glove_visits_large_data.pkl","wb") as f:
        pickle.dump(newglove, f)

    # with open("repo_tokenized.pkl","wb") as f:
    #     pickle.dump(corpus_tokens, f)
if __name__ == '__main__':
    main()