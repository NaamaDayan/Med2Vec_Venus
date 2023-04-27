import csv
import numpy as np
from mittens import GloVe, Mittens
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import _stop_words
from collections import Counter
import pickle
import os
import nltk
nltk.download('brown')
from nltk.corpus import brown


# set download directory
# download_dir = os.path.join(os.getcwd(), 'brown_data')
#
# # make sure the download directory exists
# if not os.path.isdir(download_dir):
#     os.makedirs(download_dir)
#
# # add the download directory to the NLTK data path
# nltk.data.path.append(download_dir)
#
# # download the Brown corpus to the specified directory
# nltk.download('brown', download_dir=download_dir)


def glove2dict(glove_filename):
    with open(glove_filename, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:])))
                for line in reader}
    return embed

def get_rareoov(xdict, val):
    return [k for (k,v) in Counter(xdict).items() if v<=val]

def main():
    glove_path = "glove.6B.50d.txt"
    pre_glove = glove2dict(glove_path)

    sw = list(_stop_words.ENGLISH_STOP_WORDS)
    print("sw is:", sw)
    brown_data = brown.words()[:200000]
    brown_nonstop = [token.lower() for token in brown_data if (token.lower() not in sw)]
    oov = [token for token in brown_nonstop if token not in pre_glove.keys()]

    #oov_rare = get_rareoov(oov, 1)
    #corp_vocab = list(set(oov) - set(oov_rare))
    #brown_tokens = [token for token in brown_nonstop if token not in oov_rare]
    #brown_doc = [' '.join(brown_tokens)]

    corp_vocab = list(set(oov))
    brown_doc = [' '.join(brown_nonstop)]
    print("brown_doc is:", brown_doc[:1])

    cv = CountVectorizer(ngram_range=(1,1), vocabulary=corp_vocab)
    X = cv.fit_transform(brown_doc)
    Xc = (X.T * X)
    Xc.setdiag(0)
    coocc_ar = Xc.toarray()

    mittens_model = Mittens(n=50, max_iter=1000)

    new_embeddings = mittens_model.fit(
        coocc_ar,
        vocab=corp_vocab,
        initial_embedding_dict= pre_glove)

    newglove = dict(zip(corp_vocab, new_embeddings))
    with open("repo_glove.pkl","wb") as f:
        pickle.dump(newglove, f)

if __name__ == '__main__':
    main()