""" 
code to replicate parts of Bolukbasi et. al 2016 
code source: https://github.com/tolga-b/debiaswe
data source: https://drive.google.com/drive/folders/0B5vZVlu2WoS5dkRFY19YUXVIU2M?resourcekey=0-rZ1HR4Fb0XCi4HFUERGhRA
"""

from matplotlib import pyplot as plt
import random
import numpy as np
from gensim.models import KeyedVectors

import metrics


def read_wordembedding(fname):
    """
    reads wordembeddings in .txt format and turns it into a dictionary

    Parametrs
    -----------
    fname : string
        name of .txt file with embeddings

    Returns
    --------
    embed_dict: dictionary
        maps strings (words) to numpy arrays (embedding)
    """

    print("*** Reading data from " + fname)
    embed_dict = {}
    vecs = []

    with open(fname, "r", encoding="utf8") as f:
        for line in f:
            s = line.split()
            v = np.array([float(x) for x in s[1:]])
            if len(vecs) and vecs[-1].shape != v.shape:
                print("Got weird line", line)
                continue
            v = np.array(v, dtype="float32")
            embed_dict.update({s[0]: v})

    return embed_dict


if __name__ == "__main__":

    # load google news word2vec
    # Load vectors directly from the file
    model = KeyedVectors.load_word2vec_format(
        "../data/GoogleNews-vectors-negative300.bin", binary=True
    )

    # definde seeds
    seed_female = [
        "she",
        "her",
        "woman",
        "Mary",
        "herself",
        "daughter",
        "mother",
        "gal",
        "girl",
        "female",
    ]
    seed_male = [
        "he",
        "him",
        "man",
        "John",
        "himself",
        "son",
        "father",
        "guy",
        "boy",
        "male",
    ]

    # shufffled seeds
    shuffled_list = seed_female + seed_male
    random.shuffle((shuffled_list))
    seed1_shuffled = shuffled_list[:10]
    seed2_shuffled = shuffled_list[10:]

    # do pca
    pca_ordered = metrics.do_pca(seed_female, seed_male, model)
    pca_shuffled = metrics.do_pca(seed1_shuffled, seed2_shuffled, model)
    variance_ordered = pca_ordered.explained_variance_ratio_
    variance_shuffled = pca_shuffled.explained_variance_ratio_
    print(" \n Variance of ordered gender pairs: \n", variance_ordered)
    print(" \n Variance of shuffled gender pairs: \n", variance_shuffled)
    # plt.bar(range(10), pca_ordered.explained_variance_ratio_)
    # plt.show()
