""" 
code to replicate parts of Bolukbasi et. al 2016 
code source: https://github.com/tolga-b/debiaswe
data source: https://drive.google.com/drive/folders/0B5vZVlu2WoS5dkRFY19YUXVIU2M?resourcekey=0-rZ1HR4Fb0XCi4HFUERGhRA
"""

from matplotlib import pyplot as plt
import random
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import copy
import os
from ast import literal_eval

import metrics
import seedbank


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


# def replicate_figure(seed1, seed2, dataset1, dataset2):


if __name__ == "__main__":

    models = []

    # load google news word2vec
    # Load vectors directly from the file
    # models.append(
    #     KeyedVectors.load_word2vec_format(
    #         "../data/GoogleNews-vectors-negative300.bin", binary=True
    #     )
    # )

    # replicate fig. 3 with NYT dataset

    direct = os.fsencode("../models/nytimes_news_articles_preprocessed/")

    for filename in os.listdir(direct):
        print(filename)
        f = os.path.join(direct, filename)

        # checking if it is a file
        if os.path.isfile(f):
            f = os.fsdecode(f)
            models.append(KeyedVectors.load(f))

    # get desired seeds:

    gender_seed_list = [
        "definitional_female-Bolukbasi_et_al_2016",
        "definitional_male-Bolukbasi_et_al_2016",
    ]
    class_seeds_list = [
        "upperclass-Kozlowski_et_al_2019",
        "lowerclass-Kozlowski_et_al_2019",
    ]
    names_seeds_lists = [
        "names_chinese-Garg_et_al_2018",
        "names_hispanic-Garg_et_al_2018",
    ]

    seed = seedbank.seedbanking("../data/seeds/seeds.json")
    seed.set_index("Seeds ID", inplace=True)

    # lower case seeds? she didnt do it in appendix (doesnt make sense tho)
    seed_list = seedbank.get_seeds(seed, gender_seed_list)
    seed1 = [item.lower() for item in seed_list[0]]
    seed2 = [item.lower() for item in seed_list[1]]
    # print(seed1)

    # draw random words from word2vec
    seed1_rnd = [random.randint(1, 4000) for i in range(10)]
    seed2_rnd = [random.randint(1, 4000) for i in range(10)]

    # # shufffled seeds (not needed as of my interpretation)
    # shuffled_list = seed_female + seed_male
    # random.shuffle((shuffled_list))
    # seed1_shuffled = shuffled_list[:10]
    # seed2_shuffled = shuffled_list[10:]

    # shuffled in place to test for cherry picking
    # seedf_inshuffle = copy.deepcopy(seed1)
    # (random.shuffle((seedf_inshuffle)))
    # seedm_inshuffle = copy.deepcopy(seed2)
    # (random.shuffle((seedm_inshuffle)))

    seedf_inshuffle = [
        "herself",
        "woman",
        "daughter",
        "Mary",
        "her",
        "girl",
        "mother",
        "she",
        "female",
        "gal",
    ]
    seedm_inshuffle = [
        "man",
        "his",
        "he",
        "son",
        "guy",
        "himself",
        "father",
        "boy",
        "male",
        "John",
    ]

    # seedf_inshuffle = ['richer', 'opulent', 'luxury', 'affluent', 'rich', 'affluence', 'richest', 'expensive']
    # seedm_inshuffle = ['poorer', 'impoverished', 'poorest', 'cheap', 'needy', 'poverty', 'inexpensive', 'poor']

    # seedf_inshuffle = ['tang', 'chang', 'chu', 'yang', 'wu', 'hong', 'huang', 'wong', 'hu', 'liu', 'lin', 'chen', 'liang', 'chung', 'li', 'ng', 'wang']
    # seedm_inshuffle = ['ruiz', 'rodriguez', 'diaz', 'perez', 'lopez', 'vargas', 'alvarez', 'garcia', 'cruz', 'torres', 'gonzalez', 'soto', 'martinez', 'medina', 'rivera', 'castillo', 'castro', 'mendoza', 'sanchez', 'gomez']

    variance_ordered = np.zeros((10, len(models)))
    variance_rnd = np.zeros((10, len(models)))
    variance_inshuffle = np.zeros((10, len(models)))

    for idx, model in enumerate(models):
        pca_ordered = metrics.do_pca(seed1, seed2, model)
        pca_rnd = metrics.do_pca(seed1_rnd, seed2_rnd, model)
        pca_inshuffle = metrics.do_pca(seedf_inshuffle, seedm_inshuffle, model)
        variance_ordered[:, idx] = pca_ordered.explained_variance_ratio_
        variance_rnd[:, idx] = pca_rnd.explained_variance_ratio_
        variance_inshuffle[:, idx] = pca_inshuffle.explained_variance_ratio_

    # Viz

    print(" \n Variance of ordered pairs: \n", variance_ordered)
    print(" \n Variance of random word pairs: \n", variance_rnd)
    print(" \n Variance of inplace shuffled pairs: \n", variance_inshuffle)
    plt.bar(
        range(10),
        np.mean(variance_ordered, axis=1),
        yerr=np.std(variance_ordered, axis=1),
    )
    plt.title("Variance of ordered pairs")
    plt.show()
    plt.bar(
        range(10), np.mean(variance_rnd, axis=1), yerr=np.std(variance_ordered, axis=1)
    )
    plt.title("Variance of random word pairs")
    plt.show()
    plt.bar(
        range(10),
        np.mean(variance_inshuffle, axis=1),
        yerr=np.std(variance_ordered, axis=1),
    )
    plt.title("Variance of inplace shuffled pairs")
    plt.show()

    # thoughts:
    # can kind of replicate with given seeds b) but seems super pruned to suffle seed (in reality they seem to be quite close together)
    # can not really replicate, but a) 'gal' is just in no dataset - explained variance of 1PC huge
    # doesnt make any sense for her to not lower case seeds, when corpus is lower cased wtf
