"""
replicates figure 4 in Atoniak et al. (2021)

basically extract 1st PC and compute cosine similarity between it and listed words

## still pretty ugly 
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import os
from sklearn.metrics.pairwise import cosine_similarity

import metrics
import seedbank
import replicate_bolukbasi
from utils import catch_keyerror


def figure_4(variance_ordered, variance_rnd, variance_inshuffle, sim_list):

    pc_ordered = np.mean(variance_ordered, axis=1)[0]

    pc_rnd = np.mean(variance_rnd, axis=1)[0]
    pc_inshuffle = np.mean(variance_inshuffle, axis=1)[0]

    similarity = []

    for idx, (pc, word_list) in enumerate(
        zip([pc_ordered, pc_rnd, pc_inshuffle], sim_list)
    ):
        temp_list = []
        # print(pc.shape)
        for word in word_list:
            if word is not None:
                temp_list.append(cosine_similarity([pc, word])[0, 1])
        similarity.append(temp_list)

    return similarity


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

    direct = os.fsencode("../data/models/nytimes_news_articles_min100/")

    for filename in os.listdir(direct):
        f = os.path.join(direct, filename)

        # checking if it is a file
        if os.path.isfile(f):
            f = os.fsdecode(f)
            models.append(KeyedVectors.load(f))

    # get desired seeds:

    seed = seedbank.seedbanking("../data/seeds/seeds.json")
    seed.set_index("Seeds ID", inplace=True)

    gender_seed_list = [
        "definitional_female-Bolukbasi_et_al_2016",
        "definitional_male-Bolukbasi_et_al_2016",
    ]

    # lower case seeds? she didnt do it in appendix (doesnt make sense tho)
    seed_list = seedbank.get_seeds(seed, gender_seed_list)
    seed1 = [item.lower() for item in seed_list[0]]
    seed2 = [item.lower() for item in seed_list[1]]

    # hard coded shuffled seeds from paper
    seed1_shuf = [
        "herself",
        "woman",
        "daughter",
        "mary",
        "her",
        "girl",
        "mother",
        "she",
        "female",
        "gal",
    ]
    seed2_shuf = [
        "man",
        "his",
        "he",
        "son",
        "guy",
        "himself",
        "father",
        "boy",
        "male",
        "john",
    ]

    (
        variance_ordered,
        variance_rnd,
        variance_inshuffle,
    ) = replicate_bolukbasi.pca_seeds_model(
        seed1, seed2, models, seed1_shuf, seed2_shuf, components=True
    )

    # average embeddings?? sounds sketch tbh -> didnt do it for now
    list_a = [
        "herself",
        "ms",
        "her",
        "she",
        "pregnant",
        "pitching",
        "baseball",
        "syndergraad",
        "himself",
        "his",
    ]

    list_b = [
        "likelihood",
        "eurozone",
        "incentive",
        "downturn",
        "setback",
        "photographed",
        "tales" "hood",
        "danced",
        "gracia",
    ]

    list_c = [
        "outcomes",
        "son",
        "father",
        "mother",
        "aunt",
        "potentially",
        "male",
        "hood",
        "garcia",
        "md",
    ]

    embed_b = [catch_keyerror(models, word) for word in list_b]
    embed_a = [catch_keyerror(models, word) for word in list_a]
    embed_c = [catch_keyerror(models, word) for word in list_c]

    sim = figure_4(
        variance_ordered, variance_rnd, variance_inshuffle, [embed_a, embed_b, embed_c]
    )

    print(list_a, "\n", sim[0], "\n")
    print(list_b, "\n", sim[1], "\n")
    print(list_c, "\n", sim[2], "\n")
