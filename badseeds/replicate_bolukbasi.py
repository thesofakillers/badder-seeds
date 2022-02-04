""" 
code to replicate parts of Bolukbasi et. al 2016 
code source: https://github.com/tolga-b/debiaswe
data source: https://drive.google.com/drive/folders/0B5vZVlu2WoS5dkRFY19YUXVIU2M?resourcekey=0-rZ1HR4Fb0XCi4HFUERGhRA
"""

import json
import argparse
import copy
import os
import collections


from matplotlib import pyplot as plt
import random
import numpy as np
from gensim.models import KeyedVectors

import badseeds.seedbank as seedbank
import badseeds.metrics as metrics

random.seed(42)


def pca_seeds_model(
    seed1,
    seed2,
    models,
    seed1_shuf=False,
    seed2_shuf=False,
    seed1_rnd=False,
    seed2_rnd=False,
    components=False,
):
    """
    replicates figure 3

    Parametrs
    -----------
    seed1 : list of floats
        embeddings of seeds
    seed2 : list of floats
        embeddings of seeds
    seed1_shuf : list of floats
        embeddings of shuffled seeds
    seed2_shuf : list of floats
        embeddings of shuffled seeds
    models: list of KeyedVector objects (Gensim)
        trained skipgram model


    Returns
    --------
    variance_ordered: numpy array of arrays
        pca on seed on different models
    variance_rnd: numpy array of arrays
        pca on seed on different models
    variance_inshuffle: numpy array of arrays
        pca on seed on different models
    """

    # draw random words from word2vec
    if seed1_rnd == False and seed2_rnd == False:
        seed1_rnd = [random.randint(1, 4000) for i in range(10)]
        seed2_rnd = [random.randint(1, 4000) for i in range(10)]
        # ensure that random word is picked that is present across all models
        collect = collections.Counter(models[0].index_to_key)
        for model in models[1:]:
            s += len(model.index_to_key)
            collect = (collect & collections.Counter(model.index_to_key))
        overlap_list = list((collect).elements())
        seed1_rnd = np.asarray(overlap_list)[seed1_rnd]
        seed2_rnd = np.asarray(overlap_list)[seed2_rnd]
        print('random words:', seed1_rnd, seed2_rnd)

    # shuffled in place to test for cherry picking
    if seed1_shuf == False and seed2_shuf == False:
        seed1_shuf = copy.deepcopy(seed1)
        (random.shuffle((seed1_shuf)))
        seed2_shuf = copy.deepcopy(seed2)
        (random.shuffle((seed2_shuf)))

    variance_ordered = []
    variance_rnd = []
    variance_inshuffle = []

    for idx, model in enumerate(models):
        pca_ordered = metrics.do_pca(seed1, seed2, model)
        if len(seed1_rnd) > 0 and len(seed2_rnd) > 0:
            pca_rnd = metrics.do_pca(seed1_rnd, seed2_rnd, model)
        pca_inshuffle = metrics.do_pca(seed1_shuf, seed2_shuf, model)
        if components:
            variance_ordered.append(pca_ordered.components_)
            variance_rnd.append(pca_rnd.components_)
            variance_inshuffle.append(pca_inshuffle.components_)

        else:
            variance_ordered.append(pca_ordered.explained_variance_ratio_)
            variance_rnd.append(pca_rnd.explained_variance_ratio_)
            variance_inshuffle.append(pca_inshuffle.explained_variance_ratio_)

    return (
        np.asarray(variance_ordered),
        np.asarray(variance_rnd),
        np.asarray(variance_inshuffle),
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Replicates figure 4 in Atoniak et al. (2021)"
    )
    parser.add_argument(
        "-c",
        "--config",
        help="path to config JSON file containing path to seeds",
        default="config.json",
        type=str,
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)

    models = []

    # load google news word2vec
    # Load vectors directly from the file
    models.append(
        KeyedVectors.load_word2vec_format(
            os.path.join(
                config["models"]["dir_path"], config["models"]["google_news_subpath"]
            )
            + ".bin",
            binary=True,
        )
    )

    # direct = os.fsencode(
    #     os.path.join(
    #         config["models"]["dir_path"], config["models"]["nyt_subpath"]["10"]
    #     )
    # )

    # for filename in os.listdir(direct):
    #     print(filename)
    #     f = os.path.join(direct, filename)

    #     # checking if it is a file
    #     if os.path.isfile(f):
    #         f = os.fsdecode(f)
    #         models.append(KeyedVectors.load(f))

    # get desired seeds:

    seed = seedbank.seedbanking(config["seeds"]["dir_path"] + "seeds.json", index="ID")

    seed_genres = ["gender pairs", "social class pairs", "chinese-hispanic name pairs"]

    seed_list = [
        [
            "definitional_female-Bolukbasi_et_al_2016",
            "definitional_male-Bolukbasi_et_al_2016",
        ],
        [
            "upperclass-Kozlowski_et_al_2019",
            "lowerclass-Kozlowski_et_al_2019",
        ],
        [
            "names_chinese-Garg_et_al_2018",
            "names_hispanic-Garg_et_al_2018",
        ],
    ]

    # hard coded shuffled seeds from paper

    shuffled_seeds = [
        [
            [
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
            ],
            [
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
            ],
        ],
        [
            [
                "richer",
                "opulent",
                "luxury",
                "affluent",
                "rich",
                "affluence",
                "richest",
                "expensive",
            ],
            [
                "poorer",
                "impoverished",
                "poorest",
                "cheap",
                "needy",
                "poverty",
                "inexpensive",
                "poor",
            ],
        ],
        [
            [
                "tang",
                "chang",
                "chu",
                "yang",
                "wu",
                "hong",
                "huang",
                "wong",
                "hu",
                "liu",
                "lin",
                "chen",
                "liang",
                "chung",
                "li",
                "ng",
                "wang",
            ],
            [
                "ruiz",
                "rodriguez",
                "diaz",
                "perez",
                "lopez",
                "vargas",
                "alvarez",
                "garcia",
                "cruz",
                "torres",
                "gonzalez",
                "soto",
                "martinez",
                "medina",
                "rivera",
                "castillo",
                "castro",
                "mendoza",
                "sanchez",
                "gomez",
            ],
        ],
    ]

    # Visualization

    x = np.arange(10)
    width = 0.4
    fig, axes = plt.subplots(1, 3)

    # for row in axes
    for idx, ax in enumerate(axes):

        # lower case seeds? she didnt do it in appendix (doesnt make sense tho)
        seed_lists = [seed.loc[seed_set]["Seeds"] for seed_set in seed_list[idx]]
        seed1 = [item.lower() for item in seed_lists[0]]
        seed2 = [item.lower() for item in seed_lists[1]]

        seed1_shuf = (shuffled_seeds[idx])[0]
        seed2_shuf = (shuffled_seeds[idx])[1]

        variance_ordered, variance_rnd, variance_inshuffle = pca_seeds_model(
            seed1, seed2, models, seed1_shuf, seed2_shuf
        )

        ax.bar(
            x - 0.2,
            np.mean(variance_ordered, axis=0),
            width,
            yerr=np.std(variance_ordered, axis=0),
            label="original order",
        )

        ax.bar(
            x + 0.2,
            np.mean(variance_inshuffle, axis=0),
            width,
            yerr=np.std(variance_inshuffle, axis=0),
            label="shuffled",
        )
        ax.legend()
        ax.set_xlabel("Prinicipal Component")
        ax.set_ylabel("Explained Variance")
        ax.set_title(seed_genres[idx])
    plt.show()

    # thoughts:
    # can kind of replicate with given seeds b) but seems super cherrypicked to suffle seed (in reality they seem to be quite close together)
    # can not really replicate, but a) 'gal' is just in no dataset - explained variance of 1PC huge
    # doesnt make any sense for her to not lower case seeds, when corpus is lower cased wtf
