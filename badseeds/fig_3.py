"""
replicates figure 2 from Antoniak et. al (2021)
"""
import os
import argparse
import json

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from gensim.models import KeyedVectors
import pandas as pd

import seedbank
import utils


def figure_3(seeds, datasets):
    """
    replicates figure 2 in Antoniak et al. (2016)

    Parameters
    -----------
    seeds : list of list
        list of list of seeds
    datasets : list of KeyedVector
        list of models containing word embeddings

    Returns
    --------
    similarity : list of array-like
        list of arrays with cosine similarity for each seedset and each dataset (model)
    """

    similarity = []
    for data in datasets:
        embeds = [[] for i in range(len(seeds))]
        avg_unpleasent = utils.get_average_embeddings(
            "unpleasantness", data, allow_missing=True
        )
        for i, seed in enumerate(seeds):
            s = utils.get_average_embeddings(seed, data, allow_missing=True)
            embeds[i].append(s)

        avg_unpleasent = np.mean(avg_unpleasent, axis=0)

        s = []
        for per_seed in embeds:
            temp = []
            for idx, seed in enumerate(per_seed):
                if seed.ndim < 2:
                    seed = seed.reshape(1, -1)
                seed = np.asarray([s for s in seed if np.isfinite(s).any()])
                temp.append(cosine_similarity(seed, [avg_unpleasent]))
            s.append(np.asarray(temp).flatten())
        similarity.append(s)

    return similarity


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Replicates figure 2 in Atoniak et al. (2021)"
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

    seeds = seedbank.seedbanking(config["seeds"]["dir_path"] + "seeds.json", index=True)

    seed_sets = [
        "female-Kozlowski_et_al_2019",
        "female_1-Caliskan_et_al_2017",
        "definitional_female-Bolukbasi_et_al_2016",
        "female_singular-Hoyle_et_al_2019",
        "female_definition_words_2-Zhao_et_al_2018",
        "female_stereotype_words-Zhao_et_al_2018",
    ]

    extracted_seeds = [seeds.loc[seed_set]["Seeds"] for seed_set in seed_sets]

    datasets = []

    filenames = [
        "goodreads_r_subpath",
        "goodreads_hb_subpath",
    ]

    for f in filenames:
        models = []
        direct = os.fsencode(
            os.path.join(config["models"]["dir_path"], config["models"][f]["0"])
        )

        for filename in os.listdir(direct):
            f = os.path.join(direct, filename)

            # checking if it is a file
            if os.path.isfile(f):
                f = os.fsdecode(f)
                if ".npy" not in f:
                    models.append(KeyedVectors.load(f))

        datasets.append(models)

    similarity = figure_3(extracted_seeds, datasets)

    for sim in similarity:
        for i, j in zip(extracted_seeds, sim):
            print(i, "\n")
            print(j, "\n \n")

    # viz
    df1 = pd.DataFrame(
        zip(similarity[0], seed_sets, ["history and biography"] * len(seeds)),
        columns=["cosine similarity", "seed set", "dataset"],
    )
    df2 = pd.DataFrame(
        zip(similarity[1], seed_sets, ["romance"] * len(seeds)),
        columns=["cosine similarity", "seed set", "dataset"],
    )

    df = pd.concat([df1, df2])
    df = df.explode("cosine similarity")
    df["cosine similarity"] = df["cosine similarity"].astype("float")

    # Creating plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots()
    ax1 = sns.boxplot(
        x="cosine similarity", y="seed set", hue="dataset", data=df, palette="Accent"
    )
    ax2 = sns.stripplot(
        x="cosine similarity",
        y="seed set",
        hue="dataset",
        data=df,
        jitter=True,
        palette="Accent",
        dodge=True,
        linewidth=1,
        edgecolor="gray",
    )

    legend = ax1.get_legend()
    handles = legend.legendHandles
    ax.legend(handles, ["history and biography", "romance"])
    plt.xlabel("cosine similairty to unpleasentness")
    plt.show()
