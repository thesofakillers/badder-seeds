"""
replicates figure 4 in Atoniak et al. (2021)
"""

import argparse
import json
import os
import collections

from matplotlib import pyplot as plt
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

import badseeds.seedbank as seedbank
import badseeds.utils as utils
import badseeds.replicate_bolukbasi as replicate_bolukbasi


def figure_4(variance_ordered, variance_rnd, variance_inshuffle, models):
    """
    replicates figure 4 in Antoniak et al. (2016)

    Parameters
    -----------
    variance_ordered : array-like of float
        components of PCA on ordered seed set
    variance_rnd : array-like of float
        components of PCA on random seed set
    variance_inshuffle : array-like of float
        components of PCA on suffled seed set
    models : list of KeyedVectors
        list of all KeyedVectors obtained through bootstrapping

    Returns
    --------
    gender_pairs_values : list of floats
        cosine similarity of top & bottom 10 words in corpus
    gender_pairs_words : list of strings
        words with cosine similarity of top & bottom 10 words in corpus
    random_pairs_values : list of floats
        cosine similarity of top & bottom 10 words in corpus
    random_pairs_words : list of strings
        words with cosine similarity of top & bottom 10 words in corpus
    shuffled_gender_pairs_values : list of floats
        cosine similarity of top & bottom 10 words in corpus
    shuffled_gender_pairs_words : list of strings
        words with cosine similarity of top & bottom 10 words in corpus
    """

    collect = collections.Counter(models[0].index_to_key)
    s = 0
    for model in models[1:]:
        s += len(model.index_to_key)
        collect = collect & collections.Counter(model.index_to_key)

    overlap_list = list((collect).elements())
    overlap_embed = utils.get_embeddings(overlap_list, models, query_strat="average")
    vals = []
    words = []
    for var in [variance_ordered, variance_rnd, variance_inshuffle]:
        temp = np.mean(var, axis=0)[0]
        cos_sim = cosine_similarity(overlap_embed, [temp]).flatten()
        v, w = zip(*sorted(zip(list(cos_sim), overlap_list)))
        vals.append(v[:10] + v[-10:])
        words.append((w[:10] + w[-10:]))

    return (
        np.asarray(vals[0]),
        words[0],
        np.asarray(vals[1]),
        words[1],
        np.asarray(vals[2]),
        words[2],
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
    parser.add_argument(
        "-d",
        "--corpus",
        type=str,
        default="nyt",
        help="Use embeddings from skip-gram trained on this corpus",
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)

    models = []
    # load google news word2vec
    # Load vectors directly from the file
    if args.corpus == "googlenews":
        models.append(
            KeyedVectors.load_word2vec_format(
                os.path.join(
                    config["models"]["dir_path"],
                    config["models"]["google_news_subpath"],
                )
                + ".bin",
                binary=True,
            )
        )

    # replicate fig. 3 with NYT dataset
    elif args.corpus == "nyt":
        # get embeddings trained on NYT with min freq of 100
        direct = os.fsencode(
            os.path.join(
                config["models"]["dir_path"], config["models"]["nyt_subpath"]["100"]
            )
        )

        for filename in os.listdir(direct):
            f = os.path.join(direct, filename)

            # checking if it is a file
            if os.path.isfile(f):
                f = os.fsdecode(f)
                models.append(KeyedVectors.load(f))

    else:
        print("this corpus is not implemented")
        exit()

    # get desired seeds:
    seed = seedbank.seedbanking(config["seeds"]["dir_path"] + "seeds.json", index="ID")

    gender_seed_list = [
        "definitional_female-Bolukbasi_et_al_2016",
        "definitional_male-Bolukbasi_et_al_2016",
    ]

    seed_list = [seed.loc[seed_set]["Seeds"] for seed_set in gender_seed_list]
    seed1 = seed_list[0]
    seed2 = seed_list[1]

    # hard coded shuffled seeds from paper
    seed1_shuf = [
        "female",
        "she",
        "woman",
        "gal",
        "her",
        "daughter",
        "girl",
        "herself",
        "mother",
        "mary",
    ]
    # misses seed
    seed2_shuf = [
        "john",
        "man",
        "son",
        "father",
        "male",
        "himself",
        "guy",
        "he",
        "his",
        "boy",
    ]

    seed2_rnd = [
        "chun",
        "brush",
        "dictates",
        "caesar",
        "fewest",
        "breitbart",
        "rod",
        "heaped",
        "julianna",
        "longest",
    ]
    seed1_rnd = [
        "negatives",
        "vel",
        "theirs",
        "canoe",
        "meet",
        "bilingual",
        "mor",
        "facets",
        "fari",
        "lily",
    ]

    (
        variance_ordered,
        variance_rnd,
        variance_inshuffle,
    ) = replicate_bolukbasi.pca_seeds_model(
        seed1,
        seed2,
        models,
        seed1_shuf,
        seed2_shuf,
        components=True,
    )

    (
        gender_pairs_values,
        gender_pairs_words,
        random_pairs_values,
        random_pairs_words,
        shuffled_gender_pairs_values,
        shuffled_gender_pairs_words,
    ) = figure_4(variance_ordered, variance_rnd, variance_inshuffle, models)

    # plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(w=6.50127, h=5)
    fig.tight_layout(rect=[0, 0, 0.9, 1], pad=6)

    all_values = np.concatenate(
        [gender_pairs_values, random_pairs_values, shuffled_gender_pairs_values]
    )
    vmin, vmax = np.min(all_values), np.max(all_values)

    ax1 = sns.heatmap(
        gender_pairs_values[::-1, np.newaxis],
        yticklabels=gender_pairs_words,
        xticklabels=[],
        cmap=plt.get_cmap("PiYG"),
        ax=ax1,
        cbar=False,
        annot=True,
        center=np.zeros(1),
        linewidths=0.5,
        vmin=vmin,
        vmax=vmax,
    )
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=10)
    ax1.set_xlabel("gender word \n pairs")

    ax2 = sns.heatmap(
        random_pairs_values[::-1, np.newaxis],
        yticklabels=random_pairs_words,
        xticklabels=[],
        cmap=plt.get_cmap("PiYG"),
        ax=ax2,
        cbar=False,
        annot=True,
        center=0,
        linewidths=0.5,
        vmin=vmin,
        vmax=vmax,
    )
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=10)
    ax2.set_xlabel("random word \n pairs")

    ax3 = sns.heatmap(
        shuffled_gender_pairs_values[::-1, np.newaxis],
        yticklabels=shuffled_gender_pairs_words,
        xticklabels=[],
        cmap=plt.get_cmap("PiYG"),
        ax=ax3,
        cbar=False,
        annot=True,
        center=0,
        linewidths=0.5,
        vmin=vmin,
        vmax=vmax,
    )
    ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0, fontsize=10)
    ax3.set_xlabel("shuffled gender \n word pairs")

    plt.show()

