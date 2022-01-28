"""
replicates figure 2 from Antoniak et. al (2021)
"""
import metrics
import seedbank
import utils

import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import numpy.ma as ma
import os
from gensim.models import KeyedVectors
from itertools import repeat
import pandas as pd


def figure_2(seeds, datasets):
    """
    replicates figure 2 in Antoniak et al. (2016)

    Parametrs
    -----------
    seeds : list of lists
        list of list of seeds
    datasets: list of gensim KeyedVector type models
        list of models containing word embeddings

    Returns
    --------
    similarity: list of array types
        list of arrays with cosine similarity for each seedset and each dataset (model)
    """

    unpleasent = []
    similarity = []

    for data in datasets:
        embeds = [[] for i in range(len(seeds))]
        for model in data:
            u = utils.catch_keyerror(model, "unpleasantness")
            unpleasent.append(np.asarray(u if u is not None else 0))
            for i, seed in enumerate(seeds):
                embeds[i].append(
                    np.asarray(
                        [
                            utils.catch_keyerror(model, word)
                            if utils.catch_keyerror(model, word) is not None
                            else 0
                            for word in seed
                        ]
                    )
                )

        avg_embeds_list = [[] for i in range(len(seeds))]

        for i in range(len(seeds)):
            avg_embeds_list[i].append(np.mean(embeds[i], axis=1))

        avg_unpleasent = np.mean(unpleasent, axis=0)

        s = []
        for per_seed in avg_embeds_list:
            temp = []
            for idx, seed in enumerate(per_seed[0]):
                temp.append(cosine_similarity([seed], [avg_unpleasent.T])[0])
            s.append(np.asarray(temp).flatten())
        similarity.append(s)

    return similarity


if __name__ == "__main__":

    seeds = seedbank.seedbanking("../data/seeds/seeds.json")
    seeds.set_index("Seeds ID", inplace=True)
    seed_sets = [
        "female-Kozlowski_et_al_2019",
        "female_1-Caliskan_et_al_2017",
        "definitional_female-Bolukbasi_et_al_2016",
        "female_singular-Hoyle_et_al_2019",
        "female_definition_words_2-Zhao_et_al_2018",
        "female_stereotype_words-Zhao_et_al_2018",
    ]
    extracted_seeds = seedbank.get_seeds(seeds, seed_sets)
    # seed = [item.lower() for item in seed_list[0]]

    datasets = []

    filenames = [
        "../data/models/history_biography_min10/",
        "../data/models/romance_min10/",
    ]

    for f in filenames:
        models = []
        direct = os.fsencode(f)

        for filename in os.listdir(direct):
            # print(filename)
            f = os.path.join(direct, filename)

            # checking if it is a file
            if os.path.isfile(f):
                f = os.fsdecode(f)
                models.append(KeyedVectors.load(f))

        datasets.append(models)


    similarity = figure_2(extracted_seeds, datasets)

    for sim in similarity:
        for i, j in zip(extracted_seeds, sim):
            print(i, "\n")
            print(j, "\n \n")


    # viz
    df1 = pd.DataFrame(zip(similarity[0], seed_sets, ['history and biography']*len(seeds)), columns=["cosine similarity", "seed set", "dataset"])
    df2 = pd.DataFrame(zip(similarity[1],  seed_sets, ['romance']*len(seeds)), columns=["cosine similarity",  "seed set", "dataset"])
    
    df = pd.concat([df1,df2])
    df = df.explode('cosine similarity')
    df['cosine similarity'] = df['cosine similarity'].astype('float')

    # Creating plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots()
    ax1 = sns.boxplot(x="cosine similarity", y="seed set", hue="dataset", data=df, palette="Accent")
    ax2 = sns.stripplot(x="cosine similarity", y="seed set", hue="dataset", data=df, jitter=True,
              palette="Accent", dodge=True,linewidth=1,edgecolor='gray')
    
    legend = ax1.get_legend()
    handles = legend.legendHandles
    ax.legend(handles, ['history and biography', 'romance'])

    # show plot
    plt.show()
