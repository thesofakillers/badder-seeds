"""
replicate figure 2 from Antoniak et. al (2021)
"""


"""
Bias measurements depend on seeds. We cal-
culate the cosine similarities between different female
seed sets and an averaged upleasantness vector from
two embedding models. 

Results are consistent across
seeds for romance review embeddings, but vary widely
between sets for history and biography. We find simil



Figure 2 shows a motivating
example, in which we imagine a digital humanities
scholar interested in measuring whether women
are portrayed more negatively in different genres
of book reviews. As in the WEAT test, each seed is
plotted according to its cosine similarity to an aver-
aged unpleasantness vector (Caliskan et al., 2017).
For some sets, no significant difference is visible,
while for other sets, there are much larger differ-
ences, causing the researcher to draw different con-
clusions when comparing biases across datasets.

"""

import metrics
import seedbank
from sklearn.metrics.pairwise import cosine_similarity
import utils
import numpy as np


def figure_5(seeds, datasets):
    unpleasent = []
    embeds = []
    for data in datasets:
        for model in data:
            unpleasent.append((ultis.catch_keyerror('unpleasentness')))
            for seed in seeds:
                embeds.append(([catch_keyerror(models, word) for word in seed]))

            avg_embeds = np.mean(embeds, axis = 1)
            avg_unpleasent = np.mean(unpleasent)

            for i in avg_embeds:
                print(cosine_similarity(i, avg_unpleasent))
                


if __name__ == "__main__":
    seeds = seedbank.seedbanking("../data/seeds/seeds.json")
    seed_sets = ["female-Kozlowski_et_al_2019", "female_1-Caliskan_et_al_2017", "definitional_female-Bolukbasi_et_al_2016", "female_singular-Hoyle_et_al_2019", "female_definition_words_2-Zhao_et_al_2018", "female_stereotype_words-Zhao_et_al_2018"]
    extracted_seeds = seedbank.get_seeds(seeds, seed_sets)
    # seed = [item.lower() for item in seed_list[0]]

    datasets = []

    filenames = ["../data/models/history_biography_min10/", "../data/models/romance_min10/"]

    for f in filenames:
        models = []
        direct = os.fsencode(f)

        for filename in os.listdir(direct):
            print(filename)
            f = os.path.join(direct, filename)

            # checking if it is a file
            if os.path.isfile(f):
                f = os.fsdecode(f)
                models.append(KeyedVectors.load(f))
        
        datasets.append(models)

    






