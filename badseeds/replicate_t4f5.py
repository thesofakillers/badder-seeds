import utils, os
import gensim.models as gm
import seedbank
import pandas as pd
from tqdm import tqdm
import numpy as np


def build_row_table4(model: gm.KeyedVectors, seeds: pd.DataFrame) -> dict:
    results = {"Coherence": [], "Gathered Set A": [], "Gathered Set B": []}
    skipped = []
    for i in range(seeds.shape[0]):
        for j in range(i + 1, seeds.shape[0]):
            if len(seeds.Seeds[i]) > 0 and len(seeds.Seeds[j]) > 0:
                if "Category" in seeds.columns:
                    try:
                        coh = utils.coherence(s, seeds.Seeds[i], seeds.Seeds[j])
                    except:
                        # print("One of seeds not found in model.")
                        break
                    results["Coherence"].append(coh)
                    results["Gathered Set A"].append(
                        [seeds.Category[i]] + seeds.Seeds[i]
                    )
                    results["Gathered Set B"].append(
                        [seeds.Category[j]] + seeds.Seeds[j]
                    )
    return results


if __name__ == "__main__":
    # NOTE: still missing Maria's response to question
    # TODO: think about how to rank coherence stuff
    import argparse

    # get root dir and set it as working directory
    fdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(fdir)

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings_dir",
        "-d",
        default="models/nytimes_news_articles_min10",
        type=str,
        help="Path to directory of embeddings. If relative path, relative to root directory. Default is NYT dataset embeddings.",
    )
    args = parser.parse_args()

    # load embeddings
    models = []
    for file in os.listdir(args.embeddings_dir):
        if file.endswith(".kv"):
            models.append(gm.KeyedVectors.load(os.path.join(args.embeddings_dir, file)))

    # part 1: gathered seeds
    # load in all gathered seeds to memory, clean up
    all_coherence = []
    seeds = seedbank.seedbanking("data/seeds/seeds.json")
    seeds = seeds.sort_index()
    seeds["Seeds"] = (
        seeds["Seeds"].str.replace("[\[\]']", "", regex=True).str.split(", ")
    )

    seeds["Seeds"] = seeds["Seeds"].map(
        lambda x: [] if len(x) == 1 and x[0] == "" else x
    )

    # empty seed sets
    criterion = seeds.Seeds.map(lambda x: len(x) == 0)
    print(seeds[criterion])

    # get coherence numbers and normalize
    for s in tqdm(models, unit="models"):
        coh = build_row_table4(s, seeds)
        coh["Coherence"] /= np.max(coh["Coherence"])
        all_coherence.append(coh)
    print(all_coherence[0])

    # average coherence scores across seeds

    # pair and rank

    # part 2: random seeds

    # generate random seeds

    # coherence for each seed (normalize)

    # pair and rank

    # save to file
