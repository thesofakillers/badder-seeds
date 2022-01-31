import os

from torch import cosine_similarity
import gensim.models as gm
import pandas as pd
from tqdm import tqdm
import numpy as np

import metrics
import seedbank


def build_row_table4(model: gm.KeyedVectors, seeds: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a dataframe of coherence metrics
    for every possible pair of seed sets given embeddings
    :param gm.KeyedVectors model: embeddings model
    :param pd.Dataframe seeds: Dataframe of seed sets. Needs at least 1 "Seeds" column.
    :returns pd.DataFrame results: dataframe of coher. metrics for every poss. pair of seed sets
    """
    results = {
        "Coherence": [],
        "Gathered Set A": [],
        "Gathered Set B": [],
    }

    for i in range(seeds.shape[0]):
        for j in range(i + 1, seeds.shape[0]):
            if len(seeds.Seeds[i]) > 0 and len(seeds.Seeds[j]) > 0:
                try:
                    coh = metrics.coherence(model, seeds.Seeds[i], seeds.Seeds[j])
                except KeyError:
                    # print("One of seeds not found in model.")
                    continue
                results["Coherence"].append(coh)
                if "Category" in seeds.columns:
                    results["Gathered Set A"].append(
                        seeds.Category[i].upper() + ": " + str(seeds.Seeds[i]).strip()
                    )
                    results["Gathered Set B"].append(
                        seeds.Category[j].upper() + ": " + str(seeds.Seeds[j]).strip()
                    )
                else:
                    results["Gathered Set A"].append(str(seeds.Seeds[i]).strip())
                    results["Gathered Set B"].append(str(seeds.Seeds[j]).strip())
    # normalize
    results["Coherence"] /= np.max(results["Coherence"])

    # make into df
    results = pd.DataFrame(results)
    return results


def build_row_table4_window(
    model: gm.KeyedVectors, seeds: pd.DataFrame
) -> pd.DataFrame:
    """
    Builds a dataframe of coherence metrics with moving window as pairing method, given embeddings
    :param gm.KeyedVectors model: embeddings model
    :param pd.Dataframe seeds: Dataframe of seed sets. Needs at least 1 "Seeds" column.
    :returns dict results: dataframe of coher. metrics for consecutive pairs of seed sets
    """
    results = {
        "Coherence": [],
        "Gathered Set A": [],
        "Gathered Set B": [],
    }

    for i in range(seeds.shape[0] - 1):
        j = i + 1
        if len(seeds.Seeds[i]) > 0 and len(seeds.Seeds[j]) > 0:
            try:
                coh = metrics.coherence(model, seeds.Seeds[i], seeds.Seeds[j])
            except:
                # print("One of seeds not found in model.")
                continue
            results["Coherence"].append(coh)
            if "Category" in seeds.columns:
                results["Gathered Set A"].append(
                    seeds.Category[i].upper() + ": " + str(seeds.Seeds[i]).strip()
                )
                results["Gathered Set B"].append(
                    seeds.Category[j].upper() + ": " + str(seeds.Seeds[j]).strip()
                )
            else:
                results["Gathered Set A"].append(str(seeds.Seeds[i]).strip())
                results["Gathered Set B"].append(str(seeds.Seeds[j]).strip())
    # normalize
    results["Coherence"] /= np.max(results["Coherence"])

    # make into df
    results = pd.DataFrame(results)
    return results


if __name__ == "__main__":
    # NOTE: still missing Maria's response to question
    import argparse

    # get root dir and set it as working directory
    fdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(fdir)

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings_dir",
        "-d",
        default="models/nytimes_news_articles_min0",
        type=str,
        help="Path to directory of embeddings."
        " If relative path, relative to root directory."
        " Default is NYT dataset embeddings.",
    )
    args = parser.parse_args()

    # load embeddings
    models = []
    for file in os.listdir(args.embeddings_dir):
        if file.endswith(".kv"):
            models.append(gm.KeyedVectors.load(os.path.join(args.embeddings_dir, file)))

    if len(models) == 0:
        raise ValueError("No embeddings found in directory.")

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

    # get coherence numbers and normalize
    for model in tqdm(models, unit="models"):
        coh = build_row_table4_window(model, seeds)
        all_coherence.append(coh)

    # average coherence scores across seeds
    coh_con = pd.concat(all_coherence)
    coh_avg = coh_con.groupby(["Gathered Set A", "Gathered Set B"]).agg(
        {"Coherence": ["mean"]}
    )
    coh_avg.columns = ["Coherence"]
    coh_avg = coh_avg.reset_index()

    # sort and display
    coh_avg = coh_avg[["Coherence", "Gathered Set A", "Gathered Set B"]]
    print(coh_avg.columns)
    coh_avg = coh_avg.sort_values(by="Coherence", ascending=False)
    coh_avg.Coherence = coh_avg.Coherence.round(3)
    with pd.option_context("display.max_rows", 20):
        print(coh_avg)

    # TODO: part 2: random seeds
    # TODO: refactor in prettier way

    # generate random seeds

    # coherence for each seed (normalize)

    # pair and rank

    # save to file
