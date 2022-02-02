import os, string

import gensim.models as gm
import pandas as pd
from tqdm import tqdm
import numpy as np
import random

from badseeds import utils, metrics, seedbank


def agg_coherence(all_coh):
    """
    Returns average and rounded coherence across models
    :param list all_coh: list of coherence dataframes for each model
    :returns pd.DataFrame coh_avg: dataframe of average coherence across models for seed pairs
    """

    # average coherence scores across models
    coh_con = pd.concat(all_coh)
    coh_avg = coh_con.groupby(["Set A", "Set B"]).agg({"Coherence": ["mean"]})
    coh_avg.columns = ["Coherence"]
    coh_avg = coh_avg.reset_index()

    # sort and display
    coh_avg = coh_avg[["Coherence", "Set A", "Set B"]]
    coh_avg = coh_avg.sort_values(by="Coherence", ascending=False)
    coh_avg.Coherence = coh_avg.Coherence.round(3)

    return coh_avg


def append_row(coh, results, seeds, i, j, pretty_category: bool = False):
    """
    Appends a row of coherence metrics and seed sets to results dataframe
    :param float coh: coherence score
    :param dict results: dict of lists {'Coherence': [], 'Set A': [], 'Set B': []}
    :param pd.DataFrame seeds: Dataframe of seed sets. Needs 1 "Seeds" column.
    :param int i: index of seed set A
    :param int j: index of seed set B
    :param bool pretty_category: whether to use category names to pretty print
    :returns None
    """
    # build row
    results["Coherence"].append(coh)

    if pretty_category:
        results["Set A"].append(seeds.Category[i].upper() + ": " + str(seeds.Seeds[i]))
        results["Set B"].append(seeds.Category[j].upper() + ": " + str(seeds.Seeds[j]))
    else:
        results["Set A"].append(str(seeds.Seeds[i]).strip())
        results["Set B"].append(str(seeds.Seeds[j]).strip())
    return


def build_row_table4(
    model: gm.KeyedVectors,
    seeds: pd.DataFrame,
    pairing_method: str = "window",
    pair_path: str = None,
    category_print: bool = False,
) -> pd.DataFrame:
    """
    Builds a dataframe of coherence metrics
    for every possible pair of seed sets given embeddings
    :param gm.KeyedVectors model: embeddings model
    :param pd.Dataframe seeds: Dataframe of seed sets. Needs at least 1 "Seeds" column.
    :param str pairing_method: pairing method to use.
        'window' for moving window, 'all' for all possible pairs
        'file' when loading pairing data, requires pair_path
    :param str pair_path: path to pairing data, if pairing_method is 'file'. Default is None
    :param bool category_print: whether to print the seed category or not
    :returns pd.DataFrame results: dataframe of coher. metrics for every poss. pair of seed sets
    """
    if pairing_method == "file":
        if pair_path:
            pairs = pd.read_csv(pair_path)
        else:
            print("Need a path to pairing data if pairing_method is 'file'")

    results = {
        "Coherence": [],
        "Set A": [],
        "Set B": [],
    }
    if pairing_method != "file":
        for i in range(seeds.shape[0]):
            if pairing_method == "window":
                lim = min(i + 2, seeds.shape[0])
            else:
                lim = seeds.shape[0]
            for j in range(i + 1, lim):
                if len(seeds.Seeds[i]) > 0 and len(seeds.Seeds[j]) > 0:
                    try:
                        # to avoid overlapping seeds
                        if set(seeds.Seeds[i]) & set(seeds.Seeds[j]):
                            continue
                        coh = metrics.coherence(model, seeds.Seeds[i], seeds.Seeds[j])
                    except KeyError:
                        # print("One of seeds not found in model.")
                        continue
                    append_row(
                        coh, results, seeds, i, j, pretty_category=category_print
                    )
    else:
        for k in range(pairs.shape[0]):
            # find seeds in seeds dataframe
            id1 = pairs[pairs.columns[0]][k]
            id2 = pairs[pairs.columns[1]][k]
            i = seeds[seeds["Seeds ID"] == id1].index[0]
            j = seeds[seeds["Seeds ID"] == id2].index[0]

            # do coherence
            if len(seeds.Seeds[i]) > 0 and len(seeds.Seeds[j]) > 0:
                try:
                    coh = metrics.coherence(model, seeds.Seeds[i], seeds.Seeds[j])
                except KeyError:
                    continue

                append_row(coh, results, seeds, i, j, pretty_category=category_print)

    # normalize
    results["Coherence"] /= np.max(results["Coherence"])

    # make into df
    results = pd.DataFrame(results)
    return results


if __name__ == "__main__":
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
        help="Path to directory of embeddings."
        " If relative path, relative to root directory."
        " Default is NYT dataset embeddings.",
    )
    parser.add_argument(
        "--mode",
        "-m",
        default="gathered",
        type=str,
        help="Generated or gathered seeds.",
    )
    args = parser.parse_args()

    # load embeddings
    models = []
    for file in os.listdir(args.embeddings_dir):
        if file.endswith(".kv"):
            models.append(gm.KeyedVectors.load(os.path.join(args.embeddings_dir, file)))

    if len(models) == 0:
        raise ValueError("No embeddings found in directory.")

    if args.mode == "gathered":
        # part 1: gathered seeds
        # load in all gathered seeds to memory, clean up
        seeds = seedbank.seedbanking("data/seeds/seeds.json")
        prefix = "Gathered"

        # get coherence numbers
        all_coherence = []
        for model in tqdm(models, unit="model"):
            coh = build_row_table4(
                model,
                seeds,
                pairing_method="file",
                pair_path="./seed_set_pairings.csv",
                category_print=True,
            )
            all_coherence.append(coh)

        coh_avg = agg_coherence(all_coherence)

        coh_avg.columns = ["Coherence", prefix + " Set A", prefix + " Set B"]
        coh_avg.replace(r"\[|\]|'", "", regex=True, inplace=True)

        coh_avg.to_csv("data/table4_gathered.csv", index=False)

        # display
        with pd.option_context("display.max_rows", 9, "display.max_colwidth", 60):
            print(coh_avg)

    elif args.mode == "generated":
        check = string.printable
        np.random.seed(42)
        random.seed(42)
        # generate random seeds, ignore non-alpha characters
        sampled = []
        for model in random.choices(models, k=50):
            while True:
                s = utils.generate_seed_set(model)
                if 0 not in [c in check for w in s for c in w]:
                    sampled.append(s)
                    break
        g_seeds = pd.DataFrame(data=pd.Series(sampled), columns=["Seeds"])

        prefix = "Generated"

        # do coherence
        all_coherence = []
        for model in tqdm(models, unit="model"):
            coh = build_row_table4(model, g_seeds, pairing_method="all")
            all_coherence.append(coh)

        coh_avg = agg_coherence(all_coherence)

        coh_avg.columns = ["Coherence", prefix + " Set A", prefix + " Set B"]
        coh_avg.replace(r"\[|\]|'", "", regex=True, inplace=True)

        coh_avg.to_csv("data/table4_generated.csv", index=False)

        # display
        with pd.option_context("display.max_rows", 9, "display.max_colwidth", 100):
            print(coh_avg)

    else:
        raise NotImplementedError
