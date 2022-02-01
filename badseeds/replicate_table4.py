import os

import gensim.models as gm
import pandas as pd
from tqdm import tqdm
import numpy as np
import utils
import random

import metrics
import seedbank


def agg_coherence(all_coh, seed_gen="gathered"):

    if seed_gen == "gathered":
        prefix = "Gathered"
    else:
        prefix = "Generated"

    # average coherence scores across models
    coh_con = pd.concat(all_coh)
    coh_avg = coh_con.groupby([prefix + " Set A", prefix + " Set B"]).agg(
        {"Coherence": ["mean"]}
    )
    coh_avg.columns = ["Coherence"]
    coh_avg = coh_avg.reset_index()

    # sort and display
    coh_avg = coh_avg[["Coherence", prefix + " Set A", prefix + " Set B"]]
    coh_avg = coh_avg.sort_values(by="Coherence", ascending=False)
    coh_avg.Coherence = coh_avg.Coherence.round(3)

    return coh_avg


def build_row_table4(
    model: gm.KeyedVectors,
    seeds: pd.DataFrame,
    pairing_method: str = "window",
    seed_gen: str = "gathered",
) -> pd.DataFrame:
    """
    Builds a dataframe of coherence metrics
    for every possible pair of seed sets given embeddings
    :param gm.KeyedVectors model: embeddings model
    :param pd.Dataframe seeds: Dataframe of seed sets. Needs at least 1 "Seeds" column.
    :param str pairing_method: pairing method to use.
        'window' for moving window, 'all' for all possible pairs
    :param str seed_gen: 'gathered' for seeds from gathered set, 'generated' for seeds from generated set
    :returns pd.DataFrame results: dataframe of coher. metrics for every poss. pair of seed sets
    """

    if seed_gen == "gathered":
        prefix = "Gathered"
    else:
        prefix = "Generated"
    results = {
        "Coherence": [],
        prefix + " Set A": [],
        prefix + " Set B": [],
    }
    for i in range(seeds.shape[0]):
        if pairing_method == "window":
            lim = min(i + 2, seeds.shape[0])
        else:
            lim = seeds.shape[0]
        for j in range(i + 1, lim):
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
                    results["Generated Set A"].append(str(seeds.Seeds[i]).strip())
                    results["Generated Set B"].append(str(seeds.Seeds[j]).strip())
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
        all_coherence = []
        seeds = seedbank.seedbanking("data/seeds/seeds.json")

        # get coherence numbers
        for model in tqdm(models, unit="model"):
            coh = build_row_table4(model, seeds, seed_gen=args.mode)
            all_coherence.append(coh)

        coh_avg = agg_coherence(all_coherence, seed_gen=args.mode)
        coh_avg.to_csv("data/table4_gathered.csv", index=False)

        # display
        with pd.option_context("display.max_rows", 7):
            print(coh_avg)

    elif args.mode == "generated":

        np.random.seed(42)
        random.seed(42)
        # generate random seeds
        g_seeds = pd.DataFrame(
            data=pd.Series(
                [
                    utils.generate_seed_set(model)
                    for model in random.choices(models, k=50)
                ]
            ),
            columns=["Seeds"],
        )

        # do coherence
        all_coherence = []
        for model in tqdm(models, unit="model"):
            coh = build_row_table4(
                model, g_seeds, pairing_method="all", seed_gen=args.mode
            )
            all_coherence.append(coh)

        coh_avg = agg_coherence(all_coherence, seed_gen=args.mode)
        coh_avg.to_csv("data/table4_generated.csv", index=False)

        # display
        with pd.option_context("display.max_rows", 7):
            print(coh_avg)

    else:
        raise NotImplementedError
