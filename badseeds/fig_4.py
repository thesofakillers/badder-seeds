"""
Computes Fig 5 metrics given a set of parameters.
"""
import os
import random
import itertools
import argparse
import json
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import gensim.models as gm
import matplotlib.pyplot as plt

from badseeds import seedbank, utils, metrics, tab_a2


def comp_fig_4_metrics(
    seeds: pd.core.frame.DataFrame,
    pair_df: pd.core.frame.DataFrame,
    config: dict,
    corpus: str = "wiki",
    mode: str = "PCA",
    min_freq: int = 10,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Computes metrics for plotting figure 4 given a set of parameters.

    Parameters
    ----------
    seeds : pd.core.frame.DataFrame
        Dataframe containing of seeds produced by `badseeds.seedbank.seedbanking`.
    pair_df : pd.core.frame.DataFrame
        DataFrame of paired IDs, read from seed_set_pairings.csv.
    config : dict
        Dictionary specifying file paths.
    corpus : str, default wiki
        Which corpus to use. One of "goodreads_hb", "goodreads_r", "wiki", "nyt"
    mode : str, default PCA
        Which mode to use. One of "PCA", "Coherence"
    min_freq : int, default 10
        Minimum frequency of a word to be included in the model.
    seed : int, default 42
        Random seed.
    verbose : bool, default False
        Whether to print progress.

    Returns
    -------
    gathered_y_values : List[float]
        List of y values (Explained Variance Ratio or Coherence) for the gathered seeds
    generated_y_values : List[float]
        List of y values (Explained Variance Ratio or Coherence) for the generated seeds
    gathered_set_sim : List[float]
        List of set similarity scores for the gathered seeds
    generated_set_sim : List[float]
        List of set similarity scores for the generated seeds
    """
    assert corpus in [
        "goodreads_hb",
        "goodreads_r",
        "wiki",
        "nyt",
    ], "corpus must be one of goodreads_hb, goodreads_r, wiki, nyt"
    assert mode in ["PCA", "Coherence"], "mode must be one of PCA, Coherence"
    assert min_freq in [0, 10], "min_freq must be one of 0, 10"
    # set seed for replicability
    np.random.seed(42)
    random.seed(42)
    print(
        f"Reading model trained on {corpus} with min_freq={min_freq}..."
    ) if verbose else None
    embeddings_dir = os.path.join(
        config["models"]["dir_path"],
        config["models"][f"{corpus}_subpath"][str(min_freq)],
    )
    models = []
    for file in tqdm(os.listdir(embeddings_dir), disable=not verbose):
        if file.endswith(".kv"):
            models.append(gm.KeyedVectors.load(os.path.join(embeddings_dir, file)))
    print("Obtaining gathered and generated seed sets...") if verbose else None
    # gathered seed sets
    gathered_seeds = seeds["Seeds"]
    # generated seed sets
    # 50 generated seed sets of size 25, avoiding duplicates
    generated_seeds = []
    # mirrored set to for quick lookup
    generated_seed_sets = set()
    for model in tqdm(random.choices(models, k=50), disable=not verbose):
        while True:
            seed_list = utils.generate_seed_set(model, n=24)
            # make list hashable for duplicate checking
            seed_set = frozenset(seed_list)
            if seed_set not in generated_seed_sets:
                generated_seed_sets.add(seed_set)
                generated_seeds.append(seed_list)
                break
    print(
        "Obtaining gathered and generated seed set embeddings..."
    ) if verbose else None
    # embeddings
    gathered_seeds_embeddings = [
        utils.get_embeddings(seed_set, models, query_strat="average")
        for seed_set in tqdm(gathered_seeds, disable=not verbose)
    ]
    generated_seeds_embeddings = [
        utils.get_embeddings(seed_set, models, query_strat="average")
        for seed_set in tqdm(generated_seeds, disable=not verbose)
    ]
    print("Pairing gathered and generated seed sets...") if verbose else None
    #   gathered seeds
    # sorting for coherence match
    sorted_pair_df = pair_df.sort_values(["ID_A", "ID_B"], inplace=False)
    pair_ids = [list(x) for x in sorted_pair_df.to_records(index=False)]
    pair_idxs = [
        seeds[seeds["Seeds ID"].isin(pair)].index.to_list()
        for pair in tqdm(pair_ids, disable=not verbose)
    ]
    gathered_emb_pairs = [
        [gathered_seeds_embeddings[i], gathered_seeds_embeddings[j]]
        for (i, j) in tqdm(pair_idxs, disable=not verbose)
    ]
    #   generated seeds
    generated_pairs = list(itertools.combinations(generated_seeds, 2))
    generated_emb_pairs = list(itertools.combinations(generated_seeds_embeddings, 2))
    # sorting for coherence match
    gen_pair_sort_idxs = [
        i[0]
        for i in sorted(enumerate(generated_pairs), key=lambda x: (x[1][0], x[1][1]))
    ]
    sorted_gen_emb_pairs = [generated_emb_pairs[i] for i in gen_pair_sort_idxs]

    if mode == "PCA":
        print(
            "Computing gathered and generated seed set explained variance..."
        ) if verbose else None
        #   gathered seeds
        gathered_pca_models = [
            metrics.do_pca_embeddings(set_a, set_b, 3)
            for (set_a, set_b) in tqdm(gathered_emb_pairs, disable=not verbose)
        ]
        gathered_y_values = [
            model.explained_variance_ratio_[0] if model is not None else np.nan
            for model in gathered_pca_models
        ]
        #   generated seeds
        generated_pca_models = [
            metrics.do_pca_embeddings(set_a, set_b, 3)
            for (set_a, set_b) in tqdm(sorted_gen_emb_pairs, disable=not verbose)
        ]
        generated_y_values = [
            model.explained_variance_ratio_[0] if model is not None else np.nan
            for model in generated_pca_models
        ]
    elif mode == "Coherence":
        print(
            "Computing gathered and generated seed set coherence..."
        ) if verbose else None
        #   gathered seeds
        gathered_coherences = []
        for model in tqdm(models, disable=not verbose):
            coh = tab_a2.build_row_table_a2(
                model,
                seeds,
                pairing_method="file",
                pair_path="./seed_set_pairings.csv",
                nan_not_skip=True,
            )
            gathered_coherences.append(coh)
        gathered_y_values = tab_a2.agg_coherence(gathered_coherences, False)[
            "Coherence"
        ].tolist()
        #   generated seeds
        generated_coherences = []
        for model in tqdm(models, disable=not verbose):
            coh = tab_a2.build_row_table_a2(
                model,
                pd.DataFrame(data=pd.Series(generated_seeds), columns=["Seeds"]),
                pairing_method="all",
                nan_not_skip=True,
            )
            generated_coherences.append(coh)
        generated_y_values = tab_a2.agg_coherence(generated_coherences, False)[
            "Coherence"
        ].tolist()
    print(
        "Computing gathered and generated seed set similarity..."
    ) if verbose else None
    #   gathered seeds
    gathered_set_sim = [
        metrics.set_similarity(set_a, set_b, True)
        for (set_a, set_b) in tqdm(gathered_emb_pairs, disable=not verbose)
    ]
    #   generated seeds
    generated_set_sim = [
        metrics.set_similarity(set_a, set_b, True)
        for (set_a, set_b) in tqdm(sorted_gen_emb_pairs, disable=not verbose)
    ]
    print("Done.") if verbose else None
    return gathered_y_values, generated_y_values, gathered_set_sim, generated_set_sim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes Fig 4 metrics given a set of parameters."
    )
    parser.add_argument(
        "-c", "--config", type=str, default="./config.json", help="Path to config file."
    )
    parser.add_argument(
        "-d",
        "--corpus",
        type=str,
        default="wiki",
        help="Use embeddings from skip-gram trained on this corpus",
    )
    parser.add_argument(
        "-f",
        "--min_freq",
        type=int,
        default=10,
        help="Use embeddings from skip-gram trained"
        " with this minimum frequency of words",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="PCA",
        help="Use PCA or coherence metrics",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Seed for random number generator",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print progress to stdout",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    seeds = seedbank.seedbanking(
        config["seeds"]["dir_path"] + "seeds.json", index=False
    )
    pair_df = pd.read_csv(config["pairs"]["dir_path"] + "seed_set_pairings.csv")

    (
        gathered_y_values,
        generated_y_values,
        gathered_set_sim,
        generated_set_sim,
    ) = comp_fig_4_metrics(
        seeds,
        pair_df,
        config,
        args.corpus,
        args.mode,
        args.min_freq,
        args.seed,
        args.verbose,
    )

    # plot
    valid_idx = np.isfinite(generated_set_sim) & np.isfinite(generated_y_values)
    gen_coef = np.polyfit(
        np.array(generated_set_sim)[valid_idx],
        np.array(generated_y_values)[valid_idx],
        1,
    )
    gen_poly1d_fn = np.poly1d(gen_coef)
    # highlighted seed sets
    names_idx = pair_df[
        pair_df["ID_A"] == "white_names-Knoche_et_al_2019"
    ].index.to_list()[0]
    roles_idx = pair_df[
        pair_df["ID_B"] == "caucasian_roles-Manzini_et_al_2019"
    ].index.to_list()[0]

    fig, ax = plt.subplots(1, 1)

    fig.set_size_inches(w=6.50127, h=5)

    # generated
    ax.plot(
        generated_set_sim,
        generated_y_values,
        "o",
        generated_set_sim,
        gen_poly1d_fn(generated_set_sim),
        color="#B8CCE1",
        linewidth=3,
        markersize=10,
        markerfacecolor="#B8CCE1",
        markeredgecolor="white",
    )
    # gathered
    ax.plot(
        gathered_set_sim,
        gathered_y_values,
        "o",
        markersize=10,
        markerfacecolor="#F1B7B0",
        markeredgecolor="white",
    )

    highlighted_set_sim = [gathered_set_sim[idx] for idx in [names_idx, roles_idx]]
    highlighted_y_values = [gathered_y_values[idx] for idx in [names_idx, roles_idx]]
    # highlighted gathered
    ax.plot(
        highlighted_set_sim,
        highlighted_y_values,
        "o",
        markersize=10,
        markerfacecolor="#EC5E7B",
        markeredgecolor="white",
    )

    for i, label in enumerate(["Black vs White Names", "Black vs White Roles"]):
        ax.annotate(
            label,
            (highlighted_set_sim[i], highlighted_y_values[i]),
            xytext=(-10, 10),
            textcoords="offset points",
            horizontalalignment="right",
            color="#EC5E7B",
            weight="heavy",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="white"),
        )

    ax.set_xlabel("Set Similarity")
    ax.set_ylabel("Explained Variance Ratio" if args.mode == "PCA" else "Coherence")

    fig.set_tight_layout(True)
    plt.show()
