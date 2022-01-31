"""
converts json seed file in sensible and structured format
resources:
    https://raw.githubusercontent.com/maria-antoniak/bad-seeds/main/gathered_seeds.json

"""
import argparse
import json
import pandas as pd
from sklearn import datasets
import os


def clean(categories):
    """
    cleans seed .json file, sorts by category

    Parametrs
    -----------
    categories : pandas DatatFrame
        DataFrame with seeds and meta information

    Returns
    --------
    x: Pandas DatatFrame
        cleaned categories DataFrame with seeds and meta information

    """

    words = ["seed", "words", "terms", "attributes"]
    # print(''.join([i for i in categories if not i.isdigit()]))
    x = "".join([i for i in categories if not i.isdigit()])
    x = " ".join([w for w in x.split() if not w in words])
    x = x.replace("_", " ")
    x = x.replace("/", " ")

    return x


def get_seeds(seeds, seed_list, id_loc="index"):
    """
    returns seed by seed id

    Parametrs
    -----------
    seeds: pd DataFrame
        dataframe with seeds, fetched via seedbanking
    seed_list : list of strings
        list of seed IDs
    id_loc : string, default "index"
        name of column containing seed IDs
        if "index" then seeds.index is used

    Returns
    --------
    extracted_seeds: list of lists
        list of seeds

    """
    if id_loc == "index":
        extracted_seeds = seeds.loc[seed_list, ["Seeds"]].values.tolist()
    else:
        extracted_seeds = seeds[seeds[id_loc].isin(seed_list)]["Seeds"].values.tolist()
    return extracted_seeds


def seedbanking(dataset, index=False):
    """
    loads .json as pandas DataFrame

    Parametrs
    -----------
    dataset : string
        seed.json directory
    index: boolean, default False
        whether to use "Seed ID" as index

    Returns
    --------
    seeds: Pandas DataFrame
        ordered by category, DataFrame with seeds and meta information
    """
    seeds = pd.read_json(dataset)
    seeds["Category"] = seeds["Category"].apply(clean)
    # convert string representation of list to list
    seeds["Seeds"] = seeds["Seeds"].apply(lambda x: eval(clean(x)))
    if index:
        seeds.set_index("Seeds ID", inplace=True)

    # remove bigrams in seed sets
    seeds["Seeds"] = seeds["Seeds"].apply(
        lambda x: [i for i in x if len(i.split()) == 1]
    )

    # remove seed sets with only one element
    seeds = seeds[seeds["Seeds"].apply(len) > 1]
    seeds.reset_index(inplace=True)

    return seeds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="displays seeds")
    parser.add_argument(
        "-c",
        "--config",
        help="config JSON file containing path to seeds",
        type=str,
        default="./config.json",
    )

    fdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(fdir)

    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    seeds = seedbanking(config["seeds"]["dir_path"] + "seeds.json")
    with pd.option_context(
        "display.max_rows",
        20,
        "display.max_columns",
        3,
        "display.precision",
        3,
    ):
        print(seeds)
