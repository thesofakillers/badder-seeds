"""
converts json seed file in sensible and structured format 
resources: https://raw.githubusercontent.com/maria-antoniak/bad-seeds/main/gathered_seeds.json

"""
import re
import pandas as pd
from sklearn import datasets


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


def seedbank(dataset):
    """ 
    loads .json as pandas DataFrame

    Parametrs
    -----------
    dataset : string
        seed.json directory

    Returns
    --------
    seeds: Pandas DatatFrame
        ordered by category, DataFrame with seeds and meta information    
    """
    seeds = pd.read_json(dataset)
    seeds["Category"] = seeds["Category"].apply(clean)
    seeds = seeds.sort_values(by="Category")

    return seeds


if __name__ == "__main__":
    seeds = seedbank("../data/seeds/seeds.json")
    with pd.option_context(
        "display.max_rows",
        None,
        "display.precision",
        3,
    ):
        print(seeds)
