"""
converts json seed file in sensible and structured format 
resources: https://raw.githubusercontent.com/maria-antoniak/bad-seeds/main/gathered_seeds.json

"""
import re
import pandas as pd


def clean(categories):
    """ """
    words = ["seed", "words", "terms", "attributes"]
    # print(''.join([i for i in categories if not i.isdigit()]))
    x = "".join([i for i in categories if not i.isdigit()])
    x = " ".join([w for w in x.split() if not w in words])
    x = x.replace("_", " ")
    x = x.replace("/", " ")

    return x


def seedbank():
    """ """
    seeds = pd.read_json("../data/seeds/seeds.json")
    seeds["Category"] = seeds["Category"].apply(clean)
    seeds = seeds.sort_values(by="Category")

    return seeds


if __name__ == "__main__":
    seeds = seedbank()
    with pd.option_context(
        "display.max_rows",
        None,
        "display.precision",
        3,
    ):
        print(seeds)
