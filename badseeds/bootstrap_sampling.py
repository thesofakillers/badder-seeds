""" 
bootstrap sampling of given dataset 
"""

import pickle
import numpy as np
from preprocess import read_file


def bootstrap(dataset, n=20):
    """ "
    makes bootstrap samples from given dataset

    Parametrs
    -----------
    n : int
        number of bootstrap samples

    dataset: string
        name of the dataset

    Returns
    --------
    bootstrap_samples: list of arrays
        list of arrays (bootstrapped samples)
    """

    # load in file
    x = read_file(dataset)

    print(type(x))
    bootstrap_samples = []
    data = np.asarray(x)
    length = len(data)
    for i in range(n):
        bootstrap_samples.append(np.random.choice(data, replace=True, size=length))

    return bootstrap_samples


if __name__ == "__main__":
    bootstrap("../data/processed/nytimes_news_articles.pkl")
