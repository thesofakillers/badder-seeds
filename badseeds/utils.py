import numpy as np
from preprocess import read_pproc_dataset
from metrics import *
import gensim.models as gm


def bootstrap(dataset, n=20):
    """ "
    makes bootstrap samples from given dataset

    Parameters
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
    x = read_pproc_dataset(dataset)

    print(type(x))
    bootstrap_samples = []
    data = np.asarray(x)
    length = len(data)
    for i in range(n):
        bootstrap_samples.append(np.random.choice(data, replace=True, size=length))

    return bootstrap_samples


def generate_seed_set(
    embeddings, f: list[str] = ["NN", "NNP"], n: int = 4
) -> list[str]:
    """
    Generate random seed set.

    Parameters
    ----------
    embeddings : dictionary of strings mapped to array of floats or gensim KeyedVectors struct.
        word embedding vectors keyed by word.
    f : list of strings
        Only words with the following POS tags will be selected
    mode : string
        Mode to use to extract bias subspace vector. Options are 'weat' and 'pca'. Default is 'weat'.

    Returns
    -------
    float
        Calculated coherence metric.
    """

    # randomly pick word that matches POS
    success = False
    vocab_len = len(embeddings.index_to_key)
    while not success:
        if type(embeddings) == gm.KeyedVectors:
            idx = np.random.choice(vocab_len)
            sample = embeddings.index_to_key[idx]
            tags = embeddings.get_vecattr(sample, "pos")
            for f_tag in f:
                if f_tag in tags:
                    success = True
        else:
            print("did not feed gensim KeyedVectors struct as embeddings")
            raise NotImplementedError

    first = [sample]

    # find n closest vectors
    neighbors_result = np.argsort(embeddings.most_similar(positive=first, topn=None))
    neighbors_result = neighbors_result[:-1][::-1]
    neighbors = []
    found = 0
    for i in neighbors_result:
        w = embeddings.index_to_key[i]
        tags = embeddings.get_vecattr(w, "pos")
        for f_tag in f:
            if f_tag in tags:
                neighbors.append(w)
                found += 1
                break
        if found == n:
            break

    # return seed list
    return first + neighbors
