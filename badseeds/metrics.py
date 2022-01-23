"""Functionality for metrics used in our work"""
import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
import gensim.models as gm


def comp_assoc(
    target_set: npt.NDArray[np.float_],
    attr_set_a: npt.NDArray[np.float_],
    attr_set_b: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    """
    Compute associations between target set and attribute sets.

    Parameters
    ----------
    target_set : array-like of float
        (N, D) array of word embeddings, constituting the target set.
    attr_set_a : array-like of float
        (A, D) array of word embeddings, constituting the first attribute set.
    attr_set_b : array-like of float
        (B, D) array of word embeddings, constituting the second attribute set.

    Returns
    -------
    array-like of float
        (N, 1) Array of association values.
    """
    # (N, A) array
    cos_sim_a = cosine_similarity(target_set, attr_set_a)
    # (N, B) array
    cos_sim_b = cosine_similarity(target_set, attr_set_b)
    # (N, ) array
    return cos_sim_a.mean(axis=1) - cos_sim_b.mean(axis=1)


def calc_weat(
    target_set_x: npt.NDArray[np.float_],
    target_set_y: npt.NDArray[np.float_],
    attr_set_a: npt.NDArray[np.float_],
    att_set_b: npt.NDArray[np.float_],
) -> float:
    """
    Calculate WEAT given two sets of target and attribute word-embeddings.
    X, Y, A and B are the number of elements in each array.
    D is the dimensionality of the word-embeddings.

    Parameters
    ----------
    target_set_x : array-like of float
        (X, D) array of word embeddings, constituting the first target set.
    target_set_y : array-like of float
        (Y, D) array of word embeddings, constituting the second target set.
    attr_set_a : array-like of float
        (A, D) array of word embeddings, constituting the first attribute set.
    att_set_b : array-like of float
        (B, D) array of word embeddings, constituting the second attribute set.

    Returns
    -------
    float
        WEAT metric.
    """
    # (X,) array
    t_set_1_assoc: npt.NDArray[np.float_] = comp_assoc(
        target_set_x, attr_set_a, att_set_b
    )
    # (Y, ) array
    t_set_2_assoc: npt.NDArray[np.float_] = comp_assoc(
        target_set_y, attr_set_a, att_set_b
    )
    # scalar
    weat_score: float = t_set_1_assoc.sum() - t_set_2_assoc.sum()
    return weat_score


def do_pca(seed1, seed2, embedding, num_components=10):
    """
    PCA metric as described in Bolukbasi et al. (2016).
    original code base of the authors: https://github.com/tolga-b/debiaswe/blob/master/debiaswe/we.py

    Embeddings of bias word pairs are used to calculate the bias subspace.
    Number of words in each word set N.

    Parameters
    ----------
    seed1 : array-like of strings
        (10,1) array of strings,

     seed2 : array-like of strings
        (10,1) array of strings,

    embedding : dictionary of strings mapped to array of floats
        (N) string mapped to array of floats, maps word to its embedding

    num_components : int
        indicates number of principal components wanted for extraction

    Returns
    -------
    pca : matrix of floats
        (num_components, B) matrix of floats, consitutes principle components of bias direction = bias subspace

    """

    matrix = []
    for a, b in zip(seed1, seed2):
        center = (embedding[a] + embedding[b]) / 2
        matrix.append(embedding[a] - center)
        matrix.append(embedding[b] - center)
    matrix = np.array(matrix)
    pca = PCA(n_components=num_components)
    pca.fit(matrix)

    return pca


def coherence(
    embeddings: gm.KeyedVectors,
    set1: list[str],
    set2: list[str],
    mode: str = "weat",
    **kwargs
) -> float:

    # obtain bias subspace vector according to mode
    if mode == "weat":
        i = 0
        vset1 = [embeddings[w] for w in set1]
        vset2 = [embeddings[w] for w in set2]
        mean1, mean2 = np.mean(vset1, axis=0), np.mean(vset2, axis=0)
        bsubspace_v = mean1 - mean2
    elif mode == "pca":
        matrix = do_pca(set1, set2, embeddings)

    else:
        print("subspace mode type not yet implemented")

    # rank all words in vocab by cosine similarity to bias subspace

    # calculate mean rank of paired seed sets

    # calculate absolute difference in mean rank
    pass
