"""Functionality for metrics used in our work"""
import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
import gensim.models as gm
import pandas as pd


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
    embeddings,
    set1: list[str],
    set2: list[str],
    mode: str = "weat",
) -> float:
    """
    Compute coherence metric between two seed sets.

    Parameters
    ----------
    embeddings : dictionary of strings mapped to array of floats or gensim KeyedVectors struct.
        word embedding vectors keyed by word.
    set1 : array-like of strings
        seed set 1, N seed words
    set2 : array-like of strings
        seed set 2, N seed words
    mode : string
        Mode to use to extract bias subspace vector. Options are 'weat' and 'pca'. Default is 'weat'.

    Returns
    -------
    float
        Calculated coherence metric.
    """

    # obtain bias subspace vector according to mode
    if mode == "weat":
        # bias subspace is just difference of average vector of seed set (see Caliskan 2017)
        i = 0
        vset1 = [embeddings[w] for w in set1]
        vset2 = [embeddings[w] for w in set2]
        mean1, mean2 = np.mean(vset1, axis=0), np.mean(vset2, axis=0)
        bias_subspace_v = mean1 - mean2
    elif mode == "pca":
        # bias subspace is first principal component of PCA
        pca_matrix = do_pca(set1, set2, embeddings, num_components=1)
        bias_subspace_v = pca_matrix.components_[0, :]
    else:
        print("subspace mode type not yet implemented")

    # rank all words in vocab by cosine similarity to bias subspace
    # make matrix of vectors while keeping track of index of word
    if type(embeddings) == gm.KeyedVectors:
        matrix = np.zeros((len(embeddings.index_to_key), len(embeddings.vector_size)))
        for i, w in enumerate(embeddings.index_to_key):
            matrix[i, :] = embeddings[w]
    else:
        matrix = np.zeros((len(embeddings), len(list(embeddings.values())[0]))))
        idx2w = []
        for i, w in enumerate(embeddings.keys()):
            matrix[i, :] = embeddings[w]
            idx2w.append(w)

    # calculate cosine similarity between bias subspace and all words
    cos_sim = cosine_similarity(matrix, bias_subspace_v)

    # argsort gets us a ranking of words by cosine similarity to bias subspace
    cos_sim_rank = np.argsort(cos_sim, axis=0)

    # calculate mean rank of seed sets
    cumulative_rank1, cumulative_rank2 = 0
    for rank, idx in enumerate(cos_sim_rank):
        if idx2w[idx] in set1:
            cumulative_rank1 += rank
        if idx2w[idx] in set2:
            cumulative_rank2 += rank
    cumulative_rank1 /= len(set1)
    cumulative_rank2 /= len(set2)

    # calculate absolute difference in mean rank
    return np.abs(cumulative_rank1 - cumulative_rank2)
