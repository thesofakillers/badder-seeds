"""Functionality for metrics used in our work"""
import numpy as np
import numpy.typing as npt
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore


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


def calc_pca():
    """PCA metric as described in Bolukbasi et al. (2016)"""
    # TODO
    raise NotImplementedError()