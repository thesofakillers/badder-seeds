import numpy as np
from badseeds.preprocess import read_pproc_dataset, docbin_to_docs
import gensim.models as gm
import scipy.stats.stats as st


def get_embeddings(word_list, models, query_strat="average"):
    """
    Gets embeddings for a list of words

    Parameters
    -----------
    word_list: array-like of strings
        list of words to get embeddings for
    models: list of gensim.models.KeyedVectors
        list of KeyVectors objects mapping word:embedding
        representing a set of models that may contain
        embeddings for the words in word_list
    query_strat: string, default "average"
        strategy to use to get embeddings. Options are:
        - "average": average the embeddings of all models that contain the word.
        - "skip": skips word if one model does not contain it, otherwise average
        - "skip-model": averaging across only models that contain all the words
        - "first": take the embedding of the first model that contains the word.

    Returns
    -------
    embeddings: array-like of floats
        array of embeddings for the words in word_list
        if 'skip', respective embedding for skipped words will be array-like of np.nan
    """
    if query_strat == "average":
        embeddings = get_average_embeddings(word_list, models, allow_missing=True)
    elif query_strat == "skip":
        embeddings = get_average_embeddings(word_list, models, allow_missing=False)
    elif query_strat == "first":
        embeddings = get_first_embeddings(word_list, models)
    elif query_strat == "skip-model":
        raise NotImplementedError("skip-model strategy not implemented")
    else:
        raise ValueError("query_strat must be one of 'average', 'skip', or 'first'")
    return embeddings


def get_first_embeddings(word_list, models):
    """
    Gets embeddings for a list of words from the first model that contains the word

    Parameters
    -----------
    word_list: array-like of strings
        list of W words to get embeddings for
    models: list of gensim.models.KeyedVectors
        list of KeyVectors objects mapping word:embedding
        representing a set of M models that may contain
        D-dimensional embeddings for the words in word_list

    Returns
    -------
    embeddings: array-like of floats
        (W, D) array of embeddings for the words in word_list
        for words not found in any model, the embedding is set to an array of np.NaN
    """
    num_words = len(word_list)
    # assuming embeddings in models are all the same dimensionality
    emb_size = len(models[0][0])
    embeddings = np.full((num_words, emb_size), np.nan)
    for i, word in enumerate(word_list):
        # in case word is not found in any model, set embedding to NaN
        embedding = np.full(emb_size, np.nan)
        for model in models:
            if word in model:
                embedding = model[word]
                # exit loop early if word found
                break
        embeddings[i] = embedding
    return embeddings


def get_average_embeddings(word_list, models, allow_missing=True):
    """
    Gets average embeddings for a list of words across a set of models

    Parameters
    -----------
    word_list: array-like of strings
        list of W words to get embeddings for
    models: list of gensim.models.KeyedVectors
        list of KeyedVectors
        representing a set of M models that may contain
        D-dimensional embeddings for the words in word_list
    allow_missing: boolean, default True
        if True, when embeddings are missing in a model,
        the average is taken over all models that contain the word.
        if False, the embedding is set to an array of np.NaN for the missing word.


    Returns
    -------
    embeddings: array-like of floats
        (W, D) array of embeddings for the words in word_list
        The embedding is set to an array of np.NaN when none
        of the models contain the word, regardless of allow_missing
    """
    num_models = len(models)
    num_words = len(word_list)
    # assuming embeddings in models are all the same dimensionality
    emb_size = len(models[0][0])

    embeddings = np.full((num_words, emb_size), np.nan)
    # get (M, D) embedding for word
    for w, word in enumerate(word_list):
        # first initialize with NaNs
        embedding = np.full((num_models, emb_size), np.nan)
        # fill in embeddings for models that contain the word
        for i, model in enumerate(models):
            if word in model:
                embedding[i, :] = model[word]
        # can now get (D, ) average embedding
        if allow_missing:
            # if word not found anywhere, just return array of nans
            if np.isnan(embedding).all():
                embedding = np.full(emb_size, np.nan)
            # average embedding, allow for some nans
            else:
                embedding = np.nanmean(embedding, axis=0)
        else:
            # average embeddings, not allowing for nans
            embedding = np.mean(embedding, axis=0)
            # if any nans in result, set entire embedding to NaN
            if np.isnan(embedding).any():
                embedding = np.full((emb_size), np.nan)

        # set embedding for word
        embeddings[w] = embedding

    return embeddings


def bootstrap(dataset, n=20):
    """
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
    data = np.asarray(list(docbin_to_docs(x)), dtype=object)
    length = len(data)
    for i in range(n):
        bootstrap_samples.append(np.random.choice(data, replace=True, size=length))

    return bootstrap_samples


def generate_seed_set(
    embeddings, f: list[str] = ["NN", "NNS"], n: int = 4
) -> list[str]:
    """
    Generate random seed set.

    Parameters
    ----------
    embeddings : dictionary of strings mapped to array of floats or gensim KeyedVectors struct.
        word embedding vectors keyed by word.
    f : list of strings
        Only words with the following POS tags will be selected. Default is only common nouns (singular and plural)
    mode : string
        Mode to use to extract bias subspace vector. Options are 'weat' and 'pca'. Default is 'weat'.

    Returns
    -------
    list
        list of n + 1 seed words.
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


def catch_keyerror(models, word):
    """
    gets word from first model, but doesnt throw an error when no key found

    Parametrs
    -----------

    models: list of KeyedVector
        list of skipgram models
    word: string
        word that we want to get the embedding from

    """
    try:
        if type(models) is list:
            avg = [
                catch_keyerror(model, word)
                if catch_keyerror(model, word) is not None
                else np.zeros((100))
                for model in models
            ]
            return np.mean(avg, axis=0)
        else:
            return models[word]
    except KeyError as e:
        print(e)
        return None
