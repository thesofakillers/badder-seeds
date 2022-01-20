import gensim.models as gm
from boostrap_sampling import *
from tqdm import tqdm
import os


def train_word2vec(data: list, params: dict) -> gm.keyedvectors.KeyedVectors:
    """Trains word2vec gensim model on fed sentence data
    :param list data: list of sentences to feed gensim model
    :param dict **params: parameters of the gensim function
    :returns KeyedVectors word_vectors: embeddings for gensim model keyed by word"""

    model = gm.Word2Vec(sentences=data, **params)
    return model.wv


def bootstrap_train(data_path: str, n: int, models_dir: str, params: dict) -> None:
    """Trains word2vec model through bootstrapping n times and saves.
    :param str data_path: path to the data to train on
    :param int n: number of times to bootstrap
    :param str models_dir: directory to model data
    :param dict params: parameters of word2vec model
    """

    samples = bootstrap(data_path, n)
    print("Training word2vec:")
    word_vecs = [train_word2vec(i, params) for i in tqdm(samples, unit="sample")]

    # TODO: save embeddings

    # get file dir and set it as working directory
    print("Saving:")
    fdir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(fdir)

    # extract data file name and make saving location
    name_file = os.path.splitext(data_path)
    # TODO
    save_path = ...
    for i in range(1, n + 1):
        file_name = "vectors" + str(i) + ".kv"
        # TODO: make path
        word_vecs[i - 1].save(path)


if __name__ == "__main__":
    import argparse
