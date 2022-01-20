import gensim.models as gm
from bootstrap_sampling import *
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
    fdir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(fdir)

    # make model dirs
    name_file = os.path.splitext(data_path)[0]
    save_path = os.path.join(models_dir, name_file)
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving in {save_path}.")

    for i in range(n):
        file_name = "vectors_sample" + str(i + 1) + ".kv"
        file_path = os.path.join(save_path, file_name)
        word_vecs[i].save(file_path)


if __name__ == "__main__":
    import argparse
