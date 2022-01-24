import gensim.models as gm
from bootstrap_sampling import *
from tqdm import tqdm
import os
import numpy as np


def train_word2vec(data: list, params: dict) -> gm.keyedvectors.KeyedVectors:
    """Trains word2vec gensim model on fed sentence data
    :param list data: list of sentences to feed gensim model
    :param dict **params: parameters of the gensim function
    :returns KeyedVectors word_vectors: embeddings for gensim model keyed by word"""

    model = gm.Word2Vec(sentences=data, **params)
    return model.wv


def bootstrap_train(data_path: str, models_dir: str, params: dict, n: int = 20) -> list:
    """Trains word2vec model through bootstrapping n times and saves.
    :param str data_path: path to the data to train on
    :param int n: number of times to bootstrap
    :param str models_dir: directory to model data
    :param dict params: parameters of word2vec model
    :returns list embeddings: list of embeddings for each bootstrap
    """

    samples = bootstrap(data_path, n)
    print("Building gensim input:")
    text_samples = [
        [[token.text for token in doc] for doc in s]
        for s in tqdm(samples, unit="bootstrap sample")
    ]
    print("\nTraining word2vec:")
    word_vecs = [
        train_word2vec(i, params) for i in tqdm(text_samples, unit="bootstrap sample")
    ]

    # make model dirs
    name_file = os.path.splitext(data_path)[0].split("/")[-1]
    save_path = os.path.join(models_dir, name_file)
    os.makedirs(save_path, exist_ok=True)
    print(f"\nSaving in {save_path}.")

    # save embeddings
    # do not save if already exists
    for i in range(n):
        file_name = "vectors_sample" + str(i + 1) + ".kv"
        file_path = os.path.join(save_path, file_name)
        if os.path.exists(file_path):
            print(
                f"{file_path} already exists, skipping. Delete this file to save re-trained vectors."
            )
        else:
            word_vecs[i].save(file_path)

    return word_vecs


if __name__ == "__main__":
    import argparse

    # Command line arguments
    parser = argparse.ArgumentParser()

    # directory to save model in and data file path
    # NOTE: once we have a working preprocess file that does things we can just loop over all dataset directories
    parser.add_argument(
        "--data_path",
        "-dp",
        default="data/nytimes_news_articles_preprocessed.pkl",
        type=str,
        help="Path to directory to processed data file. If relative path, relative to root directory. Default is NYT.",
    )
    parser.add_argument(
        "--models_dir",
        "-md",
        default="models/",
        type=str,
        help="Path to directory to save models in. If relative path, relative to root directory. Default is models/.",
    )

    # number of bootstrap samples
    parser.add_argument(
        "-n",
        default=20,
        type=int,
        help="Number of bootstrap samples to make. Default is 20.",
    )

    # Gensim hyperparameters
    parser.add_argument(
        "--vector_size",
        "-vs",
        default=100,
        type=int,
        help="Dimensionality of embedding vector. Default is 100.",
    )
    parser.add_argument(
        "--window",
        "-w",
        default=5,
        type=int,
        help="Window size. Default is 5.",
    )
    parser.add_argument(
        "--min_count",
        "-mc",
        default=10,
        type=int,
        help="Minimum word count to filter. Default is 10.",
    )
    parser.add_argument(
        "--negative",
        "-neg",
        default=5,
        type=int,
        help="Number of negatives to sample. Default is 5.",
    )
    parser.add_argument(
        "--workers",
        "-ws",
        default=1,
        type=int,
        help="Number of worker threads to train model. Default is 1, because >1 may break reproducibility.",
    )
    parser.add_argument(
        "--seed",
        "-s",
        default=42,
        type=int,
        help="Random seed for reproducibility. Default is 42.",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        default=5,
        type=int,
        help="Number of epochs to train skipgram. Default is 5.",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    data_path = kwargs.pop("data_path")
    models_dir = kwargs.pop("models_dir")
    n = kwargs.pop("n")

    # get root dir and set it as working directory
    fdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(fdir)

    # set numpy seed for bootstrap sample reproducibility
    np.random.seed(kwargs["seed"])

    # train word2vec models
    bootstrap_train(data_path, models_dir, kwargs, n)
