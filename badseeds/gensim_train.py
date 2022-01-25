import gensim.models as gm
from utils import *
from tqdm import tqdm
import os
import numpy as np
from metrics import *


def train_word2vec(data: list, params: dict) -> gm.keyedvectors.KeyedVectors:
    """Trains word2vec gensim model on fed sentence data
    :param list data: list of sentences to feed gensim model
    :param dict params: parameters of the gensim function
    :returns KeyedVectors word_vectors: embeddings for gensim model keyed by word"""

    model = gm.Word2Vec(sentences=data, **params)
    return model.wv


def bootstrap_train(
    data_path: str, models_dir: str, params: dict, n: int = 20
) -> list[gm.keyedvectors.KeyedVectors]:
    """Trains word2vec model through bootstrapping n times and saves.
    :param str data_path: path to the data to train on
    :param int n: number of times to bootstrap
    :param str models_dir: directory to save model data
    :param dict params: parameters of word2vec model
    :returns list embeddings: list of embeddings for each bootstrap, has attribute pos which lists all pos tags of the word.
        Can get list of all the word's POS tags with gensim's get_vecattr method.
    """

    samples = bootstrap(data_path, n)
    print("Building pos dict:")
    vocab_pos_samples = []
    for s in tqdm(samples, unit="bootstrap sample"):
        pos_vocab = {}
        for doc in s:
            for token in doc:
                if token.text in pos_vocab.keys():
                    if not token.tag_ in pos_vocab[token.text]:
                        pos_vocab[token.text].append(token.tag_)
                else:
                    pos_vocab[token.text] = [token.tag_]
        vocab_pos_samples.append(pos_vocab)

    print("Building gensim input:")
    text_samples = [
        [[token.text for token in doc] for doc in s]
        for s in tqdm(samples, unit="bootstrap sample")
    ]
    print("\nTraining word2vec:")
    word_vecs = [
        train_word2vec(i, params) for i in tqdm(text_samples, unit="bootstrap sample")
    ]

    print("\nAttaching pos tags to word vectors:")
    for i, model in enumerate(word_vecs):
        for word, tag in vocab_pos_samples[i].items():
            if word in model.key_to_index:
                model.set_vecattr(word, "pos", tag)

    # make model dirs
    name_file = (
        os.path.splitext(data_path)[0].split("/")[-1]
        + "_min"
        + str(params["min_count"])
    )
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
    models_dir = kwargs.pop("models_dir")
    n = kwargs.pop("n")

    # get root dir and set it as working directory
    fdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(fdir)

    # set numpy seed for bootstrap sample reproducibility
    np.random.seed(kwargs["seed"])

    # train word2vec models
    # bootstrap_train("data/processed/nytimes_news_articles.bin", models_dir, kwargs, n)
