import argparse
import gensim.models as gm
import utils
from tqdm import tqdm
import os
import numpy as np


def train_word2vec(data: list, params: dict) -> gm.keyedvectors.KeyedVectors:
    """Trains word2vec gensim model on fed sentence data

    Parameters
    ----------
    data : list
        list of sentences to feed gensim model
    params : dict
        parameters of the gensim function

    Returns
    -------
    word_vectors : gm.keyedvectors.KeyedVectors
        embeddings for gensim model keyed by word
    """

    model = gm.Word2Vec(sentences=data, **params)
    return model.wv


def bootstrap_train(
    data_path: str, models_dir: str, params: dict, seed: int = 42, n: int = 20
) -> None:
    """Trains word2vec model through bootstrapping n times and saves.

    Parameters
    ----------
    data_path : str
        path to the data to train on
    n : int
        number of times to bootstrap. Default is 20.
    models_dir : str
        directory to save model data
    params : dict
        parameters of word2vec model
    seed : int
        seed for random number generator. Default is 42
    """

    # set numpy seed for bootstrap sample reproducibility
    np.random.seed(seed)

    # make model dirs
    if os.path.isdir(data_path):
        name_file = data_path.split("/")[-1] + "_min" + str(params["min_count"])
    else:
        name_file = (
            os.path.splitext(data_path)[0].split("/")[-1]
            + "_min"
            + str(params["min_count"])
        )
    save_path = os.path.join(models_dir, name_file)
    os.makedirs(save_path, exist_ok=True)

    samples = utils.bootstrap(data_path, n)
    for i, s in enumerate(samples):
        print(f"Sample {i + 1}, building dataset:")
        pos_vocab: dict = {}
        sample_text = []
        for doc in tqdm(s, unit="documents"):
            doc_text = []
            for token in doc:
                doc_text.append(token.text)
                if token.text in pos_vocab.keys():
                    if token.tag_ not in pos_vocab[token.text]:
                        pos_vocab[token.text].append(token.tag_)
                else:
                    pos_vocab[token.text] = [token.tag_]
            sample_text.append(doc_text)

        print("\nTraining word2vec...")
        model = train_word2vec(sample_text, params)

        print("\nAttaching pos tags to word vectors:")
        for word, tag in tqdm(pos_vocab.items(), unit="vectors"):
            if word in model.key_to_index:
                model.set_vecattr(word, "pos", tag)

        # save embeddings
        # do not save if already exists
        file_name = "vectors_sample" + str(i + 1) + ".kv"
        file_path = os.path.join(save_path, file_name)
        if os.path.exists(file_path):
            print(
                f"{file_path} already exists, skipping. "
                "Delete this file to save re-trained vectors."
            )
        else:
            model.save(file_path)
        print(f"\nSaved in {save_path}.")
        del model
        del sample_text
        del pos_vocab

    return


if __name__ == "__main__":

    # Command line arguments
    parser = argparse.ArgumentParser()

    # directory to save model in and data file path
    parser.add_argument(
        "--models_dir",
        "-md",
        default="models/",
        type=str,
        help="Path to directory to save models in. "
        "If relative path, relative to root directory. Default is models/.",
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
        help="Number of worker threads to train model. "
        "Default is 1, because >1 may break reproducibility.",
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
    seed = kwargs.pop("seed")

    # get root dir and set it as working directory
    fdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(fdir)

    # train word2vec models
    bootstrap_train(
        "data/processed/nytimes_news_articles.bin", models_dir, kwargs, seed, n
    )
    bootstrap_train("data/processed/wiki.train.tokens.bin", models_dir, kwargs, seed, n)
    bootstrap_train("data/processed/history_biography", models_dir, kwargs, seed, n)
    bootstrap_train("data/processed/romance", models_dir, kwargs, seed, n)
