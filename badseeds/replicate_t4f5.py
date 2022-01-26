import utils, os
import numpy as np
import gensim.models as gm

if __name__ == "__main__":
    # NOTE: still missing Maria's response to question
    import argparse

    # get root dir and set it as working directory
    fdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(fdir)

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings_dir",
        "-d",
        default="models/nytimes_news_articles_min10",
        type=str,
        help="Path to directory where embeddings are stored. If relative path, relative to root directory. Default is NYT dataset embeddings.",
    )
    args = parser.parse_args()

    # load embeddings
    models = []
    for file in os.listdir(args.embeddings_dir):
        if file.endswith(".kv"):
            models.append(gm.KeyedVectors.load(os.path.join(args.embeddings_dir, file)))

    # part 1: gathered seeds
    # load in all gathered seeds to memory

    # calculate coherence for each seed (normalize)

    # pair and rank

    # part 2: random seeds

    # generate random seeds

    # coherence for each seed (normalize)

    # pair and rank

    # save to file
