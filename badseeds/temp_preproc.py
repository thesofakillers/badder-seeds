import argparse
from tqdm import tqdm
import spacy
import pickle

# tokenizer and tagger
nlp = spacy.load(
    "en_core_web_sm", disable=["ner", "lemmatizer", "parser", "attribute_ruler"]
)


def preprocess_nyt(raw_data_path):
    """
    Tokenise, lowercase and POS tag NYT data, document-wise
    returns list of Spacy Docs
    https://spacy.io/api/doc/

    A spacy doc is a list of spacy tokens.
    To access POS tag of a given token, use token.tag_
    If you want to have more coarse POS tags, need to enable 'attribute_ruler'
    Resulting tags will be available from token.pos_
    """
    documents = []
    document = []
    with open(raw_data_path, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(tqdm(lines)):
        if len(line) > 0 and line[:3] != "URL":
            document.append(line)
        if line[:3] == "URL" or i == (len(lines) - 1):
            if len(document) > 0:
                # get rid of linebreaks, lowercase
                document = " ".join(document).lower().replace("\n", "")
                # tokenize and pos tag with spacy
                document = nlp(document)
                # save
                documents.append(document)
                # reset for next document
                document = []
        else:
            continue

    return documents


if __name__ == "__main__":
    # i quickly drafted this for demo reasons, you may want to change it so
    # that it only runs if it hasnt been run already
    # that being said you only need to run this once, since it pickles it

    # takes about 5 minutes to run on my 2017 macbook pro
    parser = argparse.ArgumentParser(description="Preprocess NYT data")

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="path to raw data",
        default="data/nytimes_news_articles.txt",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="path to output file",
        default="data/nytimes_news_articles_preprocessed.pkl",
    )

    args = parser.parse_args()
    nyt_articles = preprocess_nyt(args.input)

    # save to pickle, its a very thicc file though, 4.5 GB, maybe not good
    print("done, saving to pickle")
    with open(args.output, "wb") as f:
        pickle.dump(nyt_articles, f)
