import argparse
from tqdm import tqdm
import spacy
import pickle
import os
import json

# tokenizer and tagger
nlp = spacy.load(
    "en_core_web_sm", disable=["ner", "lemmatizer", "parser", "attribute_ruler"]
)


def preprocess_nyt(path_to_file, path_to_pickle):
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
    with open(path_to_file, "r") as f:
        lines = f.readlines()
    for index, line in enumerate(tqdm(lines)):
        if len(line) > 0 and line[:3] != "URL":
            document.append(line)
        if line[:3] == "URL" or index == (len(lines) - 1):
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

    # save to pickle, its a very thicc file though, 4.5 GB, maybe not good
    print("Done, saving to pickle")
    with open(path_to_pickle, "wb") as f:
        pickle.dump(documents, f)

def preprocess_goodreads_romance(path_to_file, path_to_pickle):
    if not os.path.isdir("../data/processed/romance"):
        os.makedirs('../data/processed/romance')
    documents = []
    with open(path_to_file, 'r') as f:
        lines = f.readlines()
    for index, line in enumerate(tqdm(lines)):
        data = json.loads(line)
        document = data["review_text"]
        document = document.lower().replace("\n", "")
        document = nlp(document)
        documents.append(document)

        if index != 0 and index % 100000 == 0:
            path_to_pickle = '../data/processed/romance/' + name + '.pkl' f'_{index}.pkl'
            with open(path_to_file, 'wb') as f:
                pickle.dump(documents, f)
            documents = []

def create_documents(function, name):
    path_to_file = "../data/" + name
    name = name.split('.')[0]
    path_to_pickle = '../data/processed/' + name + '.pkl'

    if not os.path.isfile(path_to_pickle):
        print(f"Processing {name}")
        function(path_to_file, path_to_pickle)
    else:
        print(f"Skipping... File {name} already exists")


if __name__ == "__main__":
    create_documents(preprocess_nyt, "nytimes_news_articles.txt")
    create_documents(preprocess_goodreads_romance, "goodreads_reviews_romance.json")
