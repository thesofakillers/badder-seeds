import argparse
from tqdm import tqdm
import spacy
import pickle
import os
import json
import multiprocessing
from spacy.tokens import DocBin
import re

# split a list into evenly sized chunks
def chunks(l, n):
    return [l[i : i + n] for i in range(0, len(l), n)]


# tokenizer and tagger
nlp = spacy.load(
    "en_core_web_sm", disable=["ner", "lemmatizer", "parser", "attribute_ruler"]
)


def save_file(documents, path):
    doc_bin = DocBin(attrs=["TAG"])

    for doc in tqdm(
        nlp.pipe(documents, disable=["ner", "lemmatizer", "parser", "attribute_ruler"]),
        total=len(documents),
    ):
        doc_bin.add(doc)

    bytes_data = doc_bin.to_bytes()

    with open(path, "wb") as f:
        f.write(bytes_data)


def preprocess_nyt(name, save_path):
    """
    Tokenise, lowercase and POS tag NYT data, document-wise
    returns list of Spacy Docs
    https://spacy.io/api/doc/

    A spacy doc is a list of spacy tokens.
    To access POS tag of a given token, use token.tag_
    If you want to have more coarse POS tags, need to enable 'attribute_ruler'
    Resulting tags will be available from token.pos_
    """
    path_to_file = "../data/" + name + ".txt"

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

    print("Done. Now saving all documents in ", save_path)
    save_file(documents, save_path)


def preprocess_goodreads_romance(name, save_path):
    def do_job(job_id, data_slice):
        documents = []
        for index, line in enumerate(data_slice):
            data = json.loads(line)
            document = data["review_text"]
            document = document.lower().replace("\n", "")
            document = nlp(document)
            documents.append(document)

            if index != 0 and index % 15000 == 0 or index == len(data_slice) - 1:
                current_save_path = save_path + f"_{job_id}_{index}.bin"
                save_file(documents, "".join(current_save_path))
                documents = []

    if not os.path.isdir("../data/processed/romance"):
        os.makedirs("../data/processed/romance")

    path_to_file = "../data/" + name + ".json"
    with open(path_to_file, "r") as f:
        lines = f.readlines()
    save_path = "../data/processed/romance/" + name

    total = len(lines)
    chunk_size = total // 4
    slices = chunks(lines, chunk_size)
    jobs = []

    for i, s in enumerate(slices):
        j = multiprocessing.Process(target=do_job, args=(i, s))
        jobs.append(j)
    for j in jobs:
        j.start()


def preprocess_wiki(name, save_path):
    path_to_file = "../data/WikiText103/wikitext-103/" + name
    with open(path_to_file, "r") as f:
        lines = f.readlines()

    title_regex = re.compile(" = .* = \n")
    subtitle_regex = re.compile(" = = .* = = \n")
    new_line_regex = re.compile(" \n")

    doc_idxs = [
        idx
        for idx, (prev_line, cur_line, next_line) in enumerate(
            zip(lines[:-1], lines[1:], lines[2:]), 1
        )
        if new_line_regex.match(prev_line)
        and new_line_regex.match(next_line)
        and title_regex.match(cur_line)
        and not subtitle_regex.match(cur_line)
    ] + [len(lines) - 1]

    documents = [
        "".join(lines[start_idx:end_idx]).replace("\n", "").strip().lower()
        for start_idx, end_idx in zip(doc_idxs, doc_idxs[1:])
    ]

    print("Done. Now saving all documents in ", save_path)
    save_file(documents, save_path)


def create_documents(function, name):
    save_path = "../data/processed/" + name + ".bin"

    if not os.path.isfile(save_path):
        print(f"Processing all lines in {name}")
        function(name, save_path)
    else:
        print(f"Skipping... File {name} already exists")


if __name__ == "__main__":
    create_documents(preprocess_nyt, "nytimes_news_articles")
    create_documents(preprocess_wiki, "wiki.train.tokens")
    create_documents(preprocess_goodreads_romance, "goodreads_reviews_romance")
