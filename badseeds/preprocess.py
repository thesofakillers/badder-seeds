import itertools
import argparse
from tqdm import tqdm
import spacy
import pickle
import os
import json
from spacy.tokens import DocBin
import re
from collections import Counter, defaultdict
import random

# tokenizer and tagger
nlp = spacy.load(
    "en_core_web_sm", disable=["ner", "lemmatizer", "parser", "attribute_ruler"]
)


def read_pproc_dataset(path):
    """
    Reads binary file(s) from a path, returning a list of Spacy DocBins
    """
    if os.path.isdir(path):
        print("Directory detected, reading and concatenating all containing files")
        doc_bin = []
        for file in tqdm(os.listdir(path)):
            with open(os.path.join(path, file), "rb") as f:
                bytes_data = f.read()
            doc_bin.append(byte_data_to_docbin(bytes_data))
    else:
        with open(path, "rb") as f:
            bytes_data = f.read()
        doc_bin = byte_data_to_docbin(bytes_data)
    return doc_bin


def docbin_to_docs(doc_bin):
    """
    converts a Spacy DocBin to a list of Spacy Docs
    """
    if type(doc_bin) == list:
        return itertools.chain(*[dbin.get_docs(nlp.vocab) for dbin in doc_bin])
    else:
        return doc_bin.get_docs(nlp.vocab)


def byte_data_to_docbin(bytes_data):
    """
    Converts bytes data to a Spacy DocBin
    """
    doc_bin = DocBin(attrs=["TAG", "IS_ALPHA", "IS_DIGIT"]).from_bytes(bytes_data)
    return doc_bin


def run_spacy_pipe(documents):
    """
    Runs spacy pipe (tokenization and POS tagging)
    on a list of strings,
    returns only relevant bytes data
    """
    doc_bin = DocBin(attrs=["TAG"])
    for doc in tqdm(
        nlp.pipe(documents, disable=["ner", "lemmatizer", "parser", "attribute_ruler"]),
        total=len(documents),
    ):
        doc_bin.add(doc)
    bytes_data = doc_bin.to_bytes()
    return bytes_data


def spacy_and_save(documents, path):
    """
    passes preprocessed list through spacy pipe for tokenization
    and POS tagging, then saves efficiently to binary file
    """
    bytes_data = run_spacy_pipe(documents)
    print(path)
    with open(path, "wb") as f:
        f.write(bytes_data)


def preprocess_nyt(
    path_to_file="../data/nytimes_news_articles.txt",
    save_path="../data/processed/nytimes_news_articles.bin",
    serialize=True,
    return_memory=False,
):
    """
    Preprocesses the NYT dataset
    saving a list of tokenized and tagged articles to disk
    """

    if os.path.isfile(save_path):
        print("Skipping NYT because it is already processed")
        if return_memory:
            return read_pproc_dataset(save_path)
    else:
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
                    # save
                    documents.append(document)
                    # reset for next document
                    document = []
            else:
                continue
        print("Preliminary preprocessing complete; continuing with spacy")
        bytes_data = run_spacy_pipe(documents)
        if serialize:
            with open(save_path, "wb") as f:
                f.write(bytes_data)
        if return_memory:
            return byte_data_to_docbin(bytes_data)


def preprocess_goodreads(name, path_to_dir="../data/", save_dir="./data/preprocessed_data/", save_path=None):
    """
    Preprocesses Goodreads datasets

    Parameters
    ----------
    name : str
        name of the dataset to preprocess, either "romance" or "history_biography"
    path_to_dir : str, optional
        path to the directory containing the dataset, by default "../data/"
    save_dir : str, optional
        path to the directory to save the preprocessed dataset, by default "../data/processed/"
    """
    if os.path.isdir(os.path.join(save_dir, name)):
        print(f"Skipping Goodreads {name} because it is already processed")
    else:
        if not os.path.isdir(os.path.join(save_dir, name)):
            os.makedirs(os.path.join(save_dir, name))
        path_to_file = path_to_dir + ".json"
        save_path = save_dir + name

        with open(path_to_file, "r") as f:
            lines = f.readlines()

        # sample 500 reviews per book
        print("Removing all reviews with less than 20 chars and creating datastructure")
        first_dict = defaultdict()
        for line in tqdm(lines):
            data = json.loads(line)
            if len(data["review_text"]) > 20:

                if data["book_id"] not in first_dict:
                    first_dict[data["book_id"]] = []

                first_dict[data["book_id"]].append(data["review_id"])

        # remove books with fewer than 500 reviews
        print("Remove all books with fewer than 500 reviews")
        second_dict = defaultdict()
        for i in tqdm(first_dict):
            if len(first_dict[i]) >= 500:
                second_dict[i] = first_dict[i]

        # sample 500 random review ids per book
        print("Sampling 500 random reviews per book")
        for book in tqdm(second_dict):
            second_dict[book] = random.sample(second_dict[book], 500)

        print("Preprocessing all reviews now")
        documents = []
        for index, line in enumerate(tqdm(lines)):
            data = json.loads(line)
            if (
                data["book_id"] in second_dict
                and data["review_id"] in second_dict[data["book_id"]]
            ):
                document = data["review_text"]
                document = document.lower().replace("\n", "")
                documents.append(document)

            if index != 0 and index % 500000 == 0 or index == len(lines) - 1:
                current_save_path = save_path + f"/goodreads_reviews_{name}_{index}.bin"
                spacy_and_save(documents, current_save_path)
                documents = []


def preprocess_wiki(
    path_to_file="../data/WikiText103/WikiText103/wikitext-103/wiki.train.tokens",
    save_path="../data/processed/wiki.train.tokens.bin",
    serialize=True,
    return_memory=False,
):
    """Preprocesses the WikiText dataset"""
    if os.path.isfile(save_path):
        print("Skipping WikiText103 because it is already processed")
    else:
        with open(path_to_file, "r") as f:
            lines = f.readlines()
        # remove lines with formulas
        lines = [line for line in lines if line != " <formula> \n"]
        # determine document start indexes using combination of regex patterns
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
            "".join(lines[start_idx:end_idx])
            .replace("\n", "")
            .replace("<formula>", "")
            .strip()
            .lower()
            for start_idx, end_idx in zip(doc_idxs, doc_idxs[1:])
        ]

        print("Preliminary preprocessing complete; continuing with spacy")
        bytes_data = run_spacy_pipe(documents)
        if serialize:
            with open(save_path, "wb") as f:
                f.write(bytes_data)
        if return_memory:
            return byte_data_to_docbin(bytes_data)


def preprocess_datasets(config_dict):
    if not os.path.isdir(config_dict['preprocessed']['dir_path']):
        os.makedirs(config_dict['preprocessed']['dir_path'])
    """preprocesses all the datasets"""
    preprocess_nyt(
        os.path.join(config_dict["raw"]["dir_path"], config_dict["raw"]["nyt_subpath"])
        + ".txt",
        save_path=os.path.join(
            config_dict["preprocessed"]["dir_path"],
            config_dict["raw"]["nyt_subpath"] + ".bin",
        ),
    )
    preprocess_wiki(
        os.path.join(
            config_dict["raw"]["dir_path"],
            config_dict["raw"]["wiki_subpath"],
            "wikitext-103",
            "wiki.train.tokens",
        ),
        save_path=os.path.join(
            config_dict["preprocessed"]["dir_path"],
            "wiki.train.tokens" + ".bin",
        ),
    )
    preprocess_goodreads(
        "romance",
        os.path.join(
            config_dict["raw"]["dir_path"], config_dict["raw"]["goodreads_r_subpath"]
        ),
        save_path=os.path.join(
            config_dict["preprocessed"]["dir_path"],
            "goodreads_reviews_romance",
        ),
    )
    preprocess_goodreads(
        "history_biography",
        os.path.join(
            config_dict["raw"]["dir_path"], config_dict["raw"]["goodreads_hb_subpath"]
        ),
        save_path=os.path.join(
            config_dict["preprocessed"]["dir_path"],
            "goodreads_reviews_history_biography",
        ),
    )


def read_pproc_datasets(nyt_path, wiki_path, grr_dir_path, grhb_dir_path):
    """
    reads all the preprocessed datasets from disk, returning a dictionary
    if a given path is not passed, the corresponding dataset is not read
    some datasets take a while to read, so this may be desired
    """
    pproc_datasets = {
        "NYT": None,
        "WikiText": None,
        "Goodreads (Romance)": None,
        "Goodreads (History/Biography)": None,
    }
    if nyt_path:
        print("reading nyt")
        pproc_datasets["NYT"] = read_pproc_dataset(nyt_path)
    if wiki_path:
        print("reading wikitext")
        pproc_datasets["WikiText"] = read_pproc_dataset(wiki_path)
    if grr_dir_path:
        print("reading goodreads romance")
        pproc_datasets["Goodreads (Romance)"] = read_pproc_dataset(grr_dir_path)
    if grhb_dir_path:
        print("reading goodreads history/biography")
        pproc_datasets["Goodreads (History/Biography)"] = read_pproc_dataset(
            grhb_dir_path
        )
    return pproc_datasets

