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


def preprocess_nyt():
    """
    Tokenise, lowercase and POS tag NYT data, document-wise
    returns list of Spacy Docs
    https://spacy.io/api/doc/

    A spacy doc is a list of spacy tokens.
    To access POS tag of a given token, use token.tag_
    If you want to have more coarse POS tags, need to enable 'attribute_ruler'
    Resulting tags will be available from token.pos_
    """
    path_to_file = "../data/nytimes_news_articles.txt"
    save_path = "../data/processed/nytimes_news_articles.bin"

    if os.path.isfile(save_path):
        print("Skipping NYT because it is already processed")
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


def preprocess_goodreads(name):
    if os.path.isdir(f"../data/processed/{name}"):
        print(f"Skipping Goodreads {name} because it is already processed")
    else:
        if name == 'romance':
            if not os.path.isdir("../data/processed/romance"):
                os.makedirs("../data/processed/romance")
            path_to_file = "../data/goodreads_reviews_romance.json"
            save_path = "../data/processed/romance/goodreads_reviews_romance"
        else:
            if not os.path.isdir("../data/processed/history_biography"):
                os.makedirs("../data/processed/history_biography")
            path_to_file = '../data/goodreads_reviews_history_biography.json'
            save_path = "../data/processed/history_biography/goodreads_reviews_history_biography"

        with open(path_to_file, "r") as f:
            lines = f.readlines()

        #sample 500 reviews per book
        print("Removing all reviews with less than 20 chars and creating datastructure")
        first_dict = defaultdict()
        for line in tqdm(lines):
            data = json.loads(line)
            if len(data['review_text']) > 20:
                
                if data['book_id'] not in first_dict:
                    first_dict[data['book_id']] = []

                first_dict[data['book_id']].append(data['review_id'])

        #remove books with fewer than 500 reviews
        print("Remove all books with fewer than 500 reviews")
        second_dict = defaultdict()
        for i in tqdm(first_dict):
            if len(first_dict[i]) >= 500:
                second_dict[i] = first_dict[i]
        
        #sample 500 random review ids per book
        print("Sampling 500 random reviews per book")
        for book in tqdm(second_dict):
            second_dict[book] = random.sample(second_dict[book], 500)

        print("Preprocessing all reviews now")
        documents = []
        for index, line in enumerate(tqdm(lines)):
            data = json.loads(line)
            if data['book_id'] in second_dict and data['review_id'] in second_dict[data['book_id']]:
                document = data["review_text"]
                document = document.lower().replace("\n", "")
                document = nlp(document)
                documents.append(document)

            if index != 0 and index % 500000 == 0 or index == len(lines) - 1:
                    current_save_path = save_path + f"_{index}.bin"
                    save_file(documents, current_save_path)
                    documents = []


def preprocess_wiki():
    path_to_file = "../data/WikiText103/wikitext-103/wiki.train.tokens"
    save_path = '../data/processed/wiki.train.tokens.bin'
    if os.path.isfile(save_path):
        print("Skipping WikiText103 because it is already processed")
    else:
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

if __name__ == "__main__":
    preprocess_nyt()
    preprocess_wiki()
    preprocess_goodreads('romance')
    preprocess_goodreads('history_biography')
