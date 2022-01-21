import os
import json
import numpy as np
import pickle
from tqdm import tqdm

with open("../data/nytimes_news_articles.txt", "r") as f:
    count = 0
    current_review = []
    all_reviews = []
    lengths = []

    for line in tqdm(f.readlines()):
        if len(line) == 1:
            count += 1

        if count == 2:
            if len(current_review) > 0:
                all_reviews.append(current_review)
                length = np.sum([len(i) for i in current_review])
                lengths.append(length)
            count = 0
            current_review = []

        if len(line) > 1 and line[:3] != "URL":
            tokens = line.lower().rstrip().split(" ")
            current_review.append(tokens)

    words = [j for sub in all_reviews for j in sub]
    words = [j for sub in words for j in sub]
    print("Total Documents", len(all_reviews))
    print("Total Words", len(words))
    print("Vocabulary size", len(set(words)))
    print("Mean Document Length", np.mean(lengths))


with open("../data/WikiText103/wikitext-103/wiki.train.tokens", "r") as f:
    all_reviews = []
    current_review = []
    lengths = []

    for line in tqdm(f.readlines()):
        if len(line) > 2:
            if line.count("=") == 2:
                if len(current_review) > 0:
                    all_reviews.append(current_review)
                    length = np.sum([len(i) for i in current_review])
                    lengths.append(length)
                current_review = []
            elif line[:2] == " =":
                continue
            else:
                current_review.append(line.lower().rstrip().split(" "))

    words = [j for sub in all_reviews for j in sub]
    words = [j for sub in words for j in sub]
    print("Total Documents", len(all_reviews))
    print("Total Words", len(words))
    print("Vocabulary size", len(set(words)))
    print("Mean Document Length", np.mean(lengths))
