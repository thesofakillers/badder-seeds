import os
import json
import numpy as np
import pickle
from tqdm import tqdm
import random
import re

def containsNumber(value):
    numbers = re.findall('[0-9]+', value)
    return True if numbers else False 

documents = []
document = []
lengths = []
vocab = set()
with open("../data/nytimes_news_articles.txt", "r") as f:
    lines = f.readlines()
for index, line in enumerate(tqdm(lines)):
    if len(line) > 0 and line[:3] != "URL":
        document.append(line)
    if line[:3] == "URL" or index == (len(lines) - 1):
        if len(document) > 0:
            # get rid of linebreaks, lowercase
            document = " ".join(document).replace("\n", "").rstrip().lower().split(' ')[1:]
            document = [re.sub(r'[^\w\s]', '', i) for i in document if not containsNumber(i)]
            lengths.append(len(document))
            for i in document:
                vocab.add(i)
            # save
            documents.append(document)
            # reset for next document
            document = []
    else:
        continue

print("Total Documents", len(documents))
print("Total Words", np.sum(lengths))
print("Vocabulary size", len(vocab))
print("Mean Document Length", np.mean(lengths))

with open("../data/WikiText103/wikitext-103/wiki.train.tokens", "r") as f:
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
    .rstrip()
    .lower()
    .split(' ')
    for start_idx, end_idx in zip(doc_idxs, doc_idxs[1:])
]

for index, document in enumerate(tqdm(documents)):
    for index1, word in enumerate(document):
        if containsNumber(word):
            del documents[index][index1]
            continue
        documents[index][index1] = re.sub(r'[^\w\s]', '', word)
# vocab = set()
lengths = [len(i) for i in documents]
# [vocab.add(re.sub(r'[^\w\s]', '', i)) for j in documents for i in j if not containsNumber(i)]

print("Total Documents", len(documents))
print("Total Words", np.sum(lengths))
# print("Vocabulary size", len(vocab))
print("Mean Document Length", np.mean(lengths))