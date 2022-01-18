import os
import numpy as np
import nltk
import pickle
import json
from tqdm import tqdm


def process_sentence(sentence):
    return nltk.tokenize.word_tokenize(sentence.lower())


try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

if __name__ == "__main__":
    if not os.path.isdir("../data/processed"):
        os.makedirs("../data/processed")
        print("Created folder : ", "../data/processed")

    with open("../data/nytimes_news_articles.txt", "r") as f:
        all_sentences = []
        for line in tqdm(f.readlines()):
            if len(line) > 1 and line[:3] != "URL":
                [all_sentences.append(process_sentence(i)) for i in line.split(".")]
        all_sentences = [x for x in all_sentences if len(x) > 1]
        pickle.dump(
            all_sentences, open("../data/processed/nytimes_news_articles.pkl", "wb")
        )

    with open("../data/goodreads_reviews_dedup.json", "r") as f:
        all_sentences = []
        for line in tqdm(f.readlines()):
            data = json.loads(line)
            text = data["review_text"]
            text = [process_sentence(i) for i in text.split(".")]
            [all_sentences.append(i) for i in text]
        pickle.dump(
            all_sentences, open("../data/processed/goodreads_reviews_dedup.pkl", "wb")
        )

    for file in os.listdir("../data/WikiText103/wikitext-103"):
        filename = os.fsdecode(file)
        with open("../data/WikiText103/wikitext-103/" + filename, "r") as f:
            all_sentences = []
            for line in tqdm(f.readlines()):
                if len(line) > 1 and line[:2] != " =":
                    [all_sentences.append(process_sentence(i)) for i in line.split(".")]

            all_sentences = [x for x in all_sentences if len(x) > 1]
            pickle.dump(all_sentences, open(f"../data/processed/{filename}.pkl", "wb"))
