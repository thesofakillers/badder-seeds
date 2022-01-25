import torch
import numpy as np
import pickle, os
import torch.utils.data as data
from tqdm import tqdm


class DataReader:
    def __init__(self, data_path: str, min_count: int = 10):
        """
        Initializes a DataReader object
        :param str data_path: path string to data
        :param int min_count: minimum frequency of words to embed
        """
        self.data_path = data_path
        self.min_count = min_count

        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = {}

        self.vocab_size = 0

        # determine number of lines in data and randomly sample
        self.num_lines = sum(1 for line in open(self.data_path, "rb"))
        self.sampled = np.random.choice(self.num_lines, size=self.num_lines).sort()

        # build frequency list
        self.read_data()

    def read_data(self):
        """
        Reads data according to sampled lines and builds frequency dictionary, id to word mappings.
        Filters by word frequency according to minimum count
        :param self: a DataReader object
        """

        word_counts = {}
        idx_sampled = 0

        # NOTE: problem with zero length lines, counted as sampled but not actually add to data.
        # do empty lines count in line count of preprocessed?
        print("Reading words from " + self.data_path)
        with open(self.data_path, "rb") as f:
            for i, line in enumerate(tqdm(f, unit="line")):
                tokens = line.split()
                if len(line) > 0 and i == self.sampled[idx_sampled]:
                    for t in tokens:
                        if len(t) > 0:
                            word_counts[t] = word_counts.get(t, 0) + 1
                    idx_sampled += 1

        for i, w in enumerate(word_counts.keys()):
            c = word_counts[w]
            if c >= self.min_count:
                self.word2idx[w] = i
                self.idx2word[i] = w
                self.word_freq[i] = c
        print("Total terms to embed " + str(len(self.word_freq)))

        return
