#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File          : vsm.py
@Description   : 实现向量空间模型
@Date          : 2021/10/09 21:33:47
@Author        : ZelKnow
@Github        : https://github.com/ZelKnow
"""
__author__ = "ZelKnow"
from collections import defaultdict
import math
from util.dataloader import load_data


def df2idf(docfreq, totaldocs, log_base=2.0, add=0.0):
    return add + math.log(float(totaldocs) / docfreq) / math.log(log_base)


def bit_product_sum(x, y):
    intersection = x.keys() & y.keys()
    return sum([x[k] * y[k] for k in intersection])


def norm(x):
    return math.sqrt(sum([v * v for v in x.values()]))


class VectorSpaceModel:
    def __init__(self, path_to_data, path_to_stopwords=None):
        self.bag_of_words = []
        self.token2id = {}
        self.idfs = {}
        self.tfs = []
        self.docs = load_data(path_to_data, path_to_stopwords)
        self.num_docs = len(self.docs)
        self.initialize_bow()
        self.compute_tfs()
        self.compute_idfs()
        self.compute_tfidfs()

    def initialize_bow(self):
        for doc in self.docs:
            counter = defaultdict(int)
            for word in doc:
                if word not in self.token2id:
                    self.token2id[word] = len(self.token2id)
                counter[self.token2id[word]] += 1
            self.bag_of_words.append(counter)

    def compute_idfs(self):
        dfs = {}
        for bow in self.bag_of_words:
            for termid in bow:
                dfs[termid] = dfs.get(termid, 0) + 1
        self.idfs = {
            termid: df2idf(df, self.num_docs)
            for termid, df in dfs.items()
        }

    def compute_tfs(self):
        for bow in self.bag_of_words:
            total_fs = sum(bow.values())
            self.tfs.append(
                {termid: fs / total_fs
                 for termid, fs in bow.items()})

    def compute_tfidfs(self):
        self.tfidfs = [{
            termid: tf * self.idfs.get(termid)
            for termid, tf in adoc_tfs.items()
        } for adoc_tfs in self.tfs]

    def compute_similarity_naive(self):
        similarity = [[0.0 for i in range(self.num_docs)]
                      for j in range(self.num_docs)]
        for docno_1, idfs_1 in enumerate(self.tfidfs):
            for docno_2, idfs_2 in enumerate(self.tfidfs):
                if docno_2 < docno_1:
                    similarity[docno_1][docno_2] = similarity[docno_2][docno_1]
                else:
                    dot_product = bit_product_sum(idfs_1, idfs_2)
                    similarity[docno_1][docno_2] = dot_product / (
                        norm(idfs_1) * norm(idfs_2))
        self.similarity = similarity


if __name__ == "__main__":
    vsm = VectorSpaceModel("data/199801_clear.txt", "data/stopwords.txt")
    vsm.compute_similarity_naive()
