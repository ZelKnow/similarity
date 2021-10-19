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
import time
from util.dataloader import load_data
from multiprocessing import Pool


def df2idf(docfreq, totaldocs, log_base=2.0):
    """Compute inverse document frequency(idf) for term t.

    Args:
        docfreq (int): number of documents where the term t appears
        totaldocs (int): total number of documents
        log_base (float, optional): Defaults to 2.0.

    Returns:
        int: idf result
    """
    return math.log(float(totaldocs) / docfreq) / math.log(log_base)


def compute_dot_product(x, y):
    """Compute the sum of the products of the values with the same key.

    Args:
        x (dict): input dict
        y (dict): input dict

    Returns:
        dict: dot product
    """
    intersection = x.keys() & y.keys()
    return sum([x[k] * y[k] for k in intersection])


def norm(x):
    """Compute the norm of the values array of dict x.

    Args:
        x (dict)

    Returns:
        float: norm result
    """
    return math.sqrt(sum([v * v for v in x.values()]))


class VectorSpaceModel:
    def __init__(self, path_to_data, path_to_stopwords=None):
        """Initialize the Vector Space Model

        Args:
            path_to_data (str): path to the data
            path_to_stopwords (str, optional): path to the stopwords. Defaults to None.
        """
        self.bag_of_words = []  # 词袋
        self.token2id = {}
        self.idfs = {}
        self.tfs = []
        self.docs = load_data(path_to_data, path_to_stopwords)  # 读取数据
        self.num_docs = len(self.docs)
        self.similarity = [[0.0 for i in range(self.num_docs)]
                           for j in range(self.num_docs)]  # 初始化相似度矩阵
        self.initialize_bow()  # 初试化词袋
        self.compute_tfs()  # 计算tf
        self.compute_idfs()  # 计算idf
        self.compute_tfidfs()  # 计算tf-idf

    def initialize_bow(self):
        """Generate the bag of words for given data
        """
        for doc in self.docs:
            counter = defaultdict(int)
            for word in doc:
                if word not in self.token2id:
                    self.token2id[word] = len(self.token2id)  # 为词进行编号
                counter[self.token2id[word]] += 1
            self.bag_of_words.append(counter)

    def compute_idfs(self):
        """Compute idfs for given data
        """
        dfs = {}
        for bow in self.bag_of_words:
            for termid in bow:
                dfs[termid] = dfs.get(termid, 0) + 1
        self.idfs = {
            termid: df2idf(df, self.num_docs)
            for termid, df in dfs.items()
        }

    def compute_tfs(self):
        """Compute tfs for given data
        """
        for bow in self.bag_of_words:
            total_fs = sum(bow.values())
            self.tfs.append(
                {termid: fs / total_fs
                 for termid, fs in bow.items()})

    def compute_tfidfs(self):
        """Compute tf-idf
        """
        self.tfidfs = [{
            termid: tf * self.idfs.get(termid)
            for termid, tf in adoc_tfs.items()
        } for adoc_tfs in self.tfs]

    def compute_similarity_naive(self):
        """Naive implement of the cosine similarity
        """
        for docno_1, idfs_1 in enumerate(self.tfidfs):
            for docno_2, idfs_2 in enumerate(self.tfidfs):
                dot_product = compute_dot_product(idfs_1, idfs_2)  # 分子
                self.similarity[docno_1][docno_2] = dot_product / (
                    norm(idfs_1) * norm(idfs_2))

    def compute_similarity_subprocess(self, processes=1, i=0):
        """Subprocess function for computing the cosine similarity

        Args:
            processes (int, optional): processes number. Defaults to 1.
            i (int, optional): process id. Defaults to 0.
        """
        similarity = []
        length = math.ceil(self.num_docs / processes)
        for idfs_1 in self.tfidfs[length * i:length * (i + 1)]:
            similarity.append([])
            for idfs_2 in self.tfidfs:
                dot_product = compute_dot_product(idfs_1, idfs_2)  # 分子
                similarity[-1].append(dot_product /
                                      (norm(idfs_1) * norm(idfs_2)))
        return similarity

    def compute_similarity_multiprocess(self, processes):
        """Multiprocess implement of the cosine similarity

        Args:
            processes (int): processes number.
        """
        res = []
        with Pool(processes) as pool:
            for i in range(processes):
                res.append(
                    pool.apply_async(self.compute_similarity_subprocess,
                                     (processes, i)))
            pool.close()
            pool.join()
        similarity = []
        for i in range(processes):
            similarity += res[i].get()
        self.similarity = similarity


if __name__ == "__main__":
    vsm = VectorSpaceModel("data/199801_clear.txt", "data/stopwords.txt")

    t = time.time()
    vsm.compute_similarity_naive()  # 单进程计算
    print("单进程计算相似度花费时间：{}s.".format(time.time() - t))

    # 记录结果并将相似度矩阵重置
    res_multiprocess = [[
        vsm.similarity[j][i] for i in range(len(vsm.similarity[j]))
    ] for j in range(len(vsm.similarity))]
    vsm.similarity = [[0.0 for i in range(len(res_multiprocess[j]))]
                      for j in range(len(res_multiprocess))]

    t = time.time()
    vsm.compute_similarity_multiprocess(4)  # 多线程计算相似度
    print("多进程计算相似度花费时间：{}s.".format(time.time() - t))

    print("单进程与多进程计算结果是否相同：{}".format(res_multiprocess == vsm.similarity))
