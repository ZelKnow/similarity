#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File          : dataloader.py
@Description   : 读取数据
@Date          : 2021/10/09 12:03:35
@Author        : ZelKnow
@Github        : https://github.com/ZelKnow
"""
__author__ = "ZelKnow"


def read_file(path):
    """Read file.

    Args:
        path (str): path to the file

    Returns:
        list: list of lines
    """
    f = open(path)
    return f.read().splitlines()


def load_data(path_to_data, path_to_stopwords=None):
    """load data

    Args:
        path_to_data (str): path to the data
        path_to_stopwords (str, optional): path to the stopwords. Defaults to None.

    Returns:
        list: each element is a list of words of a doc.
    """
    docs = []
    cur_doc = []
    cur_doc_name = ""
    stopwords = read_file(
        path_to_stopwords) if path_to_stopwords is not None else []
    for line in read_file(path_to_data):
        if len(line) < 21:  # 空行
            continue
        line_splited = line.split()
        doc_name = line_splited[0][:15]  # 提取出文章编号
        if doc_name != cur_doc_name:  # 如果文章编号与当前处理的文章不同，则将当前文章放入结果中
            if len(cur_doc) != 0:
                docs.append(cur_doc)
            cur_doc = []  # 重新处理下一个文章
            cur_doc_name = doc_name
        cur_line = [x[:x.rfind('/')]
                    for x in line_splited[1:]]  # 去掉末尾的词性以及第一个词
        if stopwords:
            cur_line = [x for x in cur_line if x not in stopwords]  # 去除停用词
        cur_doc += cur_line
    if len(cur_doc) != 0:
        docs.append(cur_doc)
    return docs
