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
    f = open(path)
    return f.read().splitlines()


def load_data(path_to_data, path_to_stopwords=None):
    docs = []
    cur_doc = []
    cur_doc_name = ""
    stopwords = read_file(
        path_to_stopwords) if path_to_stopwords is not None else []
    for line in read_file(path_to_data):
        if len(line) < 21:
            continue
        line_splited = line.split()
        doc_name = line_splited[0][:15]
        if doc_name != cur_doc_name:
            if len(cur_doc) != 0:
                docs.append(cur_doc)
            cur_doc = []
            cur_doc_name = doc_name
        cur_line = [x[:x.rfind('/')] for x in line_splited[1:]]
        if stopwords:
            cur_line = [x for x in cur_line if x not in stopwords]
        cur_doc += cur_line
    if len(cur_doc) != 0:
        docs.append(cur_doc)
    return docs
