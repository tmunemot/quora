#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" extract features from question texts """
import sys
import os
import csv
import traceback
import warnings
import argparse

import tqdm
import scipy.stats
import sklearn.metrics.pairwise
import numpy as np
import pandas as pd
import multiprocessing
import ctypes

from collections import OrderedDict
from nltk import ngrams
from nltk.corpus import stopwords

import utils

eng_stopwords = set(stopwords.words("english"))
num_instances = -1
max_sequence_length = -1
num_words = -1
embedding_dim = -1

def ngram_val(s1, s2):
    """
        calculate a number of common elements in two sets and its ratio to an union set
        
        Args:
        s1: set 1
        s2: set 2

        Returns:
        None
    """
    count = float(len(s1 & s2))
    ratio = count / np.max([len(s1 | s2), 1])
    return count, ratio


def extract_ngram(q1, q2, row):
    """
        extract n-gram features
        
        Args:
        q1: question 1
        q2: question 2
        row: dictionary for storing outputs
        
        Returns:
        None
    """
    row["base_wlen_diff"] = np.abs(len(q1) - len(q2))
	
    # unigram
    s1 = set(q1)
    s2 = set(q2)
    row["base_unigram_count"], row["base_unigram_ratio"] = ngram_val(s1, s2)
	
	# bigram
    s1 = set([v for v in ngrams(q1, 2)])
    s2 = set([v for v in ngrams(q2, 2)])
    row["base_bigram_count"], row["base_bigram_ratio"] = ngram_val(s1, s2)
		
	# trigram
    s1 = set([v for v in ngrams(q1, 3)])
    s2 = set([v for v in ngrams(q2, 3)])
    row["base_trigram_count"], row["base_trigram_ratio"] = ngram_val(s1, s2)

	
def basic_stats(x, row, prefix):
    """
        extract statistics from an array
        
        Args:
        x: samples
        row: dictionary for storing outputs
        prefix: feature group name prepended to statistics value names
        
        Returns:
        None
    """
    N = len(x)
    for field in ["mean", "std", "skew", "kurtosis", "min", "max", "range", "q1", "median", "q3", "iqrange", "mad"]:
        row[prefix + "_" + field] = np.nan

	if N > 0:
		row[prefix + "_mean"] = np.mean(x)
		row[prefix + "_median"] = np.median(x)
		row[prefix + "_min"] = np.min(x)
		row[prefix + "_max"] = np.max(x)

	if N > 2:
		row[prefix + "_std"] = np.std(x, ddof=1)
		row[prefix + "_range"] = row[prefix + "_max"] - row[prefix + "_min"]		
		row[prefix + "_mad"] = np.sum(np.abs(x - row[prefix + "_mean"])) / float(len(x))
		
	if N > 5:
		row[prefix + "_skew"] = scipy.stats.skew(x)
		row[prefix + "_kurtosis"] = scipy.stats.kurtosis(x)
		row[prefix + "_q1"] = np.percentile(x, 25)
		row[prefix + "_q3"] = np.percentile(x, 75)
		row[prefix + "_iqrange"] = row[prefix + "_q3"] - row[prefix + "_q1"]


def extract_sentence_similarity(q1, q2, weights, feature):
    """
        extract similarities from two sets of vectors

        Args:
        q1: question 1
        q2: question 2
        weights: a matrix contains word embedding weights
        feature: dictionary for storing outputs

        Returns:
        None
    """
    v1 = np.array([weights[i] for i in q1])
    v2 = np.array([weights[i] for i in q2])
    cos = sklearn.metrics.pairwise.cosine_similarity(v1, v2)
    basic_stats(cos.reshape((1, -1)), feature, "cosine_full")
    basic_stats(np.max(cos, axis=0), feature, "cosine_col")
    basic_stats(np.max(cos, axis=1), feature, "cosine_row")


def worker(indices):
    """
        worker function for extracting features in parallel
        
        Args:
        indices: list of indices to be processed in a current process
        
        Returns:
        None
    """
    global num_instances, max_sequence_length, num_words, embedding_dim
    question1 = np.frombuffer(shared_question1, dtype=np.int32).reshape((num_instances, max_sequence_length))
    question2 = np.frombuffer(shared_question2, dtype=np.int32).reshape((num_instances, max_sequence_length))
    weights = np.frombuffer(shared_weights, dtype=np.float32).reshape((num_words, embedding_dim))
    features=[]
    for i in indices:
        feature = OrderedDict()
        q1 = question1[i]
        q2 = question2[i]
        extract_ngram(q1, q2, feature)
        extract_sentence_similarity(q1, q2, weights, feature)
        features.append(feature)
    return pd.DataFrame.from_dict(features)


def prepare_shared_array(arr, data_type):
    """
        create a shared array and copy values
        
        Args:
        arr: numpy 2d array
        nprocs: number of processes used for extracting features
        
        Returns:
        None
    """
    shared = multiprocessing.Array(data_type, arr.size, lock=False)
    shared[:] = arr.flatten()
    return shared


def add_features(dataset, weights, nprocs):
    """
        extract features from tokenized questions
        
        Args:
        dataset: a list contains questions
        weights: a matrix contains word embedding weights
        nprocs: number of processes used for extracting features
        
        Returns:
        None
    """
    global num_instances, max_sequence_length, num_words, embedding_dim
    num_instances, max_sequence_length = dataset[0].shape
    num_words, embedding_dim = weights.shape
    
    shared_question1 = prepare_shared_array(dataset[0], ctypes.c_int32)
    shared_question2 = prepare_shared_array(dataset[1], ctypes.c_int32)
    shared_weights = prepare_shared_array(weights, ctypes.c_float)

    df_features = []
    pool = multiprocessing.Pool(processes=nprocs, initializer=init_shared, initargs=(shared_question1, shared_question2, shared_weights))
    with tqdm.tqdm(total=nprocs*10) as pb:
        for i, df_feature in tqdm.tqdm(enumerate(pool.imap(worker, utils.split(range(num_instances), nprocs*10)))):
            df_features.append(df_feature)
            pb.update()
    df = pd.concat(df_features)
    dataset.insert(2, df.values)
    print dataset[0].shape
    print df.shape


def init_shared(question1, question2, weights):
    """
        initialize shared numpy array for multiprocessing
        
        Args:
        question1: tokenized question 1 stored in 2d numpy array
        question2: tokenized question 2 stored in 2d numpy array
        weights: a matrix contains word embedding weights
        
        Returns:
        None
    """
    global shared_question1, shared_question2, shared_weights
    shared_question1 = question1
    shared_question2 = question2
    shared_weights = weights


