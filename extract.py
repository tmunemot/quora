#!/usr/bin/python
# -*- coding: utf-8 -*-
""" extract features from question texts """
import sys
import os
import csv
import traceback
import warnings
import argparse
import multiprocessing

import tqdm
import preprocess
import scipy.stats
import scipy.spatial
import sklearn.metrics.pairwise
import numpy as np
import pandas as pd

from collections import defaultdict, OrderedDict
from timeit import default_timer as timer
from gensim.models.keyedvectors import KeyedVectors
from nltk import ngrams
from nltk.corpus import stopwords

PATH_GLOVE_DATA = "./resource/glove.840B.300d.txt"
glove_weights = {}

eng_stopwords = set(stopwords.words("english"))
model_word2vec = None

def parse_args(argv):
	parser = argparse.ArgumentParser(description="Performs feature extraction.")
	parser.add_argument("-n", "--nprocs", help="number of processes", type=int, default=15)
	parser.add_argument("-t", "--test", nargs="+", help="list of paths to test csv files")
	parser.add_argument("train", help="path to a train csv file")
	parser.add_argument("outdir", help="path to an output directory")
	return parser.parse_args(argv)

def worker(args):
    """
        worker funciton for feature extraction
        
        Args:
        args: dictionary contains question strings
        
        Returns:
        feature values stored in an ordered dictionary
    """
	q1 = args["q1"]
	q2 = args["q2"]

	if len(q1) < 2 or len(q2) < 2:
		print "skipped an empty question for {0}".format(args["id"])
		return None

	row = OrderedDict()	
	row["id"] = args["id"]
	
	# n-gram features
	extract_base(q1, q2, row)
    
	# glove features
	try:
		extract_glove(q1, q2, row)
	except:
		return None
	
    # word2vec
	#extract_word2vec(q1, q2, row)
	if args["classlabel"] is not None:
		row["classlabel"] = args["classlabel"]
	
	return row

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
	
def extract_base(q1, q2, row):
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
	
	global eng_stopwords
	
	q1_cl = [v for v in q1 if v not in eng_stopwords]
	q2_cl = [v for v in q2 if v not in eng_stopwords]
	# unigram
	s1 = set(q1_cl)
	s2 = set(q2_cl)
	row["base_ugram_count"], row["base_ugram_ratio"] = ngram_val(s1, s2)
	
	# bigram
	s1 = set([v for v in ngrams(q1_cl, 2)])
	s2 = set([v for v in ngrams(q2_cl, 2)])
	row["base_bgram_count"], row["base_bgram_ratio"] = ngram_val(s1, s2)
		
	# trigram
	s1 = set([v for v in ngrams(q1_cl, 3)])
	s2 = set([v for v in ngrams(q2_cl, 3)])
	row["base_tgram_count"], row["base_tgram_ratio"] = ngram_val(s1, s2)

	
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
	for field in ["mean", "std", "skew", "kurtosis", "min", "max", "range", "q1", "median", "q3", "iqrange", "mad", "coefvar"]:
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
		row[prefix + "_coefvar"] = scipy.stats.variation(x)
		
	if N > 5:
		row[prefix + "_skew"] = scipy.stats.skew(x)
		row[prefix + "_kurtosis"] = scipy.stats.kurtosis(x)
		row[prefix + "_q1"] = np.percentile(x, 25)
		row[prefix + "_q3"] = np.percentile(x, 75)
		row[prefix + "_iqrange"] = row[prefix + "_q3"] - row[prefix + "_q1"]
	
def extract_glove(q1, q2, row):
    """
        extract statistics of cosine similarities calculated over Glove vectors from each question
        
        Args:
        q1: question 1
        q2: question 2
        
        Returns:
        None
    """
	v1 = get_glove_vecs(q1)
	v2 = get_glove_vecs(q2)
	cos = sklearn.metrics.pairwise.cosine_similarity(v1, v2)
	cos = cos.reshape((1, -1))
	basic_stats(cos, row, "glove_cos")

def get_glove_vecs(q):
    """
        convert texts to a list of Glove vectors
        
        Args:
        q: texts
        
        Returns:
        a list of Clove vectors stored in a two dimensional array
    """
	global glove_weights, eng_stopwords
	v = []
	for w in q:	
		if w in glove_weights:
			v.append(glove_weights[w])
		elif (len(w) >= 2) and (w[-1] == 's') and (w[:-1] in glove_weights):
			v.append(glove_weights[w[:-1]])
		else:
			continue
	return np.array(v)		


def init_glove():
    """
        load Glove and store contents as a global variable
        
        Args:
        None
        
        Returns:
        None
    """
	start = timer()
	print_div("initializing Glove")
	global glove_weights
	with open(PATH_GLOVE_DATA) as f:
		for line in f.readlines():
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			glove_weights[word] = coefs
	print "{0} sec".format(np.round(timer() - start))
	
def extract_word2vec(q1, q2, row):
    """
        extract similarities from two sets of vectors
        
        Args:
        q1: question 1
        q2: question 2
        row: dictionary for storing outputs
        
        Returns:
        None
    """
    global model_word2vec
    s = []
    for w1 in q1:
        for w2 in q2:
			try:
				s.append(model_word2vec.similarity(w1, w2))
			except:
				print "skip ({0},{1})".format(w1, w2) 				
    basic_stats(s, row, "word2vec_")

def print_div(msg, N=75):
    """
        print divider with a message inlined
        
        Args:
        msg: a message
        N: length of a divider
        
        Returns:
        None
    """
	print msg + ' ' + '-' * (N - len(msg))
	
def extract(train, tests, outdir, nprocs):
    """
        extract features
        
        Args:
        train: path to a train csv file
        tests: a list of paths to test csv files
        outdir: output directory
        
        Returns:
        None
    """
	init_glove()
    split_text = lambda t: t.lower().split()

	df_train = pd.read_csv(train)	
	print_div('preprocess ' + os.path.basename(train))
	start = timer()
	df_train = preprocess.preprocess_texts(df_train)
	df_train["question1"] = df_train["question1"].apply(split_text)
	df_train["question2"] = df_train["question2"].apply(split_text)
	print "{0} sec".format(np.round(timer() - start))
	extract_in_parallel(df_train, nprocs, os.path.join(outdir, os.path.basename(train)))

	# test
	if tests is not None:
		for test in tests:
			print test
			df_test = preprocess.preprocess_texts(pd.read_csv(test))
			df_test["question1"] = df_test["question1"].apply(split_text)
			df_test["question2"] = df_test["question2"].apply(split_text)
			extract_in_parallel(df_test, n_threads, os.path.join(outdir, os.path.basename(test)))
	
def extract_in_parallel(df, nprocs, outfile):
    """
        extract features in parallel
        
        Args:
        df: Pandas dataframe
        nprocs: number of processes
        outfile: path to an output file
        
        Returns:
        None
    """
	with open(outfile, "wb") as f:
		worker_args = [{
			"q1": r["question1"],
			"q2": r["question2"],
			"id": r["id"] if "id" in r else r["test_id"],
			"classlabel": r["is_duplicate"] if "is_duplicate" in r else None
		} for i, r in df.iterrows()]
		pool = multiprocessing.Pool(processes=nprocs)
		with open(outfile, "wb") as f:
			firstline=True
			#for row in tqdm.tqdm(pool.imap_unordered(worker, worker_args)):
			for args in tqdm.tqdm(worker_args):
				row = worker(args)
				if row is None:
					continue
				if firstline:
					print >> f, ",".join(row.keys())
					firstline = False
				print >> f, ",".join([str(v) for v in row.values()])

def main(argv):
	args=parse_args(argv)
	extract(args.train, args.test, args.outdir, args.nprocs)
    
if __name__ == '__main__':
    exit(main(sys.argv[1:]))
