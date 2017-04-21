#!/usr/bin/python
import sys
import os
import csv
import traceback
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
import multiprocessing
import warnings
import tqdm
import preprocess
import scipy.stats
import scipy.spatial
import gensim

model_w2v = None
path_w2v = "./model/GoogleNews-vectors-negative300.bin"

def parse_args(argv):
	parser = argparse.ArgumentParser(description="Performs feature extraction.")
	parser.add_argument('-n', '--n-threads', help="number of threads", type=int, default=15)
	parser.add_argument('csvfile', help="input csv file")
	parser.add_argument('outfile', help="a path to output csv file")
	return parser.parse_args(argv)

def worker(args):
	q1 = args["q1"]
	q2 = args["q2"]

	row = OrderedDict()	
	row["id"] = args["id"]
	
	# base features
	extract_base(q1, q2, row)
    
    # word2vec
    extract_w2v(q1, q2, row)

	row["classlabel"] = args["classlabel"]
	
	return row

def extract_base(q1, q2, row):
	row["base_wlen_diff"] = np.abs(len(q1) - len(q2))	
	s1 = set(q1)
	s2 = set(q2)
	row["base_wlen_common"] = len(s1 & s2)
	row["base_wlen_xor"] = len(s1 ^ s2)

def basic_stats(x, row, prefix):
    row[prefix + "_mean"] = np.mean(x)
    row[prefix + "_std"] = np.std(x, ddof=1)
    row[prefix + "_skew"] = scipy.stats.skew(x)
    row[prefix + "_kurtosis"] = scipy.stats.kurtosis(x)
    row[prefix + "_min"] = np.min(x)
    row[prefix + "_max"] = np.max(x)
    row[prefix + "_range"] = row[prefix + "_max"] - row[prefix + "_min"]
    row[prefix + "_q1"] = np.percentile(x, 25)
    row[prefix + "_median"] = np.median(x)
    row[prefix + "_q3"] = np.percentile(x, 75)
    row[prefix + "_iqrange"] = row[prefix + "_q3"] - row[prefix + "_q1"]
    row[prefix + "_mad"] = np.abs(x - row[prefix + "_mean"]) / float(len(x))
    row[prefix + "_coefvar"] = scipy.stats.variation(x)

def extract_w2v(q1, q2):
    global model_w2v
    if model_w2v is None:
        model_w2v = gensim.models.Word2Vec.load_word2vec_format(path_w2v, binary=True)
    #dist = scipy.spatial.distance.cdist(q1_glove, q2_glove, 'cosine')
    #dist = dist[np.triu_indices(dist.shape[0])]
    s = []
    for w1 in q1:
        for w2 in q2:
            s.append(model_w2v.wv.similarity(w1, w2))
    basic_stats(s, row, "w2v_")


def extract(csvfile, outfile, n_threads):
	df = pd.read_csv(csvfile)
	print '-' * 30 
	print 'preprocess ' + csvfile
	df = preprocess.preprocess_texts(df)
	print '-' * 30
    print '-' * 30
    print 'load glove weights '
    filepath = os.path.join(preprocess.GLOVE_DIR, "")
    global glove_weights
    tk = fit_tokenizer(df)
    glove_weights, nw = preprocess.load_weights(filepath, tk)
    print '-' * 30
	worker_args = [{
		"q1": r["question1"],
		"q2": r["question2"],
		"id": r["id"],
		"classlabel": r["is_duplicate"]
	} for i, r in df.iterrows()]
	#pool = multiprocessing.Pool(processes=n_threads)
	with open(outfile, "wb") as f:
		firstline=True
		#for row in tqdm.tqdm(pool.imap_unordered(worker, worker_args)):
		for args in tqdm.tqdm(worker_args):
			row = worker(args)
			if firstline:
				print >> f, ",".join(row.keys())
				firstline = False
			print >> f, ",".join([str(v) for v in row.values()])

def main(argv):
	args=parse_args(argv)
	extract(args.csvfile, args.outfile, args.n_threads)
    
if __name__ == '__main__':
    exit(main(sys.argv[1:]))
