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
#import gensim
import sklearn.metrics.pairwise
from timeit import default_timer as timer
from gensim.models.keyedvectors import KeyedVectors
from nltk import ngrams
from nltk.corpus import stopwords

eng_stopwords = set(stopwords.words('english'))

model_w2v = None
path_w2v = "../pre/GoogleNews-vectors-negative300.bin.gz"

glove_weights = {}
path_glove = "../pre/glove.840B.300d.txt"
#path_glove = "../pre/glove.6B.300d.txt"

#stop_words = set(['the','a','an','and','but','if','or','because','as','what',
#				'which','this','that','these','those','then','just','so','than',
#				'such','both','through','about','for','is','of','while','during','to'])
#stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
#				'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
#				'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
#				'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
#				'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is',
#				'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
#				'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
#				'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 
#				'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
#				'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 
#				'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
#				'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
#				'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
#				'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
#				's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 
#				'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn',
#				'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 
#				'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']

def remove_stopwords(q):
	return [w for w in q if w not in stop_words]
	
def split_txt(q):
	return q.lower().split()
	
def parse_args(argv):
	parser = argparse.ArgumentParser(description="Performs feature extraction.")
	parser.add_argument('-n', '--n-threads', help="number of threads", type=int, default=15)
	parser.add_argument('-t', '--test', nargs="+", help="test csv file")	
	parser.add_argument('train', help="train csv file")
	parser.add_argument('outdir', help="a path to output csv file")
	return parser.parse_args(argv)

def worker(args):
	q1 = args["q1"]
	q2 = args["q2"]

	#if np.count_nonzero(q1) < 2 or np.count_nonzero(q2) < 2:
	if len(q1) < 2 or len(q2) < 2:
		print "skipped an empty question for {0}".format(args["id"])
		return None

	row = OrderedDict()	
	row["id"] = args["id"]
	
	# base features
	extract_base(q1, q2, row)
	# glove
	try:
		extract_glove(q1, q2, row)
	except:
		return None
	
    # word2vec
	#extract_w2v(q1, q2, row)
	if args["classlabel"] is not None:
		row["classlabel"] = args["classlabel"]
	
	return row

def ngram_val(s1, s2):
	count = float(len(s1 & s2))
	ratio = count / np.max([len(s1 | s2), 1])
	return count, ratio
	
def extract_base(q1, q2, row):
	row["base_wlen_diff"] = np.abs(len(q1) - len(q2))
	
	global eng_stopwords
	
	q1_cl = [v for v in q1 if v not in eng_stopwords]
	q2_cl = [v for v in q2 if v not in eng_stopwords]
	# unigram
	s1 = set(q1_cl)
	s2 = set(q2_cl)
	count, ratio = ngram_val(s1, s2)
	row["base_ugram_count"] = count
	row["base_ugram_ratio"] = ratio
	
	# bigram
	s1 = set([v for v in ngrams(q1_cl, 2)])
	s2 = set([v for v in ngrams(q2_cl, 2)])
	count, ratio = ngram_val(s1, s2)	
	row["base_bgram_count"] = count
	row["base_bgram_ratio"] = ratio
		
	# trigram
	s1 = set([v for v in ngrams(q1_cl, 3)])
	s2 = set([v for v in ngrams(q2_cl, 3)])
	count, ratio = ngram_val(s1, s2)	
	row["base_tgram_count"] = count
	row["base_tgram_ratio"] = ratio

#def extract_tfidf(q1, q2, row):
# term-frequency, inverse document-frequency
	
	
def basic_stats(x, row, prefix):
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
	v1 = get_glove_vecs(q1)
	v2 = get_glove_vecs(q2)
	cos = sklearn.metrics.pairwise.cosine_similarity(v1, v2)
	cos = cos.reshape((1, -1))
	basic_stats(cos, row, "glove_cos")

def get_glove_vecs(q):
	global glove_weights, eng_stopwords
	v = []
	for w in q:	
		#if w in eng_stopwords:
		#	continue
		if w in glove_weights:
			v.append(glove_weights[w])
		elif (len(w) >= 2) and (w[-1] == 's') and (w[:-1] in glove_weights):
			v.append(glove_weights[w[:-1]])
		else:
			#print "word '{0}' is not found".format(w)
			continue
	return np.array(v)		
			
def init_glove():
	start = timer()
	print_div('init Glove')	
	global glove_weights, path_glove
	with open(path_glove) as f:
		for line in f.readlines():
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			glove_weights[word] = coefs
	print "{0} sec".format(np.round(timer() - start))
	
def extract_w2v(q1, q2, row):
    global model_w2v
    s = []
    for w1 in q1:
        for w2 in q2:
			try:
				s.append(model_w2v.similarity(w1, w2))
			except:
				print "skip ({0},{1})".format(w1, w2) 				
    basic_stats(s, row, "w2v_")

def print_div(msg, N=75):
	print msg + ' ' + '-' * (N - len(msg))
	return
	
def extract(train, tests, outdir, n_threads):

	init_glove()

	df_train = pd.read_csv(train)	
	print_div('preprocess ' + os.path.basename(train))
	start = timer()
	df_train = preprocess.preprocess_texts(df_train)
	df_train["question1"] = df_train["question1"].apply(split_txt)
	df_train["question2"] = df_train["question2"].apply(split_txt)		
	print "{0} sec".format(np.round(timer() - start))
	extract_in_parallel(df_train, n_threads, os.path.join(outdir, os.path.basename(train)))

	# test
	if tests is not None:
		for test in tests:
			print test
			df_test = preprocess.preprocess_texts(pd.read_csv(test))
			df_test["question1"] = df_test["question1"].apply(split_txt)
			df_test["question2"] = df_test["question2"].apply(split_txt)	
			extract_in_parallel(df_test, n_threads, os.path.join(outdir, os.path.basename(test)))		
	
def extract_in_parallel(df, n_threads, outfile):
	with open(outfile, "wb") as f:
		worker_args = [{
			"q1": r["question1"],
			"q2": r["question2"],
			"id": r["id"] if "id" in r else r["test_id"],
			"classlabel": r["is_duplicate"] if "is_duplicate" in r else None
		} for i, r in df.iterrows()]
		pool = multiprocessing.Pool(processes=n_threads)
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
	extract(args.train, args.test, args.outdir, args.n_threads)
    
if __name__ == '__main__':
    exit(main(sys.argv[1:]))
