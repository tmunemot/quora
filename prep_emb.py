#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import sys
import os
import csv
import argparse
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import tqdm
import multiprocessing
from collections import Counter

import re
import gensim
import nltk
import phonenumbers
from autocorrect import spell

import utils
import preprocess

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Performs feature extraction.")
    parser.add_argument('-n', '--n-threads', help="number of threads", type=int, default=4)
    parser.add_argument('csvs', nargs="+", help="train, test csv files used to train tokenizer")
    parser.add_argument('outdir', help="a path to output csv file")
    return parser.parse_args(argv)

def	prep_emb(csvs, outdir, n_threads):
    utils.notify("load questions")
    df = pd.concat([pd.read_csv(p, usecols=["question1", "question2"]) for p in csvs])
    utils.notify("apply base preprocessings")
    df = preprocess.base(df, 12)
#    df.fillna(value="", inplace=True)
#    df["question1"] = df["question1"].apply(preprocess.base)
#    df["question2"] = df["question2"].apply(preprocess.base)
    #sentences = [s.split() for s in np.concatenate([df["question1"].values,df["question2"].values], axis=0)]
#    sentences = []
#    N=1
#    for s1, s2 in zip(df["question1"].values, df["question2"].values):
#        try:
#            for match in phonenumbers.PhoneNumberMatcher(preprocess.base(s1), "US"):
#                print match
#        except:
#            print s1
#            #return
#        try:
#            for match in phonenumbers.PhoneNumberMatcher(preprocess.base(s2), "US"):
#                print match
#        except:
#            print s2
            #return

#        try:
#            sentences.append(nltk.word_tokenize(s1))
#        except:
#            print "N="+str(N)+" " + "-" * 20 + ",s1"
#            print s1
#            print s2
#            N += 1
#        try:
#            sentences.append(nltk.word_tokenize(s2))
#        except:
#            print "N="+str(N)+" " + "-" * 20 + ",s2"
#            print s2
#            print s1
#            N += 1

    utils.notify("load word2vec")
    return
    model_w2v = gensim.models.KeyedVectors.load_word2vec_format("../d/GoogleNews-vectors-negative300.bin", binary=True)
    utils.notify("check frequency")
    counts = Counter()
    for s in sentences:
        for w in s:
            counts[w] += 1
    comm = counts.most_common()
    print comm[:100]
    print comm[-100:]
    num = Counter()
    for w, c in comm:
        if c < 20 and w not in model_w2v:
            print "{0}: {1}, {2}".format(c, w, spell(w))
            num[c] += 1
    print num

    utils.notify()
    #words = [w for s in sentences for w in s]
    


    #print words
    #print len(words)

    
#    all = []
#    L=[]
#    sw = set(nltk.corpus.stopwords.words('english'))
#    for s in sentences:
#        #L.append(len(s))
#        L.append(len([w for w in s if w not in sw]))
#        #print s
#    sns.distplot(L, bins=np.arange(0,41,1), hist=True, kde=False)
#    plt.show()
#    return

#def extract_in_parallel(df, n_threads, outfile):
#	with open(outfile, "wb") as f:
#		worker_args = [{
#			"q1": r["question1"],
#			"q2": r["question2"],
#			"id": r["id"] if "id" in r else r["test_id"],
#			"classlabel": r["is_duplicate"] if "is_duplicate" in r else None
#		} for i, r in df.iterrows()]
#		pool = multiprocessing.Pool(processes=n_threads)
#		with open(outfile, "wb") as f:
#			firstline=True
#			#for row in tqdm.tqdm(pool.imap_unordered(worker, worker_args)):
#			for args in tqdm.tqdm(worker_args):
#				row = worker(args)
#				if row is None:
#					continue
#				if firstline:
#					print >> f, ",".join(row.keys())
#					firstline = False
#				print >> f, ",".join([str(v) for v in row.values()])

def main(argv):
	args=parse_args(argv)
	prep_emb(args.csvs, args.outdir, args.n_threads)
    
if __name__ == '__main__':
    exit(main(sys.argv[1:]))
