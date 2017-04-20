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

	row["classlabel"] = args["classlabel"]
	
	return row

def extract_base(q1, q2, row):
	row["base_wlen_diff"] = np.abs(len(q1) - len(q2))	
	s1 = set(q1)
	s2 = set(q2)
	row["base_wlen_common"] = len(s1 & s2)
	row["base_wlen_xor"] = len(s1 ^ s2)
	
def extract(csvfile, outfile, n_threads):
	df = pd.read_csv(csvfile)
	print '-' * 30 
	print 'preprocess ' + csvfile	
	df = preprocess.preprocess_texts(df)
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
