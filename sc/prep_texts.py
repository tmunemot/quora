#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
import csv
import argparse
import numpy as np
import pandas as pd
import preprocess

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Performs feature extraction.")
    parser.add_argument('-n', '--n-threads', help="number of threads", type=int, default=14)
    parser.add_argument('csvs', nargs="+", help="train, test csv files used to train tokenizer")
    parser.add_argument('outdir', help="a path to output csv file")
    return parser.parse_args(argv)

def	prep_texts(csvs, outdir, n_threads):
	for p in csvs:
		print p
		df = pd.read_csv(p)
		df_proc = preprocess.base(df, n_threads)
		df_proc.to_csv(os.path.join(outdir, os.path.basename(p)), index=False)

def main(argv):
	args=parse_args(argv)
	prep_texts(args.csvs, args.outdir, args.n_threads)
    
if __name__ == '__main__':
    exit(main(sys.argv[1:]))
