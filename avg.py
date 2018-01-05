#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np
import pandas as pd
import sklearn.metrics
from collections import OrderedDict
from keras import backend as K
_EPSILON = K.epsilon()

def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out", help="a path to an output file.", type=argparse.FileType('w'), default=sys.stdout)
    parser.add_argument("-r", "--rank", help="rank average", action="store_true")
    parser.add_argument("paths", nargs="+", help="submission files")
    return parser.parse_args(argv)

def avg(paths, method):
    dfs = [pd.read_csv(f) for f in paths]
    df = dfs[0]
    if "proba" in df.columns:
        columns = ["proba"]
    else:
        columns = ["is_duplicate"]
    if method == "average":
        for i in range(1, len(dfs)):
            df[columns] += dfs[i][columns]
        df[columns] /= float(len(dfs))
    elif method == "rank":
        df[columns] = df[columns].rank(axis=0)
        for i in range(1, len(dfs)):
            df[columns] += dfs[i][columns].rank(axis=0)
        df[columns] = df[columns] - df[columns].min()
        df[columns] = df[columns] / df[columns].max()
    return df
	
def metrics(y, y_score):
	y_pred = (y_score > 0.5).astype(int)
	values = OrderedDict()
	# accuracy, precision, recall, f-measure, auc, etc.
	values["accuracy"] = sklearn.metrics.accuracy_score(y, y_pred)
	values["precision"] = sklearn.metrics.precision_score(y, y_pred)
	values["recall"] = sklearn.metrics.recall_score(y, y_pred)
	values["f1"] = sklearn.metrics.f1_score(y, y_pred)
	values["auc"] = sklearn.metrics.roc_auc_score(y, y_score)
	values["average_precision"] = sklearn.metrics.average_precision_score(y, y_score)
	values["brier_score_loss"] = sklearn.metrics.brier_score_loss(y, y_score)
	values["log_loss"] = sklearn.metrics.log_loss(y, y_score, eps=_EPSILON)
	df_metrics = pd.DataFrame.from_dict([values])
	#df_metrics = df_metrics[values.keys()]
	conf = sklearn.metrics.confusion_matrix(y, y_pred).astype(int)
	return df_metrics, conf

def main(argv):
    args = parse_args(argv)
    df = avg(args.paths, "average" if not args.rank else "rank")
    if "proba" in df.columns:
        df, conf = metrics(df.is_duplicate.values, df.proba.values)
    df.to_csv(args.out, index=False)
	
if __name__ == '__main__':
    exit(main(sys.argv[1:]))

