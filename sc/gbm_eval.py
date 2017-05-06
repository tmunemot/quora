#!/usr/bin/python
""" a script to train a classifier with given data """
import sys
import os
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import errno
from collections import OrderedDict
import sklearn.metrics
from keras import backend as K

_EPSILON = K.epsilon()

def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    add_evaluate_base_args(parser)
    return parser.parse_args(argv)

def add_evaluate_base_args(parser):
    parser.add_argument("train", help="train csv files")
    parser.add_argument("test", help="test csv files")	
    parser.add_argument("dev", help="development csv files")	
    parser.add_argument("val", help="validation csv files")
    parser.add_argument("outdir", help="a path to an output directory.")	
	
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise	
def create_prediction_df(y, y_proba, ids):
    df_id = pd.DataFrame(ids, columns=["id"])
    df_proba = pd.DataFrame(y_proba, columns=["proba"])
    df_cl = pd.DataFrame(y, columns=["is_duplicate"])
    return pd.concat([df_id, df_proba, df_cl], axis=1)

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

XGB_PARAMS={
    "colsample_bytree": 0.6,
    "silent":1,
    "eval_metric":"logloss",
    "nthread":14,
    "min_child_weight":1,
    "subsample":0.6,
    "eta":0.3,
    "objective":"binary:logistic",
    "max_depth":5,
    #"gamma":0.15,
    #"booster":"gbtree",
    "num_round":400,
}
		
def fit(df, outfile, df_test):
    # set data and parameters
    columns = [c for c in df.columns.values if c not in ("id", "classlabel")]
    y = df["classlabel"].values
    X = convert_to_dmatrix(df, columns, y)
    Xtest = convert_to_dmatrix(df_test, columns, df_test["classlabel"].values)
    print "start training xgboost ({0})".format(outfile)
    watchlist = [(X, 'train'), (Xtest, 'test')]
	
    bst = xgb.train(XGB_PARAMS, X, XGB_PARAMS["num_round"], watchlist, early_stopping_rounds=100, verbose_eval=10)
    
    # save results
    bst.save_model(outfile)
    return bst

def predict(bst, df, outpath, is_submission=False):
    y = df["classlabel"].values if not is_submission else None
    columns = [c for c in df.columns.values if c not in ("id", "classlabel")]
    X = convert_to_dmatrix(df, columns, y)
    y_pred = bst.predict(X)
    #y_pred = y_pred[:, 1]
    if not is_submission:
        # eval, dev dataset
        df_pred = create_prediction_df(y, y_pred, df["id"].values)
        df_pred.to_csv(outpath, index=False)
        df_metrics, conf = metrics(y, y_pred)
        print df_metrics
        df_metrics.to_csv(outpath[:-4] + "_metrics.csv", index=False)		
    else:
        # test on submission data
        df_id = pd.DataFrame(df["id"].values, columns=["test_id"])
        df_proba = pd.DataFrame(y_pred, columns=["is_duplicate"])
        df_pred = pd.concat([df_id, df_proba], axis=1)		
        df_pred.to_csv(outpath, index=False)        
	
def convert_to_dmatrix(df, columns, y):
    """ convert pandas dataframe to dmatrix
    """
    return xgb.DMatrix(df[columns].values, label=y, missing=np.nan)

def gbm_eval(df_train, df_test, df_dev, df_val, outdir):
    # make outdir
    mkdir_p(outdir)
	# base
    bst = fit(df_train, os.path.join(outdir, "xgb_train.model"), df_dev)
    predict(bst, df_dev, os.path.join(outdir, "eval_dev.csv"))
    predict(bst, df_val, os.path.join(outdir, "eval_val.csv"))
	# val
    #df_train_dev = pd.concat([df_train, df_dev], axis=0)		
    #bst_train_dev = fit(df_train, os.path.join(outdir, "xgb_train_dev.model"), df_val)
    #predict(bst_train_dev, df_val, os.path.join(outdir, "eval_all_val.csv"))
    #predict(bst_train_dev, df_test, os.path.join(outdir, "submission.csv"), True)	
	# submission
    #df_all = pd.concat([df_train, df_dev, df_val], axis=0)			
    #bst_all = fit(df_all, os.path.join(outdir, "xgb_all.model"))
    #predict(bst_all, df_test, os.path.join(outdir, "submission.csv"), True)
    print "finished pred"
	
def main(argv):
    args=parse_args(argv)
    df_train = pd.read_csv(args.train)
    df_dev = pd.read_csv(args.dev)	
    df_val = pd.read_csv(args.val)
    df_test = None #pd.read_csv(args.test)	
    gbm_eval(df_train, df_test, df_dev, df_val, args.outdir)
	
if __name__ == '__main__':
    exit(main(sys.argv[1:]))

