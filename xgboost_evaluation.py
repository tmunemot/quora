#!/usr/bin/env python
""" evaluate xgboost algorithm for Quora question pairs challenge """
import sys
import os
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb

from collections import OrderedDict
import args_common
import utils

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
    "num_round":400,
}


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    args_common.add_evaluation_args(parser)
    # TODO: add an option for xgboost hyperparameters
    return parser.parse_args(argv)


def fit(df, outfile, df_test):
    """
        train a xgboost model
        
        Args:
        df: training data
        outfile: output file path
        df_test: data used for early stopping
        
        Returns:
        a trained xgboost model object
    """
    # set data and parameters
    columns = [c for c in df.columns.values if c not in ("id", "classlabel")]
    y = df["classlabel"].values
    X = convert_to_dmatrix(df, columns, y)
    Xtest = convert_to_dmatrix(df_test, columns, df_test["classlabel"].values)
    print "start training xgboost ({0})".format(outfile)
    watchlist = [(X, "train"), (Xtest, "test")]
	
    bst = xgb.train(XGB_PARAMS, X, XGB_PARAMS["num_round"], watchlist, early_stopping_rounds=100, verbose_eval=10)
    
    # save results
    bst.save_model(outfile)
    return bst


def predict(bst, df, outpath, is_submission=False):
    """
        predict with a trained xgboost model and save results
        
        Args:
        bst: a trained xgboost model
        df: test data
        outpath: output file path
        is_submission: a flag to indicate the data is for submission
        
        Returns:
        None
    """
    y = df["classlabel"].values if not is_submission else None
    columns = [c for c in df.columns.values if c not in ("id", "classlabel")]
    X = convert_to_dmatrix(df, columns, y)
    y_pred = bst.predict(X)

    if not is_submission:
        df_pred = utils.create_prediction_df(y, y_pred, df["id"].values)
        df_pred.to_csv(outpath, index=False)
        df_metrics, conf = utils.metrics(y, y_pred)
        print df_metrics
        df_metrics.to_csv(outpath[:-4] + "_metrics.csv", index=False)		
    else:
        # prediction for a submission
        df_id = pd.DataFrame(df["id"].values, columns=["test_id"])
        df_proba = pd.DataFrame(y_pred, columns=["is_duplicate"])
        df_pred = pd.concat([df_id, df_proba], axis=1)		
        df_pred.to_csv(outpath, index=False)        


def convert_to_dmatrix(df, columns, y):
    """
        convert Pandas dataframe object to DMatrix format
        
        Args:
        df: dataframe
        columns: a list of columns used in training
        y: target labels
        
        Returns:
        xgboost DMatrix object
    """
    return xgb.DMatrix(df[columns].values, label=y, missing=np.nan)


def xgboost_evaluation(df_train, df_dev, df_val, outdir):
    """
        evaluate a xgboost model
        
        Args:
        df_train: training data
        df_val: development data used during training
        df_eval: evaluation data
        outdir: output directory
        
        Returns:
        None
    """
    utils.mkdir_p(outdir)
    bst = fit(df_train, os.path.join(outdir, "xgboost_model.out"), df_dev)
    predict(bst, df_dev, os.path.join(outdir, "development.csv"))
    predict(bst, df_val, os.path.join(outdir, "validation.csv"))


def main(argv):
    args=parse_args(argv)
    df_train = pd.read_csv(args.train)
    df_dev = pd.read_csv(args.dev)
    df_val = pd.read_csv(args.val)
    xgboost_evaluation(df_train, df_dev, df_val, args.outdir)


if __name__ == '__main__':
    exit(main(sys.argv[1:]))

