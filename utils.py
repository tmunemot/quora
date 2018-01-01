#!/usr/bin/python
import sys
import os
import time
import errno
import multiprocessing
import sklearn.metrics
import numpy as np
import pandas as pd


def mkdir_p(path):
    """
        emulate to "mkdir -p" in Python
        https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
        
        Args:
        path: path to a new directry
        
        Returns:
        None
        """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def add_evaluate_base_args(parser):
    parser.add_argument("train", help="training data")
    parser.add_argument("val", help="validation data")
    parser.add_argument("dev", help="development data used during training")
    parser.add_argument("outdir", help="output directory")


def notify(section, dlen=75):
    """
        an utility function for reporting processing time
        
        Args:
        section: a section id printed along with a processing time
        dlen: length of a divider
        
        Returns:
        None
    """
    if notify.t_start != -1:
        # print time elapsed
        t = time.time()
        msg = str(t - notify.t_prev) + " sec (" + str(t - notify.t_start) + ")"
        notify.t_prev = t
        print msg
    else:
        # initialize time
        notify.t_start = time.time()
        notify.t_prev = notify.t_start
    msg = "section {0}, {1}".format(notify.section_id, section)
    print msg + '-' * (dlen - len(msg))
    notify.section_id += 1
notify.t_start = -1
notify.section_id = 1


def create_prediction_df(y, y_proba, ids):
    """
        create a dataframe that contians predictions
        
        Args:
        y: target labels
        y_score: predicted probabilities ranging 0 to 1
        
        Returns:
        a dataframe contains predicted probabilities
    """
    df_id = pd.DataFrame(ids, columns=["id"])
    df_proba = pd.DataFrame(y_proba, columns=["proba"])
    df_cl = pd.DataFrame(y, columns=["is_duplicate"])
    return pd.concat([df_id, df_proba, df_cl], axis=1)


def metrics(y, y_score):
    """
        calculate evaluation metrics given predictions
        
        Args:
        y: target labels
        y_score: predicted probabilities ranging 0 to 1
        
        Returns:
        a trained xgboost model object
    """
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
    conf = sklearn.metrics.confusion_matrix(y, y_pred).astype(int)
    return df_metrics, conf

