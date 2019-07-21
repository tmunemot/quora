#!/usr/bin/env python
""" commonly used arguments and functions to parse them
"""
import os
import multiprocessing
import pandas as pd
import utils


def add_evaluation_args(parser):
    """
        add base arguments for evaluations

        Args:
        parser: parser object from argparse

        Returns:
        None
    """
    parser.add_argument("train", help="training data")
    parser.add_argument("val", help="validation data")
    parser.add_argument("dev", help="development data used during training")
    parser.add_argument("outdir", help="output directory")
    parser.add_argument("--nprocs", help="number of preprocesses",
                        type=int,
                        default=multiprocessing.cpu_count())


def process_evaluation_args(args):
    """
        process arguments for a basic evaluation

        Args:
        args: arguments loaded with a parser

        Returns:
        dataframe objects
    """

    # load dataframes
    df_train = pd.read_csv(args.train)
    df_dev = pd.read_csv(args.dev)
    df_val = pd.read_csv(args.val)

    # create an output directory
    utils.mkdir_p(args.outdir)

    return df_train, df_dev, df_val


DEFAULT_DNN_TRAIN = {
    "epochs": 60,
    "batch_size": 128,
    "pretrained": None,
    "cuda_visible_devices": -1,
}


def add_dnn_train_group(parser):
    """
        add arguments for training a deep learning model

        Args:
        parser: arguments parsed with argparse module

        Returns:
        None
    """
    group = parser.add_argument_group(
        "general arguments for training a deep learning model")
    group.add_argument("--epochs", help="number of epochs",
                       type=int,
                       default=DEFAULT_DNN_TRAIN["epochs"])
    group.add_argument("--batch-size", help="batch size",
                       type=int, default=DEFAULT_DNN_TRAIN["batch_size"])
    group.add_argument("--pretrained", help="path to a pretrained model")
    group.add_argument("--cuda-visible-devices",
                       help="specify a gpu to use", type=int, default=-1)


def parse_args_to_dict(args, args_list):
    """
        parse arguments

        Args:
        args: arguments parsed with argparse module
        args_list: specify which arguments needs to be in output

        Returns:
        a dictionary of parameters
        """
    params = {}
    for arg in vars(args):
        if arg in args_list:
            params[arg] = getattr(args, arg)
    return params
