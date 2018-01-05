#!/usr/bin/env python


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


def add_dnn_train_args(parser):
    """
        add arguments for training a model
        
        Args:
        parser: arguments parsed with argparse module
        
        Returns:
        None
    """
    parser.add_argument("--epochs", help="number of epochs", type=int, default=60)
    parser.add_argument("--batch-size", help="batch size", type=int, default=128)
    parser.add_argument("--pretrained", help="path to a pretrained model")
    parser.add_argument("--nprocs", help="number of preprocesses", type=int, default=15)

# TODO
#def parse_dnn_train_args(args):


