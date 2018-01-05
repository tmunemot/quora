#!/usr/bin/env python
""" evaluate Siamese recurrent architecture for Quora question pairs challenge

    Reference:
    Jonas  Mueller, Aditya Thyagarajan. "Siamese Recurrent Architecture for Learning Sentence Similarity" Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence, 2016
    https://pdfs.semanticscholar.org/72b8/9e45e8ad8b44bdcab524b959dc09bf63eb1e.pdf

"""
import sys
import os
import argparse
import copy
import numpy as np
import pandas as pd

from collections import OrderedDict

from keras.models import Model
from keras.layers import Input, Embedding, LSTM, GRU, Bidirectional, Merge
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras import backend as K

import preprocess
import args_common
import utils


GLOVE_FILE="./resource/glove.840B.300d.txt"
SIMIRALITY_RBF_GAMMA = 1.0
DEFAULT_ARCH={
    "max_sequence_length": 32,
    "embedding_dim": 300,
    "siamese_metric": "manhattan",
    "recurrent_unit": "lstm",
    "num_units": 64,
    "max_num_words": 20000,
}


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    args_common.add_evaluatation_args(parser)
    args_common.add_dnn_train_args(parser)
    add_siamese_architecture_args(parser)
    return parser.parse_args(argv)


def add_siamese_architecture_args(parser):
    """
        add arguments for Siamese architecture
        
        Args:
        parser: arguments parsed with argparse module
        
        Returns:
        None
    """
    parser.add_argument("--max-sequence-length",
                        help="maximum length of each question",
                        type=int, DEFAULT_ARCH["max_sequence_length"])
    parser.add_argument("--embedding-dim",
                        help="dimension of Glove vectors",
                        type=int,
                        default=DEFAULT_ARCH["embedding_dim"])
    parser.add_argument("--max-num-words",
                    help="maximum number of possible words in embedding",
                    type=int, DEFAULT_ARCH["max_num_words"])
    parser.add_argument("--siamese-metric",
                        help="a distance metric used in Siamese architecture",
                        default=DEFAULT_ARCH["siamese_metric"],
                        choices=("manhattan", "euclidean", "cosine", "sigmoid", "rbf"))
    parser.add_argument("--recurrent-unit",
                        help="recurrent units used in Siamese architecture",
                        default=DEFAULT_ARCH["recurrent_unit"],
                        choices=("lstm", "gru"))
    parser.add_argument("--num-units",
                        help="number of recurrent units",
                        default=, default=DEFAULT_ARCH["num_units"],)


def parse_siamese_architecture_params(args):
    """
        parse parameters for Siamese architecture
        
        Args:
        args: arguments parsed with argparse module
        
        Returns:
        a dictionary with parameters
    """
    params = copy.deepcopy(DEFAULT_ARCH)
    params["max_sequence_length"] = DEFAULT_ARCH["max_sequence_length"]
    params["embedding_dim"] = DEFAULT_ARCH["embedding_dim"]
    params["siamese_metric"] = DEFAULT_ARCH["siamese_metric"]
    params["test_gru"] = DEFAULT_ARCH["test_gru"]
    params["num_units"] = DEFAULT_ARCH["num_units"]
    params["max_num_words"] = DEFAULT_ARCH["max_num_words"]
    return params

def get_model(weights, params):
    """
        return a keras model
        
        Args:
        weights: weights for word embedding
        params: details of a siamese architecture
        
        Returns:
        a keral model object
    """
    # parse parameters
    num_words = params["num_words"]
    max_sequence_length = params["max_sequence_length"]
    embedding_dim = params["embedding_dim"]
    recurrent_unit = params["recurrent_unit"]
    siamese_metric = params["siamese_metric"]

    # inputs
    input_q1 = Input(shape=(max_sequence_length,), dtype="int32")
    input_q2 = Input(shape=(max_sequence_length,), dtype="int32")

    # untrainable embedding layer
    embedding_layer = Embedding(num_words,
                                    embedding_dim,
                                    weights=[weights],
                                    input_length=max_sequence_length,
                                    trainable=False)
    q1 = embedding_layer(input_q1)
    q2 = embedding_layer(input_q2)
    
    # add a shared recurrent layer
    if recurrent_unit == "lstm":
        # LSTM
        recurrent_layer = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    elif recurrent_unit == "gru":
        # GRU
        recurrent_layer = Bidirectional(GRU(64, dropout=0.2, recurrent_dropout=0.2))
    else:
        raise ValueError("recurrent unit {0} is not supported".format(recurrent_unit))
    q1_recurrent = recurrent_layer(q1)
    q2_recurrent = recurrent_layer(q2)

    # calculate a similarity
    pred = Merge(mode=similarity(siamese_metric), output_shape=lambda x: (x[0][0], 1))([q1_recurrent, q2_recurrent])

    # compile a model
    model = Model(inputs=[input_q1, input_q2], outputs=pred)
    model.compile(loss=_logloss, optimizer='adam', metrics=['accuracy', _logloss])
    return model


def _logloss(y_true, y_pred):
    """
        calculate log loss
        
        Args:
        y_true: target labels
        y_pred: predicted labels
        
        Returns:
        log loss
    """
    y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
    out = -(y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred))
    return K.mean(out, axis=-1)


def similarity(metric):
    """
        return a lambda function of a similarity metric
        
        Args:
        metric_id: an identifier string
        
        Returns:
        reference to a similarity function
    """

    axis = lambda a: len(a._keras_shape) - 1
    dot = lambda a, b: K.batch_dot(a, b, axes=axis(a))
    
    if metric == "manhattan":
        # exp (-L1(x0, x1))
        return lambda x: K.exp(-1 * K.abs(K.sum(x[0]-x[1], axis=1)))
    elif metric == "euclidean":
        # exp (-L2(x0, x1))
        return lambda x: K.exp(-1 * K.sqrt(K.sum((x[0] - x[1]) ** 2.0, axis=1)))
    elif metric == "cosine":
        # x0'x1 / sqrt(x0'*x0 * x1'*x1)
        return lambda x: dot(x[0], x[1]) / K.sqrt(dot(x[0], x[0]) * dot(x[1], x[1]))
    elif metric == "sigmoid":
        # sigmoid(x0'x1) = 0.5 * (tanh(x0'x1) + 1)
        return lambda x: 0.5 * (K.tanh(dot(x[0], x[1])) + 1.0)
    elif metric == "rbf":
        # exp (- gamma * L2(x0, x1)^2)
        return K.exp(-1 * SIMIRALITY_RBF_GAMMA * l2_norm(x[0], x[1]) ** 2.0)


def load_weights(filepath, tk, max_num_words, embedding_dim):

    # load glove
    word_index = tk.word_index
    embeddings_index={}
    with open(filepath, encoding="utf8") as f:
        for line in f.readlines():
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # create weights
    num_words = min(max_num_words, len(word_index))
    weights = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= max_num_words:
            continue
        vec = embeddings_index.get(word)
        if vec is not None:
            weights[i] = vec
    return weights, num_words


def fit_tokenizer(df_train, max_num_words):
    """
        train a tokenizer
        
        Args:
        df_train: training data
        max_num_words: limit the number of words used
        
        Returns:
        trained tokenizer instance
    """
    q = np.concatenate([df_train.question1.values, df_train.question2.values])
    tk = Tokenizer(num_words=max_num_words, lower=True, split=" ")
    tk.fit_on_texts(q)
    return tk


def transform_tokenizer(df, tk, max_sequence_length):
    """
        transform question sentences with a trained tokenizer
        
        Args:
        df: dataframe
        tk: trained tokenizer
        max_sequence_length: limit the length of sentence
        
        Returns:
        two dimensional numpy arrays and class labels
    """
    q1 = tk.texts_to_sequences(df.question1.values)
    q2 = tk.texts_to_sequences(df.question2.values)
    q1 = pad_sequences(q1, maxlen=max_sequence_length, dtype='int32',
                       padding='post', truncating='post', value=0.)
    q2 = pad_sequences(q2, maxlen=max_sequence_length, dtype='int32',
                       padding='post', truncating='post', value=0.)
    return np.array(q1), np.array(q2), df.is_duplicate.values


def fit(df, outfile, df_test, weights, epochs, batch_size, pretrained, params):
    """
        train a Siamese architecture with a recurrent layer
        
        Args:
        df: training data
        outfile: output file path
        df_test: data used for early stopping
        
        Returns:
        a trained xgboost model object
    """
    siamese_model = get_model(weights, params)
    siamese_model.summary()
    model_checkpoint = ModelCheckpoint(outfile, monitor='loss', save_best_only=True)
    
    if pretrained is not None:
        siamese_model.load_weights(pretrained)
    
    model.fit(train[:2], train[2], batch_size=batch_size,
                  epochs=epochs, verbose=1, shuffle=True,
                  callbacks=[model_checkpoint],
                  validation_data=(dev[:2], dev[2]))
    return model


def predict(siamese_model, data, outpath, is_submission=False):
    """
        make predictions with a trained Siamese model
        
        Args:
        siamese_model: a trained keras model
        data_arrays: numpy arrays that holds train data
        outpath: output file path
        is_submission: a flag to indicate the data is for submission
        
        Returns:
        None
    """
    y_pred = siamese_model.predict(test[:2], verbose=0)
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


def siamese_evaluation(df_train, df_dev, df_val, outdir, epochs, batch_size, pretrained, nprocs, params):
    """
        evaluate a siamese architecture for a semantic sentence similarity task
        
        Args:
        df_train: training data
        df_val: development data used during training
        df_eval: evaluation data
        outdir: output directory
        
        Returns:
        None
    """
    utils.mkdir_p(outdir)
    
    # parse parameters
    max_num_words = params["max_num_words"]
    max_sequence_length = params["max_sequence_length"]
    embedding_dim = params["embedding_dim"]
    
    # preprocess all data
    df_train = preprocess.base(df_train, nprocs)
    df_dev = preprocess.base(df_dev, nprocs)
    df_val = preprocess.base(df_val, nprocs)
    
    # train tokenizer and apply to dataframes
    tokenizer = fit_tokenizer(df_train, max_num_words)
    train = transform_tokenizer(df_train, tokenizer, max_sequence_length)
    dev = transform_tokenizer(df_dev, tokenizer, max_sequence_length)
    val = transform_tokenizer(df_val, tokenizer, max_sequence_length)

    # load glove weights used in word embedding
    weights = load_weights(GLOVE_FILE, tokenizer, max_num_words, embedding_dim):
    
    # train a model
    siamese_model = fit(train, os.path.join(outdir, "siamese_model.out"), dev, weights, epochs, batch_size, pretrained, params)
    
    # predict
    predict(siamese_model, dev, os.path.join(outdir, "development.csv"))
    predict(siamese_model, val, os.path.join(outdir, "validation.csv"))


def main(argv):
    args=parse_args(argv)
    df_train = pd.read_csv(args.train)
    df_dev = pd.read_csv(args.dev)
    df_val = pd.read_csv(args.val)
    siamese_evaluation(df_train, df_dev, df_val, args.outdir, epochs, batch_size, pretrained, args.nprocs, parse_siamese_architecture_params(args))


if __name__ == '__main__':
    exit(main(sys.argv[1:]))
