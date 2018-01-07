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
import gensim
import time
import tqdm
from collections import OrderedDict

from keras.models import Model
from keras.layers import Input, Embedding, LSTM, GRU, Bidirectional, merge
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras import backend as K

import preprocess
import args_common
import utils

K.set_learning_phase(1)

SIMIRALITY_RBF_GAMMA = 1.0
WORD2VEC_EMBEDDING_DIM = 300
DEFAULT_ARCH={
    "max_sequence_length": 32,
    "siamese_metric": "manhattan",
    "recurrent_unit": "lstm",
    "num_units": 64,
    "max_num_words": 20000,
    "unidirectional": False,
}


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args_common.add_evaluation_args(parser)
    args_common.add_dnn_train_group(parser)
    add_siamese_architecture_group(parser)
    return parser.parse_args(argv)


def add_siamese_architecture_group(parser):
    """
        add arguments for Siamese architecture
        
        Args:
        parser: arguments parsed with argparse module
        
        Returns:
        None
    """
    group = parser.add_argument_group("arguments for Siamese recurrent architecture")
    group.add_argument("--max-sequence-length",
                       help="maximum length of each question",
                       type=int, default=DEFAULT_ARCH["max_sequence_length"])
    group.add_argument("--max-num-words",
                       help="maximum number of possible words in embedding",
                       type=int, default=DEFAULT_ARCH["max_num_words"])
    group.add_argument("--siamese-metric",
                       help="a distance metric used in Siamese architecture",
                       default=DEFAULT_ARCH["siamese_metric"],
                       choices=("manhattan", "euclidean", "cosine", "sigmoid", "rbf"))
    group.add_argument("--recurrent-unit",
                       help="recurrent units used in Siamese architecture",
                       default=DEFAULT_ARCH["recurrent_unit"],
                       choices=("lstm", "gru"))
    group.add_argument("--num-units",
                       help="number of recurrent units",
                       default=DEFAULT_ARCH["num_units"],)
    group.add_argument("--unidirectional",
                       help="use unidirectional recurrent units instead of bidirectional",
                       action="store_true")


def get_model(weights, siamese_params):
    """
        return a keras model
        
        Args:
        weights: weights for word embedding
        siamese_params: parameters of a siamese architecture
        
        Returns:
        a keras model object
    """
    # parse parameters
    num_words, embedding_dim = weights.shape
    max_sequence_length = siamese_params["max_sequence_length"]
    recurrent_unit = siamese_params["recurrent_unit"]
    siamese_metric = siamese_params["siamese_metric"]
    num_units = siamese_params["num_units"]
    unidirectional = siamese_params["unidirectional"]
    
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
        recurrent_layer = LSTM(num_units, dropout=0.2, recurrent_dropout=0.2)
    elif recurrent_unit == "gru":
        # GRU
        recurrent_layer = GRU(num_units, dropout=0.2, recurrent_dropout=0.2)
    else:
        raise ValueError("recurrent unit {0} is not supported".format(recurrent_unit))

    if not unidirectional:
        recurrent_layer = Bidirectional(recurrent_layer)

    q1_recurrent = recurrent_layer(q1)
    q2_recurrent = recurrent_layer(q2)

    # calculate a similarity
    pred = merge([q1_recurrent, q2_recurrent], mode=similarity(siamese_metric), output_shape=lambda x: x[0])

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

    #axis = lambda a: len(a._keras_shape) - 1
    dot = lambda a, b: K.batch_dot(a, b, axes=-1)
    
    if metric == "manhattan":
        # exp (-L1(x0, x1))
        return lambda x: K.exp(-1 * K.abs(K.sum(x[0]-x[1], axis=-1)))
    elif metric == "euclidean":
        # exp (-L2(x0, x1))
        return lambda x: K.exp(-1 * K.sqrt(K.sum((x[0] - x[1]) ** 2.0, axis=-1)))
    elif metric == "cosine":
        # x0'x1 / sqrt(x0'*x0 * x1'*x1)
        return lambda x: dot(x[0], x[1]) / K.sqrt(dot(x[0], x[0]) * dot(x[1], x[1]))
    elif metric == "sigmoid":
        # sigmoid(x0'x1) = 0.5 * (tanh(x0'x1) + 1)
        return lambda x: 0.5 * (K.tanh(dot(x[0], x[1])) + 1.0)
    elif metric == "rbf":
        # exp (- gamma * L2(x0, x1)^2)
        return K.exp(-1 * SIMIRALITY_RBF_GAMMA * l2_norm(x[0], x[1]) ** 2.0)
    else:
        ValueError("unexpected similarity metrics")

def load_weights(word2vec, tokenizer, max_num_words):
    # convert to 2d numpy array
    num_words = min(len(word2vec.vocab.keys()), max_num_words)
    weights = np.zeros((num_words, WORD2VEC_EMBEDDING_DIM), dtype=np.float32)
    word_index = tokenizer.word_index
    for word, i in word_index.items():
        print "{0}, {1}".format(i, word)
        if i >= num_words:
            continue
        if word in word2vec.wv:
            weights[i] = word2vec.wv[word]
    return weights


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
    return np.array(q1, dtype=np.int32), np.array(q2, dtype=np.int32), df.is_duplicate.values


def fit(train, outfile, dev, weights, dnn_train_params, siamese_params):
    """
        train a Siamese architecture with a recurrent layer
        
        Args:
        df: training data
        outfile: output file path
        df_test: data used for early stopping
        
        Returns:
        a trained xgboost model object
    """
    # parse parameters
    epochs = dnn_train_params["epochs"]
    batch_size = dnn_train_params["batch_size"]
    pretrained = dnn_train_params["pretrained"]
    
    siamese_model = get_model(weights, siamese_params)
    print "obtained model"
    #siamese_model.summary()
    model_checkpoint = ModelCheckpoint(outfile, monitor='loss', save_best_only=True)
    
    if pretrained is not None:
        siamese_model.load_weights(pretrained)
    print "start training"
    history = siamese_model.fit(list(train[:2]), train[2], batch_size=batch_size,
              epochs=epochs, verbose=1, shuffle=True,
              callbacks=[model_checkpoint],
              validation_data=(list(dev[:2]), dev[2]))

    df_history = pd.DataFrame({"epoch": [i + 1 for i in history.epoch],
                   "acc": history.history["acc"],
                   "validation": history.history["val_acc"]})
    
    return siamese_model, df_history


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


def siamese_evaluation(df_train, df_dev, df_val, outdir, nprocs, dnn_train_params, siamese_params):
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
    # parse parameters
    max_num_words = siamese_params["max_num_words"]
    max_sequence_length = siamese_params["max_sequence_length"]
    
    # load word2vec
    print "loading {0}".format(os.path.basename(preprocess.WORD2VEC_FILE))
    t = time.time()
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(preprocess.WORD2VEC_FILE, binary=True)
    print "done ({0} sec)".format(round(time.time() - t))

    # train tokenizer and apply to dataframes
    tokenizer = fit_tokenizer(df_train, max_num_words)
    train = transform_tokenizer(df_train, tokenizer, max_sequence_length)
    dev = transform_tokenizer(df_dev, tokenizer, max_sequence_length)
    val = transform_tokenizer(df_val, tokenizer, max_sequence_length)

    # load word2vec weights used in word embedding
    weights = load_weights(word2vec, tokenizer, max_num_words)

    # train a model
    siamese_model, df_history = fit(train, os.path.join(outdir, "siamese_model.out"), dev, weights, dnn_train_params, siamese_params)
    df_history.to_csv(os.path.join(outdir, "history.csv"), index=False)

    # predict
    predict(siamese_model, dev, os.path.join(outdir, "development.csv"))
    predict(siamese_model, val, os.path.join(outdir, "validation.csv"))


def main(argv):
    args=parse_args(argv)
    
    # parse datasets
    df_train, df_dev, df_val = args_common.process_evaluation_args(args)
    
    # parse parameters
    siamese_params = args_common.parse_args_to_dict(args, DEFAULT_ARCH.keys())
    dnn_train_params = args_common.parse_args_to_dict(args, args_common.DEFAULT_DNN_TRAIN.keys())
    
    # evaluate
    siamese_evaluation(df_train, df_dev, df_val, args.outdir, args.nprocs, dnn_train_params, siamese_params)

if __name__ == '__main__':
    exit(main(sys.argv[1:]))
