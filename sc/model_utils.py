import sys
import os
import numpy as np
import pandas as pd
import keras.layers
#import importlib.util
from keras.models import Model, Sequential
from keras.layers import Embedding, Bidirectional, LSTM, GRU
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Reshape, Merge, BatchNormalization, TimeDistributed
from keras.layers.core import Lambda
from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint
import similarity
from preprocess import MAX_SEQUENCE_LENGTH, MAX_NUM_WORDS, EMBEDDING_DIM
import sklearn.metrics
from collections import OrderedDict

import models.bGRU32C_GLV42B300D
import models.bGRU64C_GLV42B300D
import models.bGRU128C_GLV42B300D
import models.bLSTM32C_GLV42B300D
import models.bLSTM64C_GLV42B300D
import models.bLSTM128C_GLV42B300D
import models.CNV64L3C_GLV42B300D
import models.GRU64C_GLV42B300D
import models.GRU128C_GLV42B300D
import models.LSTM64C_GLV42B300D
import models.LSTM128C_GLV42B300D

MODELS_ROOT=os.path.join("..", "nb", "models")

_EPSILON = K.epsilon()

def get_model(model_id, weights, num_words, max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM):
	if model_id == "bGRU32C_GLV42B300D":
		return models.bGRU32C_GLV42B300D.get_model(weights, num_words, max_sequence_length=max_sequence_length, embedding_dim=embedding_dim)
	elif model_id == "bGRU64C_GLV42B300D":
		return models.bGRU64C_GLV42B300D.get_model(weights, num_words, max_sequence_length=max_sequence_length, embedding_dim=embedding_dim)
	elif model_id == "bGRU128C_GLV42B300D":
		return models.bGRU128C_GLV42B300D.get_model(weights, num_words, max_sequence_length=max_sequence_length, embedding_dim=embedding_dim)	
	elif model_id == "bLSTM32C_GLV42B300D":
		return models.bLSTM32C_GLV42B300D.get_model(weights, num_words, max_sequence_length=max_sequence_length, embedding_dim=embedding_dim)			
	elif model_id == "bLSTM64C_GLV42B300D":
		return models.bLSTM64C_GLV42B300D.get_model(weights, num_words, max_sequence_length=max_sequence_length, embedding_dim=embedding_dim)	
	elif model_id == "bLSTM128C_GLV42B300D":
		return models.bLSTM128C_GLV42B300D.get_model(weights, num_words, max_sequence_length=max_sequence_length, embedding_dim=embedding_dim)	
	elif model_id == "CNV64L3C_GLV42B300D":
		return models.CNV64L3C_GLV42B300D.get_model(weights, num_words, max_sequence_length=max_sequence_length, embedding_dim=embedding_dim)	
	elif model_id == "GRU64C_GLV42B300D":
		return models.GRU64C_GLV42B300D.get_model(weights, num_words, max_sequence_length=max_sequence_length, embedding_dim=embedding_dim)
	elif model_id == "GRU128C_GLV42B300D":
		return models.GRU128C_GLV42B300D.get_model(weights, num_words, max_sequence_length=max_sequence_length, embedding_dim=embedding_dim)	
	elif model_id == "LSTM64C_GLV42B300D":
		return models.LSTM64C_GLV42B300D.get_model(weights, num_words, max_sequence_length=max_sequence_length, embedding_dim=embedding_dim)	
	elif model_id == "LSTM128C_GLV42B300D":
		return models.LSTM128C_GLV42B300D.get_model(weights, num_words, max_sequence_length=max_sequence_length, embedding_dim=embedding_dim)	
	else:
		raise Exception("model {0} not found".format(model_id))
	
def get_model3(weights, num_words, max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM):
	Q1 = Sequential()
	Q1.add(Embedding(num_words, embedding_dim, weights=[weights], input_length=max_sequence_length, trainable=False))
	Q1.add(TimeDistributed(Dense(embedding_dim, activation='relu')))
	Q1.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(embedding_dim, )))
	Q2 = Sequential()
	Q2.add(Embedding(num_words, embedding_dim, weights=[weights], input_length=max_sequence_length, trainable=False))
	Q2.add(TimeDistributed(Dense(embedding_dim, activation='relu')))
	Q2.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(embedding_dim, )))
	model = Sequential()
	model.add(Merge([Q1, Q2], mode='concat'))
	model.add(BatchNormalization())
	model.add(Dense(200, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(200, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(200, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(200, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy', _logloss])	
	return model	

def get_model4(weights, num_words, max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM, dist="cosine"):
	# inputs
	input_q1 = Input(shape=(max_sequence_length,), dtype='int32')
	input_q2 = Input(shape=(max_sequence_length,), dtype='int32')
	# embedded layer
	embedding_layer = Embedding(num_words,
								embedding_dim,
								weights=[weights],
								input_length=max_sequence_length,
								trainable=False)
	q1 = embedding_layer(input_q1)
	q2 = embedding_layer(input_q2)

	# GRU layer
	shared_gru = Bidirectional(GRU(64, dropout=0.2, recurrent_dropout=0.2))
	q1_gru = shared_gru(q1)
	q2_gru = shared_gru(q2)	
	merged_gru = keras.layers.concatenate([q1_gru, q2_gru], axis=-1)	
	dense_gru = Dense(64, activation='relu')(merged_gru)
	dense_gru = Dropout(0.5)(dense_gru)
	
	# LSTM layer
	shared_lstm = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
	q1_lstm = shared_lstm(q1)
	q2_lstm = shared_lstm(q2)	
	merged_lstm = keras.layers.concatenate([q1_lstm, q2_lstm], axis=-1)	
	dense_lstm = Dense(64, activation='relu')(merged_lstm)
	dense_lstm = Dropout(0.5)(dense_lstm)
	
	# CNN
	cnn1 = Conv1D(64, 2, activation='relu', padding='same')
	q1_cnn = MaxPooling1D(2)(cnn1(q1))
	q2_cnn = MaxPooling1D(2)(cnn1(q2))
	cnn2 = Conv1D(64, 2, activation='relu', padding='same')
	q1_cnn = MaxPooling1D(2)(cnn2(q1_cnn))
	q2_cnn = MaxPooling1D(2)(cnn2(q2_cnn))
	cnn3 = Conv1D(64, 2, activation='relu', padding='same')
	q1_cnn = MaxPooling1D(2)(cnn3(q1_cnn))
	q2_cnn = MaxPooling1D(2)(cnn3(q2_cnn))
	merged_cnn = keras.layers.concatenate([q1_cnn, q2_cnn], axis=-1)
	dense_cnn = Dense(64, activation='relu')(Flatten()(merged_cnn))
	dense_cnn = Dropout(0.5)(dense_cnn)

	merged = keras.layers.concatenate([dense_cnn, dense_gru, dense_lstm], axis=-1)
	dense = Dense(64, activation='relu')(merged)
	pred = Dense(1, activation='sigmoid')(Dropout(0.5)(dense))
	
	model = Model(inputs=[input_q1, input_q2], outputs=pred)
	model.compile(loss=_logloss, optimizer='adam', metrics=['accuracy'])	
	return model
	
def _logloss(y_true, y_pred):
	y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
	out = -(y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred))
	return K.mean(out, axis=-1)
	
def train(model_id, train, dev, model_path, weights, num_words, epochs=10, batch_size=128, pretrained=None):
	model = get_model(model_id, weights, num_words)
	model.summary()
	model_checkpoint = ModelCheckpoint(model_path, monitor='loss', save_best_only=True)
	if pretrained is not None:
		print(pretrained)
		model.load_weights(pretrained)
	model.fit(train[:2], train[2], batch_size=batch_size, 
				epochs=epochs, verbose=1, shuffle=True,
				callbacks=[model_checkpoint], 
				validation_data=(dev[:2], dev[2]))
	return model

#def train_all(train, model_path, weights, num_words, epochs=10, batch_size=128, pretrained=None):
#	model = get_model2(weights, num_words)
	#model.summary()
#	model_checkpoint = ModelCheckpoint(model_path, monitor='loss', save_best_only=True)
#	if pretrained is not None:
#		print(pretrained)
#		model.load_weights(pretrained)
#	model.fit(train[:2], train[2], batch_size=batch_size, 
#				epochs=epochs, verbose=1, shuffle=True,
#				callbacks=[model_checkpoint])
#	return model

def test(test, model, model_path, outfile):
	#model = get_model(model_path)
	model.load_weights(model_path)
	y_proba = model.predict(test[:2], verbose=0)
	df_metrics, conf = metrics(test[2], y_proba)
	print(df_metrics)
	print(conf)
	df_metrics.to_csv(outfile[:-4]+"_metrics.csv", index=False)
	df = create_prediction_df(test[2], y_proba, test[3])
	df.to_csv(outfile, index=False)

def test_on_submission(test, model, model_path, outfile):
	model.load_weights(model_path)
	y_proba = model.predict(test[:2], verbose=0)
	df_id = pd.DataFrame(test[2], columns=["test_id"])
	df_proba = pd.DataFrame(y_proba, columns=["is_duplicate"])
	df = pd.concat([df_id, df_proba], axis=1)
	df.to_csv(outfile, index=False)	
	
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

