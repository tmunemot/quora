import sys
import os
import numpy as np
import pandas as pd
import keras.layers
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

_EPSILON = K.epsilon()

# TODO:
# compare different embedding methods and pretrained models for glove or word2vec
# different sequense length, max nb owrds, embedding dims
# LSTM vs GRU
# test bidirectional model
# combined RNN units with Conv1D
	
def get_model(weights, num_words, max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM, dist="cosine"):
	# Bidirectional LSTM (128), loss: 0.3092 - acc: 0.8582 - val_loss: 0.3849 - val_acc: 0.831
	#val_loss: 0.3565 - val_acc: 0.8363
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

	# LSTM / GRU layer
	shared_gru = Bidirectional(GRU(128, dropout=0.2, recurrent_dropout=0.2))
	q1 = shared_gru(q1)
	q2 = shared_gru(q2)	
	
	# Dense
	#for dist in [similarity.cosine, similarity.manhattan, similarity.euclidean, similarity.rbf, similarity.gesd]:
	#merged = Lambda(getattr(similarity, dist))([q1, q2])
	#normed = BatchNormalization()(merged)
	merged = keras.layers.concatenate([q1, q2], axis=-1)	
	dense1 = Dense(64, activation='relu')(merged)
	pred = Dense(1, activation='sigmoid')(Dropout(0.5)(dense1))

	model = Model(inputs=[input_q1, input_q2], outputs=pred)
	model.compile(loss=_logloss, optimizer='adam', metrics=['accuracy'])
	return model

def get_model2(weights, num_words, max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM):
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
	
	for i in range(3):
		cnn = Conv1D(64, 2, activation='relu', padding='same')
		q1 = MaxPooling1D(2)(cnn(q1))
		q2 = MaxPooling1D(2)(cnn(q2))
		
	# Dense
	merged = keras.layers.concatenate([q1, q2], axis=-1)
	dense1 = Dense(64, activation='relu')(Flatten()(merged))
	pred = Dense(1, activation='sigmoid')(Dropout(0.5)(dense1))

	model = Model(inputs=[input_q1, input_q2], outputs=pred)
	model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy', _logloss])	
	#model.compile(loss=_logloss, optimizer='adam', metrics=['accuracy'])
	return model

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
	shared_gru = Bidirectional(GRU(128, dropout=0.2, recurrent_dropout=0.2))
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

def train(train, dev, model_path, weights, num_words, epochs=10, batch_size=128, pretrained=None):
	model = get_model2(weights, num_words)
	model.summary()
	model_checkpoint = ModelCheckpoint(model_path, monitor='loss', save_best_only=True)
	if pretrained is not None:
		print(pretrained)
		model.load_weights(pretrained)
	model.fit(train[:2], train[2], batch_size=batch_size, 
				epochs=epochs, verbose=1, shuffle=True,
				callbacks=[model_checkpoint], 
				validation_data=(dev[:2], dev[2]))


#def test(test, model_path):
#    model = get_conv3d(False)
#    model.load_weights(model_path)
#    y_score = model.predict_proba(x_test, verbose=0)
#    score = model.evaluate(x_test, y_test, verbose=0)
#    print('Test loss:', score[0])    
#    df_metrics, conf = metrics(y_test, y_score)
#    
#    display(df_metrics)
#    display(conf)
