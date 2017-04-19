import sys
import os
import numpy as np
import pandas as pd
from keras import backend as K

axis = lambda a: len(a._keras_shape) - 1
dot = lambda a, b: K.batch_dot(a, b, axes=axis(a))
l2_norm = lambda a, b: K.sqrt(K.sum((a - b) ** 2, axis=1))
l1_norm = lambda a, b: K.abs(K.sum(a-b, axis=1))

_EPSILON = K.epsilon()

params={
	"gamma": 1.0,
	"c": 1.0,
	"d": 2.0,
}

def dist_shape(s):
	return (s[1], s[1])

def cosine(x):
	return dot(x[0], x[1]) / K.sqrt(dot(x[0], x[0]) * dot(x[1], x[1]))

def polynomial(x):
	return (params['gamma'] * dot(x[0], x[1]) + params['c']) ** params['d']
# gamma = [0.5, 1.0, 1.5]
# c = 1
# d = [2, 3]	
	
def sigmoid(x):
	return K.tanh(params['gamma'] * dot(x[0], x[1]) + params['c'])
# gamma = [0.5, 1.0, 1.5]
# c = 1

def rbf(x):
	return K.exp(-1 * params['gamma'] * l2_norm(x[0], x[1]) ** 2.0)
# gamma = [0.5, 1.0, 1.5]
	
def manhattan(x):
	return K.exp(-1 * params['gamma'] * l2_norm(x[0], x[1]))
	
def euclidean(x):
	return 1 / (1 + l2_norm(x[0], x[1]))

def exponential(x):
	return K.exp(-1 * params['gamma'] * l2_norm(x[0], x[1]))

def gesd(x):
	euc = 1 / (1 + l2_norm(x[0], x[1]))
	sig = 1 / (1 + K.exp(-1 * params['gamma'] * (dot(x[0], x[1]) + params['c'])))
	return euc * sig


