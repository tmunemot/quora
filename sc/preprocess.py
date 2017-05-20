#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
import re
import numpy as np
import pandas as pd
import traceback
import time
import gensim
import nltk
import phonenumbers

import utils
from multiprocessing import Process, Pool
from multiprocessing.managers import BaseManager

class SharedResource(object):

	def __init__(self):
		print "initialize word2vec"		
		self.word2vec = gensim.models.KeyedVectors.load_word2vec_format("../d/GoogleNews-vectors-negative300.bin", binary=True)
		self.word2vec_vocab = set(self.word2vec.vocab.keys())		
	def get(self, fname):
		if fname == "word2vec_vocab":
			return self.word2vec_vocab	
		raise Exception("{0} is not defined".format(fname))

class ResourceManager(BaseManager):
	pass
ResourceManager.register("SharedResource", SharedResource)

def aggregate(ret):
	aggregate.out.append(ret)
aggregate.out = []

def prep(text, vocabs):
	text = text.decode("utf-8")
	text = re.sub("’", "'", text)
	text = re.sub("`", "'", text)
	text = re.sub("“", '"', text)
	text = re.sub("？", "?", text)
	text = re.sub("…", " ", text)
	text = re.sub("é", "e", text)
	text = re.sub(r"\.+", ".", text)
	text = text.replace("[math]", " ")
	text = text.replace("[/math]", " ")
    
	if prep.dict is None:
		prep.dict = pd.read_csv("../d/replace.csv").set_index("src")["dst"].to_dict()
		for k in prep.dict.keys():
			prep.dict[k.lower()] = prep.dict[k]
	for k in prep.dict.keys():
		if k in text:
			text = text.replace(k, prep.dict[k])
	if prep.units is None:
		with open("../d/units.csv", "rU") as f:
			prep.units = [l.replace('\n','') for l in f]

	for u in prep.units:
		text = re.sub(r"(\d+\.\d+){0}".format(u),"\\1 {0}".format(u), text)
    
	matches = re.finditer(r"([a-zA-z]*)\.([a-zA-z]*)", text)
	for match in matches:
		m01 = match.group(0)
		m0 = match.group(1)
		m1 = match.group(2)
		if m01 not in vocabs and m1 in vocabs \
							and m1.lower() not in ("com", "org","net","exe", "js", "biz", "care", "ly", "io", "in", "jp", "au", "gov", "ca", "cn", "fr", "hk", "kr", "mx") \
							and (m1.lower() in ("i","a") or len(m1) > 1) \
							and m1[-1] != ".":
			text = text.replace(m01, m01.replace(".", ". "))
	matches = re.finditer(r"([a-zA-z]*)?([a-zA-z]*)", text)
	for match in matches:
		m01 = match.group(0)
		m0 = match.group(1)
		m1 = match.group(2)
		if m01 not in vocabs and m0 in vocabs \
							and m1 in vocabs \
							and (m1.lower() in ("i","a") or len(m1) > 1):
			text = text.replace(m01, m01.replace(".", ". "))
	text = re.sub(r"/", " or ", text)
	text = text.encode("ascii","ignore")
	return text
prep.dict = None
prep.units = None

def post(tokens):
	out = []
	for t in tokens:
		if t[-1] == ".":
			out.append(t[:-1])
		else:
			out.append(t)
	return out

def tokenize_wrapper(text):
	return post(nltk.word_tokenize(text))

def base(df, nprocs=12):
	t = time.time()
	# prepare a shared resource and manager classes
	manager = ResourceManager()
	manager.start()
	shared = manager.SharedResource()
	
	# run in parallel
	pool = Pool(nprocs)
	for i, (name, df_group) in enumerate(df.groupby(utils.indices_for(df, nprocs))):
		pool.apply_async(func=base_worker, args=(shared, df_group, i), callback=aggregate)
	pool.close()
	pool.join()

	# post process
	aggregate.out.sort(key=lambda x: x[0])
	df_out = pd.concat([df_ret for i, df_ret in aggregate.out])
	aggregate.out = []
	print "Time {0}".format(time.time() - t)
	return df_out

def base_worker(shared_obj, df, iproc):
	# apply very basic preprocessing
	try:
		df.fillna(value="", inplace=True)
		vocab = shared_obj.get("word2vec_vocab")
		df["q1"] = df["question1"].apply(prep, args=(vocab,))
		df["q2"] = df["question2"].apply(prep, args=(vocab,))
	
		# tokenize
		df["q1_tokens"] = df["q1"].apply(tokenize_wrapper)
		df["q2_tokens"] = df["q2"].apply(tokenize_wrapper)	

		# add phone numbers if exists
		df["q1_phone_us"] = df["q1"].apply(extract_phone)
		df["q2_phone_us"] = df["q2"].apply(extract_phone)
	except:
		traceback.print_exc()
		raise Exception("Exception")
	return iproc, df

def extract_phone(text):
	# assume there is only one us number in one question
	try:
		for match in phonenumbers.PhoneNumberMatcher(text, "US"):
			return phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164)	
	except:
		pass
	return ""

