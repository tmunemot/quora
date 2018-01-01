#!/usr/bin/python
# -*- coding: utf-8 -*-
""" functions for preprocessing question texts in parallel
"""
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
from multiprocessing import Process, Pool
from multiprocessing.managers import BaseManager


WORD2VEC_BINARY_FILE="./resource/GoogleNews-vectors-negative300.bin"
PREPROCESS_GENERAL="./resource/replace.csv"
PREPROCESS_UNITS="./resource/units.csv"


class SharedResource(object):
    """
        hold a word2vec model and used it to preprocess texts in parallel
    """
	def __init__(self):
		print "initialize word2vec"
		self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_BINARY_FILE, binary=True)
		self.word2vec_vocab = set(self.word2vec.vocab.keys())		
	def get(self, fname):
		if fname == "word2vec_vocab":
			return self.word2vec_vocab	
		raise Exception("{0} is not defined".format(fname))

class ResourceManager(BaseManager):
	pass
ResourceManager.register("SharedResource", SharedResource)


def aggregate(ret):
    """
        append dataframe objects that are processed in parallel
        
        Args:
        ret: a dataframe object returned by a process
    """
	aggregate.out.append(ret)
aggregate.out = []


def prep(text, vocabs):
    """
        substitute special characters, replace abbreviations, and encode to ascii
        
        Args:
        text: a question
        vocabs: a list of vocabulary available in a pretrained word2vec model
        
        Returns:
        preprocessed text
    """
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
		prep.dict = pd.read_csv(PREPROCESS_GENERAL).set_index("src")["dst"].to_dict()
		for k in prep.dict.keys():
			prep.dict[k.lower()] = prep.dict[k]
	for k in prep.dict.keys():
		if k in text:
			text = text.replace(k, prep.dict[k])
	if prep.units is None:
		with open(PREPROCESS_UNITS, "rU") as f:
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
    """
        post-process output from NLTK tokenizer
        
        Args:
        tokens: a list contains a tokenized text
        
        Returns:
        processed tokens
    """
    out = []
        for t in tokens:
            if t[-1] == ".":
                out.append(t[:-1])
                else:
                    out.append(t)
        return out


def tokenize_wrapper(text):
    """
        wrapper function for NLTK tokenizer
        
        Args:
        text: a question
        
        Returns:
        a list contains a tokenized text
    """
	return post(nltk.word_tokenize(text))


def indices_for(df, nprocs):
    """
        group rows in dataframe and assign each group to each process
        
        Args:
        df: Pandas dataframe object
        nprocs: number of processes used
        
        Returns:
        indeces grouped to each process
    """
    N = df.shape[0]
    L = int(N / nprocs)
    indices = []
    for i in range(nprocs):
        for j in range(L):
            indices.append(i)
    for i in range(N - (nprocs * L)):
        indices.append(nprocs - 1)
    return indices


def base(df, nprocs=15):
    """
        perform basic preprocessing in parallel
        
        Args:
        df: Pandas dataframe object
        nprocs: number of processes used
        
        Returns:
        Pandas dataframe object with preprocessed texts
    """
            
	t = time.time()
	# prepare a shared resource and manager classes
	manager = ResourceManager()
	manager.start()
	shared = manager.SharedResource()
	
	# run in parallel
	pool = Pool(nprocs)
	for i, (name, df_group) in enumerate(df.groupby(indices_for(df, nprocs))):
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
    """
        base worker function for text preprocessing
        
        Args:
        shared_obj: word2vec dictionary
        df: Pandas dataframe object
        iproc: process index
        
        Returns:
        Pandas dataframe object with preprocessed texts
    """
	try:
		df.fillna(value="", inplace=True)
		vocab = shared_obj.get("word2vec_vocab")
        # preprocess
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
    """
        extract a phone number
        
        Args:
        text: a question
        
        Returns:
        an empty string or a phone number
    """
	try:
        # assumes there is only one us number in one question but this is likely wrong
		for match in phonenumbers.PhoneNumberMatcher(text, "US"):
			return phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164)
	except:
		pass
	return ""

