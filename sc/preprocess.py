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

from string import punctuation
from nltk.tag.stanford import StanfordNERTagger

from multiprocessing import Process, Pool
from multiprocessing.managers import BaseManager

class SharedResource(object):
	NER_HOME="../d/stanford-ner-2016-10-31"
	NER_CLF = os.path.join(NER_HOME, "classifiers/english.all.3class.distsim.crf.ser.gz")
	NER_JAR = os.path.join(NER_HOME, "stanford-ner.jar")	
	def __init__(self):
		print "initialize NERTagger"
		self.NERTagger = StanfordNERTagger(SharedResource.NER_CLF, SharedResource.NER_JAR)
	
	def get(self, fname):
		if fname == "NERTagger":
			return self.NERTagger	
		raise Exception("{0} is not defined".format(fname))

class ResourceManager(BaseManager):
	pass
ResourceManager.register("SharedResource", SharedResource)

def aggregate(ret):
	aggregate.out.append(ret)
aggregate.out = []

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
#	try:
	# apply very basic preprocessing
	df.fillna(value="", inplace=True)
	df["question1"] = df["question1"].apply(replace_char)
	df["question2"] = df["question2"].apply(replace_char)

	# extract named entity
	ner_tagger = shared_obj.get("NERTagger")		
	df["q1_ne"] = df["question1"].apply(apply_ner, args=(ner_tagger,))
	df["q2_ne"] = df["question2"].apply(apply_ner, args=(ner_tagger,))	

	# add phone numbers if exists
	df["q1_phone_us"] = df["question1"].apply(extract_phone)
	df["q2_phone_us"] = df["question2"].apply(extract_phone)
#	except:
#		traceback.print_exc()
#		raise Exception("base_worker failed")
	return iproc, df

def expand(text):
    """ convert to lowercases, expand contracted english words, remove punctuations
    """
    if expand.pattern is None:
        expand.dict = pd.read_csv("./contractions_word2vec.csv").set_index("contracted")["original"].to_dict()
        expand.pattern = re.compile(r"\b(" + "|".join(expand.dict.keys()) + r")\b")
    text = expand.pattern.sub(lambda x: expand.dict[x.group()], text)
    return re.sub(r"[^\w\s]"," ", text)
expand.dict=None
expand.pattern=None

def replace_char(text):
	# replace letters
	text = text.replace("′","'")
	text = text.replace("″","\"")
	return text

def extract_phone(text):
	# assume there is only one us number in one question
	try:
		for match in phonenumbers.PhoneNumberMatcher(text, "US"):
			return phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164)	
	except:
		pass
	return ""

def apply_ner(text, ner_tagger):
	try:
		return ner_tagger.tag(text.split())
	except:
		traceback.print_exc()
		pass
	return []
