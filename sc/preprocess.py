import sys
import os
import numpy as np
import pandas as pd
import re
from string import punctuation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

GLOVE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "pre")
MAX_SEQUENCE_LENGTH = 32
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 300

def preprocess_texts(df):
	df.question1.fillna("", inplace=True)
	df.question2.fillna("", inplace=True)
	df = revert_abbreviations(df)	
	df["question1"] = df["question1"].apply(text_to_wordlist)
	df["question2"] = df["question2"].apply(text_to_wordlist)
	return df
	#val_acc: 0.7833 - val__logloss: 0.465
	#val_acc: 0.7940 - val__logloss: 0.456

def fit_tokenizer(df, max_num_words=MAX_NUM_WORDS):
	q_all = np.concatenate([df.question1.values, df.question2.values])
	tk = Tokenizer(num_words=max_num_words, lower=True, split=" ")
	tk.fit_on_texts(q_all)
	return tk

def transform_tokenizer(df, tk, maxlen=MAX_SEQUENCE_LENGTH):
	sq1 = tk.texts_to_sequences(df.question1.values)
	sq2 = tk.texts_to_sequences(df.question2.values)
	sq1 = pad_sequences(sq1, maxlen=maxlen, dtype='int32', padding='post', truncating='post', value=0.)
	sq2 = pad_sequences(sq2, maxlen=maxlen, dtype='int32', padding='post', truncating='post', value=0.)
	return np.array(sq1), np.array(sq2)

def load_weights(filepath, tk, max_num_words=MAX_NUM_WORDS, embedding_dim=EMBEDDING_DIM):
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


	
def text_to_wordlist(text, remove_stop_words=True, stem_words=False):
	# Clean the text, with the option to remove stop_words and to stem words.

	# Clean the text
	text = re.sub(r"[^A-Za-z0-9]", " ", text)
	text = re.sub(r"what's", "", text)
	text = re.sub(r"What's", "", text)
	text = re.sub(r"\'s", " ", text)
	text = re.sub(r"\'ve", " have ", text)
	text = re.sub(r"can't", "cannot ", text)
	text = re.sub(r"n't", " not ", text)
	text = re.sub(r"I'm", "I am", text)
	text = re.sub(r" m ", " am ", text)
	text = re.sub(r"\'re", " are ", text)
	text = re.sub(r"\'d", " would ", text)
	text = re.sub(r"\'ll", " will ", text)
	text = re.sub(r"60k", " 60000 ", text)
	text = re.sub(r" e g ", " eg ", text)
	text = re.sub(r" b g ", " bg ", text)
	text = re.sub(r"\0s", "0", text)
	text = re.sub(r" 9 11 ", "911", text)
	text = re.sub(r"e-mail", "email", text)
	text = re.sub(r"\s{2,}", " ", text)
	text = re.sub(r"quikly", "quickly", text)
	text = re.sub(r" usa ", " America ", text)
	text = re.sub(r" USA ", " America ", text)
	text = re.sub(r" u s ", " America ", text)
	text = re.sub(r" uk ", " England ", text)
	text = re.sub(r" UK ", " England ", text)
	text = re.sub(r"india", "India", text)
	text = re.sub(r"switzerland", "Switzerland", text)
	text = re.sub(r"china", "China", text)
	text = re.sub(r"chinese", "Chinese", text) 
	text = re.sub(r"imrovement", "improvement", text)
	text = re.sub(r"intially", "initially", text)
	text = re.sub(r"quora", "Quora", text)
	text = re.sub(r" dms ", "direct messages ", text)  
	text = re.sub(r"demonitization", "demonetization", text) 
	text = re.sub(r"actived", "active", text)
	text = re.sub(r"kms", " kilometers ", text)
	text = re.sub(r"KMs", " kilometers ", text)
	text = re.sub(r" cs ", " computer science ", text) 
	text = re.sub(r" upvotes ", " up votes ", text)
	text = re.sub(r" iPhone ", " phone ", text)
	text = re.sub(r"\0rs ", " rs ", text) 
	text = re.sub(r"calender", "calendar", text)
	text = re.sub(r"ios", "operating system", text)
	text = re.sub(r"gps", "GPS", text)
	text = re.sub(r"gst", "GST", text)
	text = re.sub(r"programing", "programming", text)
	text = re.sub(r"bestfriend", "best friend", text)
	text = re.sub(r"dna", "DNA", text)
	text = re.sub(r"III", "3", text) 
	text = re.sub(r"the US", "America", text)
	text = re.sub(r"Astrology", "astrology", text)
	text = re.sub(r"Method", "method", text)
	text = re.sub(r"Find", "find", text) 
	text = re.sub(r"banglore", "Banglore", text)
	text = re.sub(r" J K ", " JK ", text)

	# Remove punctuation from text
	text = ''.join([c for c in text if c not in punctuation])

	# Optionally, remove stop words
	if remove_stop_words:
		text = text.split()
		#stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',
		#		'just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which',
		#		'Is','If','While','This']				
		text = [w for w in text if not w in stopwords.words('english')]
		text = " ".join(text)

	# Optionally, shorten words to their stems
	if stem_words:
		text = text.split()
		stemmer = SnowballStemmer('english')
		stemmed_words = [stemmer.stem(word) for word in text]
		text = " ".join(stemmed_words)

	# Return a list of words
	return(text)

def revert_abbreviations(df):
	punctuation='["\'?,\.]' # I will replace all these punctuation with ''
	abbr_dict={
		"what's":"what is",
		"what're":"what are",
		"who's":"who is",
		"who're":"who are",
		"where's":"where is",
		"where're":"where are",
		"when's":"when is",
		"when're":"when are",
		"how's":"how is",
		"how're":"how are",

		"i'm":"i am",
		"we're":"we are",
		"you're":"you are",
		"they're":"they are",
		"it's":"it is",
		"he's":"he is",
		"she's":"she is",
		"that's":"that is",
		"there's":"there is",
		"there're":"there are",

		"i've":"i have",
		"we've":"we have",
		"you've":"you have",
		"they've":"they have",
		"who've":"who have",
		"would've":"would have",
		"not've":"not have",

		"i'll":"i will",
		"we'll":"we will",
		"you'll":"you will",
		"he'll":"he will",
		"she'll":"she will",
		"it'll":"it will",
		"they'll":"they will",

		"isn't":"is not",
		"wasn't":"was not",
		"aren't":"are not",
		"weren't":"were not",
		"can't":"can not",
		"couldn't":"could not",
		"don't":"do not",
		"didn't":"did not",
		"shouldn't":"should not",
		"wouldn't":"would not",
		"doesn't":"does not",
		"haven't":"have not",
		"hasn't":"has not",
		"hadn't":"had not",
		"won't":"will not",
		punctuation:'',
		'\s+':' ', # replace multi space with one single space
	}
	df.question1=df.question1.str.lower()
	df.question2=df.question2.str.lower()
	df.question1=df.question1.astype(str)
	df.question2=df.question2.astype(str)
	df.replace(abbr_dict,regex=True,inplace=True)
	return df
	


