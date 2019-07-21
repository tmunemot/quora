## Quora Question Pairs

This repository contains scripts used for [Quora question pairs competition](https://www.kaggle.com/c/quora-question-pairs#description). The goal of the challenge is to build a natural language processing solution that automatically flags duplicated questions posted to Quora.

### Dependencies
Python 3.7

Python packages

* numpy
* scipy
* pandas
* sklearn
* gensim
* keras
* tqdm
* nltk
* tensorflow-gpu (==2.1.0rc0)

### Siamese Recurrent Architecture
![architecture](https://raw.githubusercontent.com/tmunemot/quora/master/resource/model.png)

### How to Run

```
# clone repositry
git clone https://github.com/tmunemot/quora.git
cd quora

# install dependencies
pipenv install
pipenv shell
```

Once you've successfully set up a virtual environment, run `sync.sh`. This script attempts to download the competition dataset via [kaggle API](https://github.com/Kaggle/kaggle-api). Please refer to the API credential section in the link if you haven't used the api previously. The command also downloads [a pretrained word embedding](https://code.google.com/archive/p/word2vec) with rsync. This is a word2vec model that captures linguistic regularities found in Google News articles.

`split.sh` randomly samples subsets from training data without replacement. `siemese_evaluation.py` is a main python script for training a model.

```
# download data and a word embedding
./sync.sh

# sample validation and development datasets
./split.sh -l 20000 train.csv validation.csv development.csv

# evaluate siamese recurrent architecture
python siamese_evaluation.py --epochs 60 \
                              --batch-size 128 \
                              --recurrent-unit lstm \
                              --distance-metric euclidean \
                              --num-units 64 \
                              train.csv validation.csv development.csv \
                              ./outdir
```

### Reference
Jonas  Mueller, Aditya Thyagarajan. "Siamese Recurrent Architecture for Learning Sentence Similarity" Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence, 2016 [\[pdf\]](https://pdfs.semanticscholar.org/72b8/9e45e8ad8b44bdcab524b959dc09bf63eb1e.pdf)
