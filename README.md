## Quora Question Pairs

This repository contains scripts used in [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs#description) competition. The competition is about building a natural language processing solution that flags duplicated questions posted to Quora.

### Dependencies
Python 2.7

Python packages

* numpy 1.13.3
* scipy 1.0.0
* pandas 0.19.2
* sklearn 0.18.1
* gensim 3.2.0
* keras 2.0.0
* tqdm 4.11.2
* nltk 3.2.2

Pretrained word embedding

* [Word2Vec](https://code.google.com/archive/p/word2vec)
* [GloVe](https://nlp.stanford.edu/projects/glove)

### Siamese Recurrent Architecture
![architecture](https://raw.githubusercontent.com/tmunemot/quora/master/resource/model.png)

### How to Run

Download a training dataset from [Kaggle](https://www.kaggle.com/c/quora-question-pairs/data), uncompress it, and run the following bash command.

```
# install requirements
sudo pip install -r requirements.txt

# create validation and development datasets by randomly sampling from training data
mkdir ./data
./random_split.sh -l 20000 train.csv ./data/{train,val,dev}.csv 

# run a script to evaluate siamese recurrent architecture
./siamese_evaluation.py --epochs 60 --batch-size 128 \
                        --recurrent-unit lstm \
                        --distance-metric manhattan \
                        --num-units 64 \
                        ./data/{train,val,dev}.csv ./results
```

### References
Jonas  Mueller, Aditya Thyagarajan. "Siamese Recurrent Architecture for Learning Sentence Similarity" Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence, 2016 [\[pdf\]](https://pdfs.semanticscholar.org/72b8/9e45e8ad8b44bdcab524b959dc09bf63eb1e.pdf)
