#/bin/bash

kaggle competitions download -p resource quora-question-pairs
wget -O resource/glove.840B.300d.zip -nc http://nlp.stanford.edu/data/glove.840B.300d.zip
wget -O resource/GoogleNews-vectors-negative300.bin.gz -nc https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
