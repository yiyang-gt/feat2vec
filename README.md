# Feature Embedding #

Author: Yi Yang

contact: yangyiycc@gmail.com

## Basic Description ##

**Python code for ICLR 2015 submission**: [Unsupervised Domain Adaptation with Feature Embeddings](http://arxiv.org/pdf/1412.4385v1.pdf).

## Requirements ##

* Install [gensim](https://github.com/piskvorky/gensim) by 
  * pip install --upgrade gensim 
* If you want a faster version of this tool, you may also want to 
  * install [Cython](http://cython.org/) by
    * pip install cython 
  * compile the code by running 
    * python setup.py build_ext --inplace

## Demo ##

A **demo** is available by running 
* python save.py --feature_dict_file data/twitter_feat_dict.txt --dim 25 data/twitter_feat.txt data/twitter_embeddings.txt

See details by running
* python save.py -h

## Domain Adaptation for Twitter POS tagging ##

A **light demo** for part-of-speech tagging of tweets is also provided, using data from CMU [Twitter NLP project](https://github.com/brendano/ark-tweet-nlp/). 

oct27 dataset is regarded as source data, and daily547 dataset is regarded as target data. We also sample some unlabeled tweets randomly (see data/twitter folder).

**Run the demo:**

1. Prepare the data (extract features, select pivots, etc.) by running
  * python twproc.py
2. Obtain the baseline (no adaptation) SVM tagging results by running
  * python twpos.py none
3. Obtain the [marginalized Denoising Autoencoders](http://www.cc.gatech.edu/~yyang319/download/yang-acl-2014.pdf) adaptation results by running
  * python twpos.py mldae
4. Obtain the [feature embedding](http://arxiv.org/pdf/1412.4385v1.pdf) adaptation results by running
  * python twpos.py feat2vec
  
  
The first step will create a file data/dataset_twitter.pkl. I got results of 0.8839, 0.8889 and 0.8924 for step 2, 3 and 4. The feat2vec results may vary a litter due to the negative sampling technique. You should obtain even better results with feat2vec by using more unlabeled data.
