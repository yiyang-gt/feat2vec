#!/usr/bin/env python

'''
Part-of-Speech tagger feature extractor
and other stuff for data processing

Author: Yi Yang
Email: yangyiycc@gmail.com
'''

import logging, re, os, sys
import numpy as np
import scipy.sparse as sp
from sys import argv
try:
    import cPickle as pickle
except:
    import pickle

logger = logging.getLogger("da.prepare.data")

def generate_data(conll_files, thre=50):
    '''
    Generate all needed data structs including:
    1. spmat: n x d instance - feature sparse matrix
    2. labels: 1 x n label array for all the instance
    3. templates: 1 x d feature template identity array (map features into feature template ids)
    4. pivots: 1 x r feature id array (required by pivot based methods only)
    '''

    source = 'oct27'
    target = 'daily547'

    feat_dict = {} # a dictionary to map features back to feature indices
    index2feat = [] # a list to map feature indices to features
    mat = [] # token index - feature index pairs
    sents = [] # sentence ids
    labels = []

    template_dict = {'word':0, 'prevword':1, 'nextword':2, 'prevprevword':3, 'nextnextword':4, 
                    'prefix1':5, 'prefix2':6, 'prefix3':7, 'prefix4':8, 'suffix1':9, 'suffix2':10,
                    'suffix3':11, 'suffix4':12, 
                    'ContainsNum':13, 'ContainsUpper':14, 'ContainsHyphen':15}
    templates = [] # template ids

    train_idx = [] # oct27
    test_idx = [] # daily547

    instidx = 0
    featidx = 0

    for fname in conll_files:
        domain = fname.split('/')[-1].split('.')[0] # get short file name: oct27 / daily547

        attrs, loclabels, locsents = attribute_getter(fname)
        for features in attrs: # iterate over instances
            for key in features.keys(): # iterate over features
                feat = key + '=' + features[key]
                if feat not in feat_dict:
                    feat_dict[feat] = featidx
                    templates.append(template_dict[key]) # feature template id
                    featidx = featidx + 1
                mat.append((instidx, feat_dict[feat]))
            if domain == source:
                train_idx.append(instidx)
            elif domain == target:
                test_idx.append(instidx)
            instidx = instidx + 1
        labels.extend(loclabels)
        sents.extend(locsents)
    
    data = []
    row = []
    col = []
    for pair in mat:
        data.append(1)
        row.append(pair[0])
        col.append(pair[1])
    spmat = sp.csr_matrix( (data,(row,col)), shape=(instidx, featidx), dtype=np.int32 )

    domain_idxs = [train_idx, test_idx]
    pivots = compute_pivots(spmat, domain_idxs, thre)

    ### begin logging
    logger.info('created a %d x %d token-feature sparse matrix' %spmat.shape)
    logger.info('created a length = %d tag list' %len(labels))
    logger.info('created a length = %d token sentence index list' %len(sents))
    logger.info('created a length = %d feature template id list' %len(templates))
    logger.info('created a length = %d pivot list' %len(pivots))
    ### end logging

    # making all the lists arrays
    return {"spmat":spmat, "labels":np.array(labels), "templates":np.array(templates), 
            "domain_idxs":(train_idx, test_idx), "pivots":np.array(pivots)}

def compute_pivots(spmat, domain_idxs, thre):
    """
    compute pivot features according to a feature frequency threshold
    spmat: the n x d data sparse matrix
    domain_idxs: a list of lists - each list consists of instance idices of one domain
    thre: frequency threshold
    """
    piv_flag = np.ones(spmat.shape[1], dtype=bool)
    for i,idxs in enumerate(domain_idxs): # go over each domain
        logger.info('%d tokens belong to domain %d' %(len(idxs),i+1))
        piv_cands = np.array(spmat[idxs,:].sum(0) > thre).ravel() # feature frequency within this domain
        piv_flag = np.multiply(piv_flag, piv_cands)
    pivots = [idx for idx,flag in enumerate(piv_flag) if flag==True]
    return pivots

def save_data(obj, fname):
    f = open(fname,"wb")
    pickle.dump(obj, f, protocol=2)
    f.close()

def load_data(fname):
    f = open(fname,"rb")
    obj = pickle.load(f)
    f.close()
    return obj

def save_features(conll_files, feat_file, feat_dict_file):
    """
    Extract features for POS tagging from COOL formated files
    save features into feat_file
    save dictionary mapping features to feature template ids into feat_dict_file
    """
    template_dict = {'word':0, 'prevword':1, 'nextword':2, 'prevprevword':3, 'nextnextword':4, 
                    'prefix1':5, 'prefix2':6, 'prefix3':7, 'prefix4':8, 'suffix1':9, 'suffix2':10,
                    'suffix3':11, 'suffix4':12, 
                    'ContainsNum':13, 'ContainsUpper':14, 'ContainsHyphen':15}
    feats = set()
    ff = open(feat_file, "w")
    fd = open(feat_dict_file, "w")
    for fname in conll_files:
        attrs, loclabels, locsents = attribute_getter(fname)
        for features in attrs: # iterate over instances
            for key in features.keys(): # iterate over features
                feat = key + '=' + features[key]
                ff.write(feat + " ")
                if feat not in feats:
                    fd.write(feat + "\t" + str(template_dict[key]) + "\n")
                    feats.add(feat)
            ff.write("\n")
    fd.close()
    ff.close()

def attribute_getter(infile):
    """
    input is a conll format file
    get triple (attributes, labels, sentence indices)
    attributes: a list of dictionaries, each dictionary contains all the features 
                correspond to an instance
    labels: a list of tags
    sentence indices: a list of integers correspond to sentence indices
    """
    attrs = []
    labels = []
    sents = []
    tags = []
    tokens = []
    sentIdx = 0
    fin = open(infile, 'r')
    for line in fin:
        line = line.strip()
        if line == '':
            for index in xrange(len(tokens)):
                features = feature_detector(tokens, index)
                attrs.append(features)
                labels.append(tags[index])
                sents.append(sentIdx)
            del tags[:], tokens[:]
            sentIdx = sentIdx + 1
        else:
            parts = line.split()
            if len(parts)<2: print line
            tags.append(parts[1])
            tokens.append(parts[0])
    fin.close()
    return attrs, labels, sents

def feature_detector(tokens, index):
    """
    extract 16 features for each token
    according to paper 'A Maximum Entropy Model for Part-Of-Speech Tagging' (Adwait Ratnaparkhi, 1996)
    return as a dictionary of 'feature type: feature value'
    """
    word = tokens[index]
    if index == 0:
        prevword = '<s>'
        prevprevword = '<null>'
    elif index == 1:
        prevword = tokens[index-1]
        prevprevword = '<s>'
    else:
        prevword = tokens[index-1]
        prevprevword = tokens[index-2]
    if index == len(tokens)-1:
        nextword = '</s>'
        nextnextword = '<null>'
    elif index == len(tokens)-2:
        nextword = tokens[index+1]
        nextnextword = '</s>'
    else:
        nextword = tokens[index+1]
        nextnextword = tokens[index+2]

    if re.search('\d', word):
        containsNum = 'true'
    else:
        containsNum = 'false'

    if re.search('[A-Z]', word):
        containsUpper = 'true'
    else:
        containsUpper = 'false'

    if re.search('-', word):
        containsHyphen = 'true'
    else:
        containsHyphen = 'false'

 #   if re.match('[0-9]+(\.[0-9]*)?|[0-9]*\.[0-9]+$', word):
 #       shape = 'number'
 #   elif re.match('\W+$', word):
 #       shape = 'punct'
 #   elif re.match('[A-Z][a-z]+$', word):
 #       shape = 'upcase'
 #   elif re.match('[a-z]+$', word):
 #       shape = 'downcase'
 #   elif re.match('\w+$', word):
 #       shape = 'mixedcase'
 #   else:
 #       shape = 'other'

    features = {
        'word': word,
        'prevprevword': prevprevword,
        'prevword': prevword,
        'nextword': nextword,
        'nextnextword': nextnextword,
        'prefix1': word[:1],
        'prefix2': word[:2],
        'prefix3': word[:3],
        'prefix4': word[:4],
        'suffix4': word[-4:],
        'suffix3': word[-3:],
        'suffix2': word[-2:],
        'suffix1': word[-1:],
        'ContainsNum' : containsNum,
        'ContainsUpper' : containsUpper,
        'ContainsHyphen' : containsHyphen,
#        'shape': shape
        }
    return features


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info('begin logging')
    base = './data/twitter/'
    conll_files = [base+fname for fname in os.listdir(base) if fname.endswith('.conll')]
    save_features(conll_files, "./data/twitter_feat.txt", "./data/twitter_feat_template.txt")
    data = generate_data(conll_files, 50)
    save_data(data, './data/dataset_twitter.pkl') 
    logger.info('end logging')

