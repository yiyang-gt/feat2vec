#!/usr/bin/env python

'''
Test different domain adaptation methods using SVM tagger
A light demo for part-of-speech tagging of tweets

Author: Yi Yang
Email: yangyiycc@gmail.com
'''

import sys, time, logging, re
import numpy as np
import scipy.sparse as sp

from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import preprocessing

from twproc import load_data, save_data
from mldae import mldae

import gensim
sys.path.append('feat2vec')
import feat2vec

try:
    import cPickle as pickle
except:
    import pickle

logger = logging.getLogger("da.twpos")
num_template = 13 # learn feature embeddings for features correspond to the first num_templates feature templates

def test(spmat, labels, templates, train_idx, test_idx, mode="none", pivots=None):

    xr = spmat[train_idx, :]
    yr = labels[train_idx]
    
    W = None
    if mode == "mldae":
        W = mldae(spmat, pivots)
    elif mode == "feat2vec":
        W = feat_embed(spmat, templates, epoch=3, dim=25, ng=15)

    if W is not None:
        if mode == "mldae":
            xr = sp.hstack([xr, sp.csr_matrix(xr * W).tanh()], format='csr')
        else:
            d, dim = W.shape
            newxr = np.ndarray(shape=(xr.shape[0], num_template*dim), dtype=np.float32)
            for t in xrange(num_template):
                D = sp.spdiags([1.0*(templates==t)],[0],len(templates),len(templates))
                Wt = D * W
                newxr[:, t*dim:(t+1)*dim] = xr * Wt
            xr = sp.hstack([xr, sp.csr_matrix(newxr).tanh()], format='csr')

    reg = 0.5
    logger.info('start training with reg = %.4f' %(reg))
    clf = svm.LinearSVC(C=reg, loss='l1', tol=1e-4, multi_class='ovr')
    clf = clf.fit(xr, yr)
    logger.info('finish training with reg = %.4f' %(reg))
    
    xe = spmat[test_idx, :]
    ye = labels[test_idx]
    if W is not None: 
        if mode == "mldae":
            xe = sp.hstack([xe, sp.csr_matrix(xe * W).tanh()], format='csr')
        else:
            newxe = np.ndarray(shape=(xe.shape[0], num_template*dim), dtype=np.float32)
            for t in xrange(num_template):
                D = sp.spdiags([1.0*(templates==t)],[0],len(templates),len(templates))
                Wt = D * W
                newxe[:, t*dim:(t+1)*dim] = xe * Wt
            xe = sp.hstack([xe, sp.csr_matrix(newxe).tanh()], format='csr', dtype=np.float32)
        
    logger.info('start testing with reg = %.4f' %(reg))
    acc = accuracy_score(ye, clf.predict(xe))
    logger.info('finish testing with reg = %.4f' %(reg))
    print 'test accuracy = %.4f' %acc


def feat_embed(spmat, templates, epoch=5, dim=50, ng=5):
    n, d = spmat.shape
    instances = get_instances(spmat, templates)
    template_dict = dict(zip(range(len(templates)), templates))
    template_dict = {key:val for key,val in template_dict.iteritems() if val<num_template}
    model = feat2vec.Feat2Vec(instances, size=dim, epoch=epoch, workers=4, negative=ng, bow=False, template_dict=template_dict)
    W = np.ndarray(shape=(d, dim), dtype=np.float32)
    for i in xrange(d):
        if i not in model:
            continue
        W[i,:] = model[i]
    return W


def get_instances(xx, templates): # for feat2vec
    # TODO: old code, need to Rewrite
    ks = xx.sum(1)
    (rows, cols) = xx.nonzero()
    instances = []
    inst = []
    idx = 0
    instlen = 0
    for i in xrange(len(cols)):
        # discard binary features
        if templates[cols[i]] < num_template:
            inst.append(cols[i])
        instlen = instlen + 1
        if instlen % int(ks[idx,0]) == 0:
            instances.append(inst)
            inst = []
            instlen = 0
            idx = idx+1
    return instances


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info('begin logging')
    dataset = load_data('./data/dataset_twitter.pkl')

    spmat, labels, templates, pivots = dataset["spmat"], dataset["labels"], dataset["templates"], dataset["pivots"]
    train_idx, test_idx = dataset["domain_idxs"]

    mode = sys.argv[1]

    test(spmat, labels, templates, train_idx, test_idx, mode, pivots)
    
    logger.info('end logging')

