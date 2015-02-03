#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 Yi Yang <yangyiycc@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
 

"""
Feature embedding: transfer each sparse feature to a dense vector [1]_.
Modification based on word2vec of gensim (http://radimrehurek.com/gensim)
Specifically, word2vec's "skip-gram model", using negative sampling [2]_.

**Install Cython with `pip install cython` to use optimized word2vec training** (70x speedup [3]_).

Initialize a model with e.g.::

>>> model = Feat2Vec(instances, size=100, min_count=0, workers=4, ...)


.. [1] Yi Yang and Jacob Eisenstein. Unsupervised Domain Adaptation with Feature Embeddings.
       Submitted to ICLR, 2015.
.. [2] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality.
       In Proceedings of NIPS, 2013.
.. [3] Optimizing word2vec in gensim, http://radimrehurek.com/2013/09/word2vec-in-python-part-two-optimizing/
"""

import logging
import sys
import os
#import heapq
import time
#from copy import deepcopy
import threading
import numpy as np
try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from numpy import exp, dot, zeros, outer, random, dtype, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, argsort, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, asarray

logger = logging.getLogger("feat2vec.feat2vec")

from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from six import iteritems, itervalues, string_types
from six.moves import xrange


try:
    from feat2vec_inner import train_instance, FAST_VERSION
except ImportError:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    FAST_VERSION = -1
    
    def train_instance(model, instance, alpha, work=None):
        """
        Update feature embeddings by training on a single instance.
    
        The instance is a list of Vocab objects (or None, where the corresponding
         is not in the vocabulary. Called internally from `Feat2Vec.train()`.
    
        This is the non-optimized, Python version. If you have cython installed, feat2vec
        will use the optimized version from feat2vec_inner instead.
    
        """
        # precompute negative labels
        labels = zeros(model.negative + 1)
        labels[0] = 1.0
    
        for pos, feat in enumerate(instance):
            if feat is None:
                continue  # OOV feature in the input instance => skip
    
            # now go over all features in the instance, predicting each one in turn
            for pos2, feat2 in enumerate(instance):
                # don't train on OOV features and on the `feature` itself
                if feat2 and not (pos2 == pos):
                    l1 = model.syn0[feat2.index]
                    neu1e = zeros(l1.shape)
    
                    # use this feature (label = 1) + `negative` other random features not from this instance (label = 0)
                    feat_indices = [feat.index]
                    while len(feat_indices) < model.negative + 1:
                        fe = model.table[random.randint(model.table.shape[0])]
                        if fe != feat.index:
                            feat_indices.append(fe)
                    l2b = model.syn1neg[feat_indices] # 2d matrix, k+1 x layer1_size
                    fb = 1. / (1. + exp(-dot(l1, l2b.T))) # propagate hidden -> output
                    gb = (labels - fb) * alpha # vector of error gradients multiplied by the learning rate
                    model.syn1neg[feat_indices] += outer(gb, l1) # learn hidden -> output
                    neu1e += dot(gb, l2b) # save error
    
                    model.syn0[feat2.index] += neu1e  # learn input -> hidden
    
        return len([feat for feat in instance if feat is not None])
    

class Vocab(object):
    """A single vocabulary item"""
    def __init__(self, **kwargs):
        self.count = 0
        self.template = -1
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "<" + ', '.join(vals) + ">"


class Feat2Vec(object):
    """
    Class for training, using and evaluating neural networks

    """

    def __init__(self, instances=None, epoch=5, size=100, alpha=0.025, seed=1, workers=1, min_count=0, 
            min_alpha=0.0001, negative=1, bow=False, template_dict=None):
        """
        Initialize the model from an iterable of `instances`. Each instance is a
        list of features (string or id) that will be used for training.

        `epoch` decides how many times the algorithm will pass the training data

        `size` is the dimensionality of the feature vectors.

        `alpha` is the initial learning rate (will linearly drop to zero as training progresses).

        `seed` = for the random number generator.

        `workers` = use this many worker threads to train the model (=faster training with multicore machines).

        `min_count` = ignore all features with total frequency lower than this.

        `negative` = the int for negative samplers, specifies how many "noise features" should be drawn (usually between 5-20).

        `bow` indicates input features are bag of words representation or not

        `template_dict` is a dictionary to map each feature to a feature template id, only work if bow = False

        """
        self.vocab = {}  # mapping from a feature (string or int) to a Vocab object
        self.index2feat = []  # map from a feature's matrix index (int) to feature (string or int)
        self.table = None # for negative sampling --> this needs a lot of RAM! consider setting back to None before saving
        self.layer1_size = int(size)
        if size % 4 != 0:
            logger.warning("consider setting layer size to a multiple of 4 for greater performance")
        self.alpha = float(alpha)
        self.seed = seed
        self.workers = workers
        self.min_count = min_count
        self.min_alpha = min_alpha
        self.negative = negative
        self.bow = bow
        if not bow:
            self.template_dict = template_dict
            self.templates = list( set( val for val in template_dict.values() ) )
        if instances is not None:
            all_insts=[]
            for e in xrange(epoch):
                rand_idx = np.random.permutation(len(instances))
                rand_insts = [instances[idx] for idx in rand_idx]
                all_insts += rand_insts
            self.build_vocab(all_insts)
            self.train(all_insts)


    def build_vocab(self, instances):
        """
        Build vocabulary from a sequence of instances. 
        Each instance must be a list of unicode strings or ints (indices).

        """
        logger.info("collecting all features and their counts")
        instance_no, vocab = -1, {}
        total_feats = 0
        for instance_no, instance in enumerate(instances):
            if instance_no % 100000 == 0:
                logger.info("PROGRESS: at instance #%i, processed %i features and %i feature types" %
                    (instance_no, total_feats, len(vocab)))
            for feat in instance:
                total_feats += 1
                if feat in vocab:
                    vocab[feat].count += 1
                else:
                    if self.bow:
                        vocab[feat] = Vocab(count=1)
                    else:
                        vocab[feat] = Vocab(count=1, template=self.template_dict[feat])
        logger.info("collected %i feature types from a corpus of %i features and %i instances" %
            (len(vocab), total_feats, instance_no + 1))

        # assign a unique index to each feature
        self.vocab, self.index2feat = {}, []
        if self.bow:
            for feat, v in iteritems(vocab):
                if v.count >= self.min_count:
                    v.index = len(self.vocab)
                    self.index2feat.append(feat)
                    self.vocab[feat] = v
        else:
            for template in self.templates:
                for feat, v in iteritems(vocab):
                    if v.count >= self.min_count and v.template == template:
                        v.index = len(self.vocab)
                        self.index2feat.append(feat)
                        self.vocab[feat] = v

        logger.info("total %i feature types after removing those with count<%s" % (len(self.vocab), self.min_count))
        
        # build the table for drawing random features (for negative sampling)
        self.make_table()
        self.reset_weights()


    def make_table(self, table_size=100000000, power=1.0):
        """
        Create a table using stored vocabulary feature counts for drawing random features in the 
        negative sampling training routines.

        Called internally from `build_vocab()`.

        """
        logger.info("constructing a table with noise distribution from %i features" % len(self.vocab))
        # table (= list of features) of noise distribution for negative sampling
        vocab_size = len(self.index2feat)
        table_size = vocab_size * 100
        self.table = zeros(table_size, dtype=uint32)

        if not vocab_size:
            logger.warning("empty vocabulary in feat2vec, is this intended?")
            return

        fidx = 0
        if self.bow:
            # compute sum of all power (Z in paper)
            train_feats_pow = float(sum([self.vocab[feat].count**power for feat in self.vocab]))
            # go through the whole table and fill it up with the feature indexes proportional to a feature's count**power
            # normalize count^power by Z
            d1 = self.vocab[self.index2feat[fidx]].count**power / train_feats_pow
            for tidx in xrange(table_size):
                self.table[tidx] = fidx
                if 1.0 * tidx / table_size > d1:
                    fidx += 1
                    d1 += self.vocab[self.index2feat[fidx]].count**power / train_feats_pow
                if fidx >= vocab_size:
                    fidx = vocab_size - 1
        else:
            table_offsets = [0]
            for template in self.templates:
                # compute sum of all power (Z in paper)
                train_feats_pow = 0.0
                template_sum = 0
                for feat in self.vocab:
                    voc = self.vocab[feat]
                    if voc.template == template:
                        train_feats_pow += voc.count**power
                        template_sum += 1
    
                # normalize count^power by Z
                d1 = self.vocab[self.index2feat[fidx]].count**power / train_feats_pow
                for tidx in xrange(template_sum*100):
                    self.table[tidx+table_offsets[-1]] = fidx
                    if 1.0 * tidx / (template_sum*100) > d1:
                        fidx += 1
                        d1 += self.vocab[self.index2feat[fidx]].count**power / train_feats_pow
                    if fidx >= table_offsets[-1]+template_sum*100:
                        fidx -= 1
                table_offsets.append(table_offsets[-1]+template_sum*100)
    
            self.table_offsets = table_offsets


    def train(self, instances, total_feats=None, feat_count=0, chunksize=100):
        """
        Update the model's neural weights from a sequence of instances.
        Each instance must be a list of unicode strings or ints (indices).

        """
        if FAST_VERSION < 0:
            import warnings
            warnings.warn("Cython compilation failed, training will be slow. Do you have Cython installed? `pip install cython`")
        logger.info("training model with %i workers on %i vocabulary and %i embedding size"
            ", and 'negative sampling'=%s" %
            (self.workers, len(self.vocab), self.layer1_size, self.negative))

        if not self.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")

        start, next_report = time.time(), [1.0]
        feat_count = [feat_count]
        total_feats = total_feats or int(sum(v.count for v in itervalues(self.vocab)))
        jobs = Queue(maxsize=2 * self.workers)  # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        lock = threading.Lock()  # for shared state (=number of words trained so far, log reports...)

        def worker_train():
            """Train the model, lifting lists of instances from the jobs queue."""
            '''
            multiple working space
            '''
            work = zeros(self.layer1_size, dtype=REAL)  # each thread must have its own work memory
            neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)

            while True:
                job = jobs.get()
                if job is None:  # data finished, exit
                    break
                # update the learning rate before every job
                alpha = max(self.min_alpha, self.alpha * (1 - 1.0 * feat_count[0] / total_feats))
                # how many words did we train on? out-of-vocabulary (unknown) features do not count
                job_words = sum(train_instance(self, instance, alpha, work) for instance in job)
                with lock:
                    feat_count[0] += job_words
                    elapsed = time.time() - start
                    if elapsed >= next_report[0]:
                        logger.info("PROGRESS: at %.2f%% features, alpha %.05f, %.0f features/s" %
                            (100.0 * feat_count[0] / total_feats, alpha, feat_count[0] / elapsed if elapsed else 0.0))
                        next_report[0] = elapsed + 1.0  # don't flood the log, wait at least a second between progress reports

        workers = [threading.Thread(target=worker_train) for _ in xrange(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        def prepare_instances():
            for instance in instances:
                sampled = [self.vocab[feat] for feat in instance
                    if feat in self.vocab]
                yield sampled

        # convert input strings to Vocab objects (eliding OOV/downsampled words), and start filling the jobs queue
        for job_no, job in enumerate(utils.grouper(prepare_instances(), chunksize)):
            logger.debug("putting job #%i in the queue, qsize=%i" % (job_no, jobs.qsize()))
            jobs.put(job)
        logger.info("reached the end of input; waiting to finish %i outstanding jobs" % jobs.qsize())
        for _ in xrange(self.workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers:
            thread.join()

        elapsed = time.time() - start
        logger.info("training on %i features took %.1fs, %.0f features/s" %
            (feat_count[0], elapsed, feat_count[0] / elapsed if elapsed else 0.0))

        return feat_count[0]


    def reset_weights(self):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        logger.info("resetting layer weights")
        random.seed(self.seed)
        self.syn0 = empty((len(self.vocab), self.layer1_size), dtype=REAL)
        # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
        for i in xrange(len(self.vocab)):
            self.syn0[i] = (random.rand(self.layer1_size) - 0.5) / self.layer1_size
        self.syn1neg = zeros((len(self.vocab), self.layer1_size), dtype=REAL)


    def save_word2vec_format(self, fname, fvocab=None, binary=False):
        """
        Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.
        """
        if fvocab is not None:
            logger.info("Storing vocabulary in %s" % (fvocab))
            with utils.smart_open(fvocab, 'wb') as vout:
                for word, vocab in sorted(iteritems(self.vocab), key=lambda item: -item[1].count):
                    vout.write(utils.to_utf8("%s %s\n" % (word, vocab.count)))
        logger.info("storing %sx%s projection weights into %s" % (len(self.vocab), self.layer1_size, fname))
        assert (len(self.vocab), self.layer1_size) == self.syn0.shape
        with utils.smart_open(fname, 'wb') as fout:
            fout.write(utils.to_utf8("%s %s\n" % self.syn0.shape))
            # store in sorted order: most frequent words at the top
            for word, vocab in sorted(iteritems(self.vocab), key=lambda item: -item[1].count):
                row = self.syn0[vocab.index]
                if binary:
                    fout.write(word + b" " + row.tostring())
                else:
                    fout.write("%s %s\n" % (word, ' '.join("%f" % val for val in row)))


    def __getitem__(self, word):
        """
        Return a feature's representations in vector space, as a 1D numpy array.

        """
        return self.syn0[self.vocab[word].index]


    def __contains__(self, word):
        return word in self.vocab


    def __str__(self):
        return "Feat2Vec(vocab=%s, size=%s, alpha=%s)" % (len(self.index2feat), self.layer1_size, self.alpha)

