#!/usr/bin/env python

'''
A demo for saving feature embeddings

Author: Yi Yang
Email: yangyiycc@gmail.com
'''

import sys, logging

import gensim
from gensim import utils
sys.path.append('feat2vec')
import feat2vec

import argparse

logger = logging.getLogger("feat2vec.save_embeddings")

def save_embeddings(feature_file, output_file, bow=False, feature_dict_file=None, dim=50, epoch=1, negative=5, workers=4, binary_output=False):
    instances = get_instances(feature_file)
    template_dict = get_template_dict(feature_dict_file)
    model = feat2vec.Feat2Vec(instances, size=dim, epoch=epoch, workers=workers, negative=negative, bow=bow, template_dict=template_dict)
    model.save_word2vec_format(output_file, binary=binary_output)

def get_template_dict(feature_dict_file):
    template_dict = {}
    with open(feature_dict_file, "rb") as f:
        for line in f:
            parts = line.strip().split()
            template_dict[parts[0]] = int(parts[1])
    return template_dict

def get_instances(feature_file):
    instances = []
    with open(feature_file, "rb") as f:
        for line in f:
            instances.append(line.strip().split())
    return instances

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_file", help="feature file, each line corresponds to an instance")
    parser.add_argument("output_file", help="output file for feature embeddings")
    parser.add_argument("--bow", type=int, help="bag of words representation, default is 0 (structured representation)")
    parser.add_argument("--feature_dict_file", help="feature-templateID mappings file, only work if --bow=0")
    parser.add_argument("--dim", type=int, help="embedding size, default is 50")
    parser.add_argument("--epoch", type=int, help="training epoch, default is 1")
    parser.add_argument("--negative", type=int, help="number of negative samplers, default is 5")
    parser.add_argument("--workers", type=int, help="number of threads, default is 4")
    parser.add_argument("--binary_output", help="number of threads, default is 4")
    args = parser.parse_args()

    if args.bow == 1: bow = True
    else: bow = False
    if bow == True and args.feature_dict_file is None:
        print "please provide --feature_dict_file argument"
        sys.exit()
    if args.dim is not None: dim = args.dim
    else: dim = 50
    if args.epoch is not None: epoch = args.epoch
    else: epoch = 1
    if args.negative is not None: negative = args.negative
    else: negative = 5
    if args.workers is not None: workers = args.workers
    else: workers = 4
    if args.binary_output == 1: binary = True
    else: binary = False

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info('begin logging')

    save_embeddings(args.feature_file, args.output_file, bow=bow, feature_dict_file=args.feature_dict_file, dim=dim, epoch=epoch, negative=negative, workers=workers, binary_output=binary)

    logger.info('end logging')
