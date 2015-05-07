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

def save_embeddings(feature_file, output_file, bow=False, feature_template_file=None, template_prefix_file=None, dim=50, epoch=1, negative=5, workers=4, binary_output=False):
    instances, feats = get_instances(feature_file)
    if feature_template_file is not None:
        template_dict = get_template_dict(feature_template_file)
    elif template_prefix_file is not None:
        template_dict = get_template_dict(template_prefix_file, feats)

    model = feat2vec.Feat2Vec(instances, size=dim, epoch=epoch, workers=workers, negative=negative, bow=bow, template_dict=template_dict)
    model.save_word2vec_format(output_file, binary=binary_output)

def get_template_dict(template_file, feats=None):
    template_dict = {}
    if feats is None:
        with open(template_file, "rb") as f:
            for line in f:
                parts = line.strip().split()
                template_dict[parts[0]] = int(parts[1])
    else:
        prefices = []
        with open(template_file, "rb") as f:
            for line in f:
                prefices.append(line.strip())
        for feat in feats:
            for i,prefix in enumerate(prefices):
                if feat.startswith(prefix):
                    template_dict[feat] = i
                    continue
    return template_dict

def get_instances(feature_file):
    instances = []
    feats = set()
    with open(feature_file, "rb") as f:
        for line in f:
            parts = line.strip().split()
            feats.update(set(parts))
            instances.append(parts)
    return instances, feats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_file", help="feature file, each line corresponds to an instance")
    parser.add_argument("output_file", help="output file for feature embeddings")
    parser.add_argument("--bow", type=int, default=0, help="bag of words representation, default is 0 (structured representation)")
    parser.add_argument("--feature_template_file", help="feature-template mappings file, only work if --bow=0")
    parser.add_argument("--template_prefix_file", help="template prefices file, only work if --bow=0")
    parser.add_argument("--dim", type=int, default=50, help="embedding size, default is 50")
    parser.add_argument("--epoch", type=int, default=1, help="training epoch, default is 1")
    parser.add_argument("--negative", type=int, default=5, help="number of negative samplers, default is 5")
    parser.add_argument("--workers", type=int, default=4, help="number of threads, default is 4")
    parser.add_argument("--binary_output", type=int, default=1, help="binary output or not, default is 1 (binary)")
    args = parser.parse_args()

    if args.bow == 1: bow = True
    else: bow = False
    if bow == False and args.feature_template_file is None and args.template_prefix_file is None:
        print "please provide --feature_template_file argument or --template_prefix_file"
        sys.exit()
    dim = args.dim
    epoch = args.epoch
    negative = args.negative
    workers = args.workers
    if args.binary_output == 1: binary = True
    else: binary = False

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info('begin logging')

    save_embeddings(args.feature_file, args.output_file, bow=bow, feature_template_file=args.feature_template_file, template_prefix_file=args.template_prefix_file, dim=dim, epoch=epoch, negative=negative, workers=workers, binary_output=binary)

    logger.info('end logging')
