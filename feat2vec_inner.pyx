#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8
#
# Copyright (C) 2014 Yi Yang <yangyiycc@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import cython
import numpy as np
cimport numpy as np

from libc.math cimport exp
from libc.string cimport memset

cdef extern from "voidptr.h":
    void* PyCObject_AsVoidPtr(object obj)

from scipy.linalg.blas import fblas

REAL = np.float32
ctypedef np.float32_t REAL_t

DEF MAX_INSTANCE_LEN = 10000

ctypedef void (*scopy_ptr) (const int *N, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*dsdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*snrm2_ptr) (const int *N, const float *X, const int *incX) nogil
ctypedef void (*sscal_ptr) (const int *N, const float *alpha, const float *X, const int *incX) nogil

ctypedef unsigned long long (*fast_instance_ptr) (
    const int negative, np.uint32_t *table, unsigned long long table_len,
    unsigned long long start_offset, REAL_t *syn0, REAL_t *syn1neg, 
    const int size, const np.uint32_t feat_index,
    const np.uint32_t feat2_index, const REAL_t alpha, REAL_t *work,
    unsigned long long next_random, unsigned long vocab_len) nogil

cdef scopy_ptr scopy=<scopy_ptr>PyCObject_AsVoidPtr(fblas.scopy._cpointer)  # y = x
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)  # y += alpha * x
cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y)
cdef dsdot_ptr dsdot=<dsdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # double = dot(x, y)
cdef snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(fblas.snrm2._cpointer)  # sqrt(x^2)
cdef sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer) # x = alpha * x
cdef fast_instance_ptr fast_instance

DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6

cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0


cdef unsigned long long fast_instance0(
    const int negative, np.uint32_t *table, unsigned long long table_len, 
    unsigned long long start_offset, REAL_t *syn0, REAL_t *syn1neg, const int size, 
    const np.uint32_t feat_index,
    const np.uint32_t feat2_index, const REAL_t alpha, REAL_t *work,
    unsigned long long next_random, unsigned long vocab_len) nogil:

    cdef long long a
    cdef long long row1 = feat2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label
    cdef np.uint32_t target_index
    cdef int d

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = feat_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == feat_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f = <REAL_t>dsdot(&size, &syn0[row1], &ONE, &syn1neg[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        saxpy(&size, &g, &syn0[row1], &ONE, &syn1neg[row2], &ONE)

    saxpy(&size, &ONEF, work, &ONE, &syn0[row1], &ONE)

    return next_random


cdef unsigned long long fast_instance1(
    const int negative, np.uint32_t *table, unsigned long long table_len,
    unsigned long long start_offset,
    REAL_t *syn0, REAL_t *syn1neg, const int size, const np.uint32_t feat_index,
    const np.uint32_t feat2_index, const REAL_t alpha, REAL_t *work,
    unsigned long long next_random, unsigned long vocab_len) nogil:

    cdef long long a
    cdef long long row1 = feat2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, lg, reg
    cdef np.uint32_t target_index
    cdef int d, m

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = feat_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len + start_offset]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == feat_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f = <REAL_t>sdot(&size, &syn0[row1], &ONE, &syn1neg[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        saxpy(&size, &g, &syn0[row1], &ONE, &syn1neg[row2], &ONE)
    
    saxpy(&size, &ONEF, work, &ONE, &syn0[row1], &ONE)

    return next_random


cdef unsigned long long fast_instance2(
    const int negative, np.uint32_t *table, unsigned long long table_len,
    unsigned long long start_offset,
    REAL_t *syn0, REAL_t *syn1neg, const int size, const np.uint32_t feat_index,
    const np.uint32_t feat2_index, const REAL_t alpha, REAL_t *work,
    unsigned long long next_random, unsigned long vocab_len) nogil:

    cdef long long a
    cdef long long row1 = feat2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label
    cdef np.uint32_t target_index
    cdef int d

    for a in range(size):
        work[a] = <REAL_t>0.0

    for d in range(negative+1):

        if d == 0:
            target_index = feat_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == feat_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f = <REAL_t>0.0
        for a in range(size):
            f += syn0[row1 + a] * syn1neg[row2 + a]
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        for a in range(size):
            work[a] += g * syn1neg[row2 + a]
        for a in range(size):
            syn1neg[row2 + a] += g * syn0[row1 + a]

    for a in range(size):
        syn0[row1 + a] += work[a]

    return next_random


def train_instance(model, instance, alpha, _work):
    cdef int negative = model.negative

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *work
    cdef REAL_t _alpha = alpha
    cdef int size = model.layer1_size

    cdef int codelens[MAX_INSTANCE_LEN]
    cdef np.uint32_t indexes[MAX_INSTANCE_LEN]
    cdef np.uint32_t start_offsets[MAX_INSTANCE_LEN]
    cdef np.uint32_t table_lens[MAX_INSTANCE_LEN]

    cdef int instance_len

    cdef int i, j, k
    cdef long result = 0

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *table
    cdef unsigned long long table_len
    cdef unsigned long long next_random

    cdef unsigned long vocab_len = len(model.vocab)
    
    syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
    table = <np.uint32_t *>(np.PyArray_DATA(model.table))
    next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    instance_len = <int>min(MAX_INSTANCE_LEN, len(instance))

    for i in range(instance_len):
        feat = instance[i]
        if feat is None:
            codelens[i] = 0
        else:
            indexes[i] = feat.index
            if model.bow:
                start_offsets[i] = 0
                table_lens[i] = len(model.table)
            else:
                start_offsets[i] = model.table_offsets[feat.template]
                table_lens[i] = model.table_offsets[feat.template+1] - start_offsets[i]
            codelens[i] = 1
            result += 1

    # release GIL & train on the instance
    with nogil:
        for i in range(instance_len):
            if codelens[i] == 0:
                continue
            for j in range(instance_len):
                if j == i or codelens[j] == 0:
                    continue
                start_offset = start_offsets[j]
                table_len = table_lens[j]
                next_random = fast_instance(negative, table, table_len, start_offset, syn0, syn1neg, size, indexes[j], indexes[i], _alpha, work, next_random, vocab_len)

    return result


def init():
    """
    Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized
    into table EXP_TABLE.

    """
    global fast_instance

    cdef int i
    cdef float *x = [<float>10.0]
    cdef float *y = [<float>0.01]
    cdef float expected = <float>0.1
    cdef int size = 1
    cdef double d_res
    cdef float *p_res

    # build the sigmoid table
    for i in range(EXP_TABLE_SIZE):
        EXP_TABLE[i] = <REAL_t>exp((i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        EXP_TABLE[i] = <REAL_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))

    # check whether sdot returns double or float
    d_res = dsdot(&size, x, &ONE, y, &ONE)
    p_res = <float *>&d_res
    if (abs(d_res - expected) < 0.0001):
        fast_instance = fast_instance0
        return 0  # double
    elif (abs(p_res[0] - expected) < 0.0001):
        fast_instance = fast_instance1
        return 1  # float
    else:
        # neither => use cython loops, no BLAS
        # actually, the BLAS is so messed up we'll probably have segfaulted above and never even reach here
        fast_instance = fast_instance2
        return 2

FAST_VERSION = init()  # initialize the module
