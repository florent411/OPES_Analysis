#!/usr/bin/env python 

''' General tools '''

import numpy as np
from itertools import chain, combinations

def powerset(iterable):
    '''Determine the powerset. The powerset of a set S is the set of all subsets of S, including the empty set and S itself.'''
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)

    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def get_column_names(filename):
    with open(filename, "r") as file:
        first_line = file.readline()            
    column_names = first_line.strip().split(" ")[2:]

    return column_names

def kldiv(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def dalonso(v1, v2):
    """Alonso divergence (dA) - A Physically Meaningful Method for the Comparison of Potential Energy Functions
    
    Parameters
    ----------
    v1, v2 : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    v1_var, v2_var = v1.var(), v2.var()
    r = np.corrcoef(v1, v2)[0,1]

    return np.sqrt((v1_var + v2_var) * (1 - r**2))

def jsdiv(p, q):
    """Jensen–Shannon divergence for measuring the similarity between two probability distributions. 
    It is based on the Kullback–Leibler divergence, with differences such as symmetry and it always has a finite value.

    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    m = (p + q) / 2

    # Calculate the Kullback-Leibler divergences: D(P||M) and D(Q||M)
    Dpm = np.sum(np.where(p != 0, p * np.log(p / m), 0))
    Dqm = np.sum(np.where(q != 0, q * np.log(q / m), 0))

    return 0.5 * Dpm + 0.5 * Dqm