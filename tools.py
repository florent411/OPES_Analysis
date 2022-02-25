#!/usr/bin/env python 

''' General tools '''

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
