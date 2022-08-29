from utils import pkl_utils
import sys

sys.path.append("../")
import config
import pprint
import numpy as np
import collections, copy


# each column represents a type,alpha controls loss weight of its child types, return shape (ntype, ntype)
def create_prior(type_info, alpha=1.0):
    type2id, typeDict = pkl_utils.load(type_info)
    num_types = len(type2id)
    prior = np.zeros((num_types, num_types))
    for x in type2id.keys():
        tmp = np.zeros(num_types)
        tmp[type2id[x]] = 1.0
        for y in typeDict[x]:  # child nodes
            tmp[type2id[y]] = alpha
        prior[:, type2id[x]] = tmp
    return prior


def istopSon(s) -> bool:
    istop = False
    counter = collections.Counter(s)
    if counter['/'] == 1:
        istop = True
    return istop


def makeSonFindermatrix(type_info):
    type2id, typeDict = pkl_utils.load(type_info)
    num_types = len(type2id)
    prior = np.zeros((num_types, num_types))
    for x in type2id.keys():
        tmp = np.zeros(num_types)
        tmp[type2id[x]] = 1.0
        for y in typeDict[x]:  # child nodes
            tmp[type2id[y]] = 1.0
        prior[type2id[x], :] = tmp
    tmp = np.zeros(num_types)
    for typename in typeDict.keys():
        if istopSon(typename):
            tmp[type2id[typename]] = 1
    fatherNotin = copy.deepcopy(prior)
    for i in range(num_types):
        fatherNotin[i, i] = 0
    return prior, fatherNotin, tmp


if __name__ == '__main__':
    type_info = '.' + config.WIKIM_TYPE
    print(type_info)
    pprint.pprint(makeSonFindermatrix(type_info)[0][101, :])
