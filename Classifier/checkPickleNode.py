import os.path
import sys
sys.path.append("..")

import numpy as np
import torch
import pickle
import pandas as pd
import torch.nn as nn
from torch.nn import DataParallel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, AutoTokenizer
from transformers import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import argparse
from sklearn.utils.class_weight import compute_class_weight
import Classifier.train as trainer
from Classifier.predict import predict
import Classifier.global_parameters as gp
from Classifier.model import BERT_Arch
from Classifier.data_creator import *
from Classifier.tokenizer import *
from UtilFunctions import *
import ast
from sklearn.model_selection import train_test_split
from statistics import mean,median


fopInputDataset='/home/hungphd/git/Open_OMP/'
fopOutputPickle='/home/hungphd/git/Open_OMP/pickle/'
fopDatabaseCodeLocation=fopInputDataset+'database/'
fpJsonDatabase=fopInputDataset+'database.json'
createDirIfNotExist(fopOutputPickle)
from ForPragmaExtractor.global_parameters import PragmaForTuple

f1 = open(fpJsonDatabase, 'r')
strInputJson = f1.read().strip()
f1.close()
jsonInput = ast.literal_eval(strInputJson)
index=0
for key in jsonInput.keys():
    index=index+1
    itemJson = jsonInput[key]
    arrFpItemCode = itemJson['code'].strip().split('/')
    fopItemCode = '/'.join(arrFpItemCode[:(len(arrFpItemCode) - 1)]) + '/'
    fnCodeFile = arrFpItemCode[len(arrFpItemCode) - 2]

    fpItemCode = fopItemCode + 'code.c'
    fpItemPragma = fopItemCode + 'pragma.c'
    fpItemPickle = fopItemCode + 'code_pickle.pkl'
    yItem = 0
    f1 = open(fpItemCode, 'r')
    strCode = f1.read().strip()
    f1.close()
    f1 = open(fpItemPickle, 'rb')
    obj = pickle.load(f1)
    # print(type(obj))
    # print(str(obj))
    # print(str(obj.pragma.string))
    # print(obj.get_coord())
    try:
        print('{} for node okk {} {}'.format(index,type(obj.for_node),obj.inner_nodes))
        # print('pass through error')
    except:
        print('{} for failed'.format(index))
        pass
    # print(obj.inner_node)
    # break


