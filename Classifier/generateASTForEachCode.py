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
# fopOutputPickle='/home/hungphd/git/Open_OMP/pickle/'
fopDatabaseCodeLocation=fopInputDataset+'database/'
fpJsonDatabase=fopInputDataset+'database.json'
fpLogCCodeParseStatus=fopInputDataset+'log_asttreesitter.txt'
# createDirIfNotExist(fopOutputPickle)
from ForPragmaExtractor.global_parameters import PragmaForTuple

f1 = open(fpJsonDatabase, 'r')
strInputJson = f1.read().strip()
f1.close()
jsonInput = ast.literal_eval(strInputJson)
index=0
lstStrJsonParseResults=[]
f1=open(fpLogCCodeParseStatus,'w')
f1.write('')
f1.close()
for key in jsonInput.keys():
    index=index+1
    itemJson = jsonInput[key]
    arrFpItemCode = itemJson['code'].strip().split('/')
    fopItemCode = '/'.join(arrFpItemCode[:(len(arrFpItemCode) - 1)]) + '/'
    fnCodeFile = arrFpItemCode[len(arrFpItemCode) - 2]
    arrFpOriginal=itemJson['original'].strip().split('/')
    fnNameOfPureCodeFile=arrFpOriginal[len(arrFpOriginal)-1].split('.c:')[0]+'.c'
    # if fnNameOfPureCodeFile=='code.c':
    #     print(fnCodeFile+' issue here')
    fpItemPureCode=fopItemCode+fnNameOfPureCodeFile


    fpItemForLoop = fopItemCode + 'code.c'
    fpItemPragma = fopItemCode + 'pragma.c'
    fpItemPickle = fopItemCode + 'code_pickle.pkl'
    f1 = open(fpItemForLoop, 'r')
    strForLoop = f1.read().strip()
    f1.close()
    f1 = open(fpItemPureCode, 'r')
    strPureCode = f1.read().strip()
    f1.close()
    isParseForLoopOK=False
    isParseAllCodeOK = False
    lstStrJsonParseResults.append('{}\t{}\t{}'.format(fnCodeFile,isParseForLoopOK,isParseAllCodeOK))
    if len(lstStrJsonParseResults)%1000==0 or len(lstStrJsonParseResults)==len(jsonInput.keys()):
        f1=open(fpLogCCodeParseStatus,'a')
        f1.write('\n'.join(lstStrJsonParseResults)+'\n')
        f1.close()
        lstStrJsonParseResults=[]
        print('index {} {}'.format(index,fnNameOfPureCodeFile))





