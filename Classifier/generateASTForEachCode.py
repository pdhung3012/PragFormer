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
from tree_sitter import Language, Parser

fopData='/home/hungphd/'
fopGithub='/home/hungphd/git/'
fopBuildFolder=fopData+'build-tree-sitter/'
fpLanguageSo=fopBuildFolder+'my-languages.so'
fopInputDataset='/home/hungphd/git/Open_OMP/'
# fopOutputPickle='/home/hungphd/git/Open_OMP/pickle/'
fopDatabaseCodeLocation=fopInputDataset+'database/'
fpJsonDatabase=fopInputDataset+'database.json'
fpLogCCodeParseStatus=fopInputDataset+'log_asttreesitter.txt'
# createDirIfNotExist(fopOutputPickle)
from ForPragmaExtractor.global_parameters import PragmaForTuple
from LibForHandleASTTreeSitter import *
CPP_LANGUAGE = Language(fpLanguageSo, 'cpp')
parser = Parser()
parser.set_language(CPP_LANGUAGE)

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
    if fnNameOfPureCodeFile=='code.c' or fnNameOfPureCodeFile=='pragma.c':
        print(fnCodeFile+' issue here')
    continue
    fpItemPureCode=fopItemCode+fnNameOfPureCodeFile
    fpItemASTPureCode = fopItemCode + fnNameOfPureCodeFile.replace('.c','.ast.txt')


    fpItemForLoop = fopItemCode + 'code.c'
    fpItemASTForLoop = fopItemCode + 'code.ast.txt'
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
    try:
        tree = parser.parse(bytes(strForLoop, 'utf8'))
        cursor = tree.walk()
        node = cursor.node
        arrForLoop=strForLoop.split()
        listId=[]
        dictJson = walkTreeAndReturnJSonObject(node, arrForLoop,listId)
        f1 = open(fpItemASTForLoop, 'w')
        f1.write(str(dictJson))
        f1.close()
        isParseForLoopOK=True
    except:
        traceback.print_exc()
        pass
    try:
        tree = parser.parse(bytes(strPureCode, 'utf8'))
        cursor = tree.walk()
        node = cursor.node
        arrPureCode=strPureCode.split()
        listId=[]
        dictJson = walkTreeAndReturnJSonObject(node, arrPureCode,listId)
        f1 = open(fpItemASTPureCode, 'w')
        f1.write(str(dictJson))
        f1.close()
        isParseAllCodeOK=True
    except:
        traceback.print_exc()
        pass
    lstStrJsonParseResults.append('{}\t{}\t{}'.format(fnCodeFile, isParseForLoopOK, isParseAllCodeOK))

    if len(lstStrJsonParseResults)%100==0 or len(lstStrJsonParseResults)==len(jsonInput.keys()):
        f1=open(fpLogCCodeParseStatus,'a')
        f1.write('\n'.join(lstStrJsonParseResults)+'\n')
        f1.close()
        lstStrJsonParseResults=[]
        print('index {} {}'.format(index,fnNameOfPureCodeFile))





