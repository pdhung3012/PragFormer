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

def findScopeOfForLoop(jsonObject,beginLineFor,lstFors):
    if jsonObject['startLine']==beginLineFor and jsonObject['type']=='for_statement':
        lstFors.append(jsonObject)
    if len(lstFors)==0 and jsonObject['endLine']>=beginLineFor and 'children' in jsonObject.keys():
        lstChildren=jsonObject['children']
        for i in range(0,len(lstChildren)):
            findScopeOfForLoop(lstChildren[i],beginLineFor,lstFors)
def getLeftStrip(strInput):

    lsStrip=[]
    for ch in strInput:
        # print(ch)
        if ch.isspace():
            # print('go here')
            lsStrip.append(ch)
        else:
            break
    # return count
    return ''.join(lsStrip)
# def getCountTabStr(count):
#     lstStr=[]
#     for k in range(0,count):
#         lstStr.append('\t')
#     return '\t'.join(lstStr)

fopData='/home/hungphd/'
fopGithub='/home/hungphd/git/'
fopBuildFolder=fopData+'build-tree-sitter/'
fpLanguageSo=fopBuildFolder+'my-languages.so'
fopInputDataset='/home/hungphd/git/Open_OMP/'
# fopOutputPickle='/home/hungphd/git/Open_OMP/pickle/'
fopDatabaseCodeLocation=fopInputDataset+'database/'
fpJsonDatabase=fopInputDataset+'database.json'
fpLogAugmentStatus=fopInputDataset+'log_augment.txt'
fopParallelInput=fopInputDataset+'Parallel_Quazi/'
fopParallelOutput=fopInputDataset+'Parallel_Augmented/'
createDirIfNotExist(fopParallelOutput)
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
f1=open(fpLogAugmentStatus,'w')
f1.write('')
f1.close()
f1=open(fopInputDataset+'dummyMethods.txt','r')
strDummyMethod=f1.read().strip()
f1.close()

indexParallel=0
for key in jsonInput.keys():
    index=index+1
    itemJson = jsonInput[key]
    arrFpItemCode = itemJson['code'].strip().split('/')
    fopItemCode = '/'.join(arrFpItemCode[:(len(arrFpItemCode) - 1)]) + '/'
    fnIdEntity = arrFpItemCode[len(arrFpItemCode) - 2]
    arrFpOriginal=itemJson['original'].strip().split('/')
    fnNameOfPureCodeFile=arrFpOriginal[len(arrFpOriginal)-1].split('.c:')[0]+'.c'
    beginLineInJson=(int)(arrFpOriginal[len(arrFpOriginal)-1].split('.c:')[1].split(':')[0])-1
    if fnNameOfPureCodeFile=='code.c' or fnNameOfPureCodeFile=='pragma.c':
        print(fnIdEntity+' issue here')
    fpItemPureCode=fopItemCode+fnNameOfPureCodeFile
    fpItemASTPureCode = fopItemCode + fnNameOfPureCodeFile.replace('.c','.ast.txt')

    fpItemParallelFileC=fopParallelInput+fnIdEntity+'.c'
    fpItemAugmentFileC = fopParallelOutput + fnIdEntity + '.c'
    # if fnIdEntity!='jrk_QuakeTM_sv_send.c_2':
    #     continue
    if os.path.isfile(fpItemParallelFileC):
        beginLineDummy=beginLineInJson
        endLineDummy=0
        try:
            f1 = open(fpItemASTPureCode, 'r')
            strItemAST=f1.read().strip()
            f1.close()
            jsonAST=ast.literal_eval(strItemAST)
            f1=open(fpItemPureCode,'r')
            strItemCode=f1.read().strip()
            f1.close()
            arrItemCode=strItemCode.split('\n')

            beginForLoop=beginLineInJson
            while(beginForLoop<len(arrItemCode)):
                strStrip=arrItemCode[beginForLoop].strip()
                if strStrip.startswith('for'):
                    break
                beginForLoop=beginForLoop+1

            lstFors=[]
            findScopeOfForLoop(jsonAST,beginForLoop,lstFors)
            indexParallel=indexParallel+1
            strItemLogStatus = '{}\t{}\t{}\tFailed'.format(fnIdEntity,beginLineInJson+1,beginForLoop+1)
            if len(lstFors)==1:
                endLineDummy=lstFors[0]['endLine']
                lstTabItems = []
                for j in range(0, lstFors[0]['startOffset']):
                    lstTabItems.append('\t')
                strTabItem = '\t'.join(lstTabItems)
                lstNewCode = []
                for j in range(0, len(arrItemCode)):
                    if j == beginLineDummy:
                        lstNewCode.append(strTabItem + 'dummyMethod1();')
                        lstNewCode.append(arrItemCode[j])
                    elif j == endLineDummy:
                        lstNewCode.append(arrItemCode[j])
                        lstNewCode.append(strTabItem + 'dummyMethod2();')
                    else:
                        lstNewCode.append(arrItemCode[j])
                lstNewCode.append('\n\n{}\n'.format(strDummyMethod))
                f1 = open(fpItemAugmentFileC, 'w')
                f1.write('\n'.join(lstNewCode))
                f1.close()
                endLineDummy += 2
                beginLineDummy += 1
                strItemLogStatus = '{}\t{}\t{}\t{}'.format(fnIdEntity, beginLineInJson + 1, beginLineDummy,
                                                           endLineDummy)

            else:
                endLineDummy=beginForLoop+1
                strStartLeftStrip=getLeftStrip(arrItemCode[beginLineDummy])
                while endLineDummy<len(arrItemCode):
                    if arrItemCode[endLineDummy].strip()!='{' and arrItemCode[endLineDummy].strip()!='' and len(getLeftStrip(arrItemCode[endLineDummy]))<=len(strStartLeftStrip):
                        break
                    endLineDummy+=1

                lstNewCode = []
                for j in range(0, len(arrItemCode)):
                    if j == beginLineDummy:
                        lstNewCode.append(strStartLeftStrip + 'dummyMethod1();')
                        lstNewCode.append(arrItemCode[j])
                    elif j == endLineDummy:
                        lstNewCode.append(arrItemCode[j])
                        lstNewCode.append(strStartLeftStrip + 'dummyMethod2();')
                    else:
                        lstNewCode.append(arrItemCode[j])
                lstNewCode.append('\n\n{}\n'.format(strDummyMethod))
                f1 = open(fpItemAugmentFileC, 'w')
                f1.write('\n'.join(lstNewCode))
                f1.close()
                endLineDummy += 2
                beginLineDummy += 1
                strItemLogStatus = '{}\t{}\t{}\t{} (adhoc) {}'.format(fnIdEntity, beginLineInJson + 1, beginLineDummy,
                                                           endLineDummy,len(strStartLeftStrip))


            lstStrJsonParseResults.append(strItemLogStatus)
        except:
            traceback.print_exc()
            pass
        # and (len(lstStrJsonParseResults) % 100 == 0 or index == len(jsonInput.keys()))
    if len(lstStrJsonParseResults)>0 and (len(lstStrJsonParseResults) % 100 == 0 or index == len(jsonInput.keys())):
        f1=open(fpLogAugmentStatus,'a')
        f1.write('\n'.join(lstStrJsonParseResults)+'\n')
        f1.close()
        lstStrJsonParseResults=[]
        print('Index {} {} {}'.format(index,indexParallel,fnIdEntity))
        # break






