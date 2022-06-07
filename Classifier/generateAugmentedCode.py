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

def augmentByASTs(jsonObject,dictItemForLoops,dictBeginForToBeginSep):
    # and jsonObject['type'] == 'for_statement'
    # print('{} check {}'.format(jsonObject['startLine'],dictBeginForToBeginSep.keys()))
    if jsonObject['startLine'] in dictBeginForToBeginSep.keys() :
        # print('go here')
        key=dictBeginForToBeginSep[jsonObject['startLine']]
        itemForLoop=dictItemForLoops[key]
        if itemForLoop[1]<jsonObject['endLine']:
            itemForLoop[1]=jsonObject['endLine']
            itemForLoop[3]=jsonObject['startOffset']
    if 'children' in jsonObject.keys():
        lstChildren=jsonObject['children']
        for i in range(0,len(lstChildren)):
            augmentByASTs(lstChildren[i],dictItemForLoops,dictBeginForToBeginSep)
def augmentAdhoc(dictItemForLoops,arrItemCodes):
    for key in dictItemForLoops.keys():
        val=dictItemForLoops[key]
        if val[1]<val[0]:
            endLineDummy = val[0] + 1
            beginLineDummy=val[0]
            strStartLeftStrip = getLeftStrip(arrItemCodes[beginLineDummy])
            while endLineDummy < len(arrItemCodes):
                if arrItemCodes[endLineDummy].strip() != '{' and arrItemCodes[endLineDummy].strip() != '' and len(
                        getLeftStrip(arrItemCodes[endLineDummy])) <= len(strStartLeftStrip):
                    break
                endLineDummy += 1
            val[1]=endLineDummy
def augmentAllDicts(arrItemCodes,dictItemForLoops,fpItemOutputC,fpDebugOutputC,strTemplate):
    lineEndInclude = 0
    dictIncludeCount={}
    while (lineEndInclude < len(arrItemCodes)):
        strStrip = arrItemCodes[lineEndInclude].strip()
        if strStrip.startswith('#include'):
            break
        lineEndInclude += 1
    lstIncludes = []
    while (lineEndInclude < len(arrItemCodes)):
        strStrip = arrItemCodes[lineEndInclude].strip()
        if not strStrip.startswith('#include'):
            break
        lstIncludes.append(strStrip)
        if strStrip not in dictIncludeCount.keys():
            dictIncludeCount[strStrip] = 1
        else:
            dictIncludeCount[strStrip] = dictIncludeCount[strStrip] + 1
        lineEndInclude += 1
    lstOutputCode=[]
    for item in arrItemCodes:
        lstOutputCode.append([item])
    strStmtInclude=''
    if len(lstIncludes) == 0:
        lineEndInclude=0
        strStmtInclude='#include <stdio.h>\nint dummyMethod1();\nint dummyMethod2();\nint dummyMethod3();\nint dummyMethod4();'
        # input('go here {}'.format(fnName))
    else:
        strStmtInclude='int dummyMethod1();\nint dummyMethod2();\nint dummyMethod3();\nint dummyMethod4();'
    lstOutputCode[lineEndInclude].insert(0,strStmtInclude)
    for key in dictItemForLoops.keys():
        valItem=dictItemForLoops[key]
        lstTabItems=[]
        for j in range(0, valItem[3]):
            lstTabItems.append('\t')
        strTabItem = '\t'.join(lstTabItems)
        if valItem[2]:
            strBeginAug='{}{}'.format(strTabItem,'dummyMethod1();')
            strEndAug = '{}{}'.format(strTabItem,'dummyMethod2();')
        else:
            strBeginAug = '{}{}'.format(strTabItem, 'dummyMethod3();')
            strEndAug = '{}{}'.format(strTabItem, 'dummyMethod4();')
        if len(lstOutputCode[key])<=1:
            lstOutputCode[key].insert(0,strBeginAug)
        else:
            lstOutputCode[key].insert(len(lstOutputCode)-1,strBeginAug)
        if len(lstOutputCode[valItem[1]])<=1:
            lstOutputCode[valItem[1]].append(strEndAug)
        else:
            lstOutputCode[valItem[1]].insert(0,strEndAug)
        # lstOutputCode[valItem[1]] = '{}\n{}'.format(lstOutputCode[key],strEndAug)
    lstOutputCode.append([strTemplate])
    lstStrFinalCode=[]
    lstStrDebugCode=[]
    for listItem in lstOutputCode:
        strFinal='\n'.join(listItem)
        strDebug=' _ENDLINE_ '.join(listItem).replace('\n',' _ENDLINE_ ')
        lstStrFinalCode.append(strFinal)
        lstStrDebugCode.append(strDebug)

    f1=open(fpItemOutputC,'w')
    f1.write('\n'.join(lstStrFinalCode))
    f1.close()
    f1=open(fpDebugOutputC,'w')
    f1.write('\n'.join(lstStrDebugCode))
    f1.close()



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

def checkIfDictItemIsPragmaItem(dictItemForLoops,arrItemCodes):
    # print('aaa')
    for key in dictItemForLoops.keys():
        beginOfSeparateRegion=key
        beginForLoop=beginOfSeparateRegion
        while(beginForLoop<len(arrItemCodes)):
            strStrip=arrItemCodes[beginForLoop].strip()
            if strStrip.startswith('for'):
                break
            beginForLoop+=1
        if beginOfSeparateRegion==beginForLoop:
            dictItemForLoops[key][2]=False
        else:
            dictItemForLoops[key][2]=True
        dictItemForLoops[key][0]=beginForLoop

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
fopParallelOutputDebug=fopInputDataset+'Parallel_Augmented_debug/'
createDirIfNotExist(fopParallelOutput)
createDirIfNotExist(fopParallelOutputDebug)
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

dictFileAndLineAugment={}
dictCopyVersion={}
for key in jsonInput.keys():
    index=index+1
    itemJson = jsonInput[key]
    arrFpItemCode = itemJson['code'].strip().split('/')
    fopItemCode = '/'.join(arrFpItemCode[:(len(arrFpItemCode) - 1)]) + '/'
    fnIdEntity = arrFpItemCode[len(arrFpItemCode) - 2]
    fnRealIdWithoutCopy=fnIdEntity.split('.c_')[0]
    arrFpOriginal=itemJson['original'].strip().split('/')
    fnNameOfPureCodeFile=arrFpOriginal[len(arrFpOriginal)-1].split('.c:')[0]+'.c'
    beginLineInJson=(int)(arrFpOriginal[len(arrFpOriginal)-1].split('.c:')[1].split(':')[0])-1
    if fnNameOfPureCodeFile=='code.c' or fnNameOfPureCodeFile=='pragma.c':
        print(fnIdEntity+' issue here')
    if fnRealIdWithoutCopy not in dictFileAndLineAugment.keys():
        dictFileAndLineAugment[fnRealIdWithoutCopy]=[beginLineInJson]
        dictCopyVersion[fnRealIdWithoutCopy]=fopItemCode+'/'+fnNameOfPureCodeFile
    else:
        dictFileAndLineAugment[fnRealIdWithoutCopy].append(beginLineInJson)
print('{}'.format(len(dictFileAndLineAugment.keys())))
# input('abc ')

index=0
for keyFileName in dictFileAndLineAugment.keys():
    index+=1
    valListBeginLine=sorted(dictFileAndLineAugment[keyFileName])
    fpItemPureCode=dictCopyVersion[keyFileName]
    arrItemPureCode=fpItemPureCode.split('/')
    fpItemAST='/'.join(arrItemPureCode[:len(arrItemPureCode)-2])+'/'+arrItemPureCode[len(arrItemPureCode)-1].replace('.c','.ast.txt')
    fpItemOutputCFile=fopParallelOutput+keyFileName+'.c'
    fpItemDebugCFile = fopParallelOutputDebug + keyFileName + '.c'

    strMessage=''
    try:
        f1 = open(fpItemAST, 'r')
        strItemAST = f1.read().strip()
        f1.close()
        jsonAST = ast.literal_eval(strItemAST)
        f1 = open(fpItemPureCode, 'r')
        strItemCode = f1.read().strip()
        f1.close()
        arrItemCodes = strItemCode.split('\n')

        dictItemForLoops={}
        for item in valListBeginLine:
            dictItemForLoops[item]=[item,item-1,False,0]
        checkIfDictItemIsPragmaItem(dictItemForLoops,arrItemCodes)

        dictBeginForToBeginSep={}
        for key1 in dictItemForLoops.keys():
            key2=dictItemForLoops[key1][0]
            dictBeginForToBeginSep[key2]=key1
        # print(dictBeginForToBeginSep)
        augmentByASTs(jsonAST, dictItemForLoops, dictBeginForToBeginSep)
        augmentAdhoc(dictItemForLoops, arrItemCodes)
        augmentAllDicts(arrItemCodes,dictItemForLoops,fpItemOutputCFile,fpItemDebugCFile,strDummyMethod)
        strMessage='{}\t{}\t{}'.format(index,keyFileName,dictItemForLoops)
    except Exception as e:
        traceback.print_exc()
        strMessage='{}\t{}\t{}'.format(index,keyFileName,str(e))
        pass
    lstStrJsonParseResults.append(strMessage)
    print('End index {} {}'.format(index, keyFileName))
    if len(lstStrJsonParseResults) % 100 == 0 or index == len(jsonInput.keys()):
        f1=open(fpLogAugmentStatus,'a')
        f1.write('\n'.join(lstStrJsonParseResults)+'\n')
        f1.close()
        lstStrJsonParseResults=[]
        print('Save index {} {}'.format(index,keyFileName))
        # break

# indexParallel=0
# for key in jsonInput.keys():
#     index=index+1
#     itemJson = jsonInput[key]
#     arrFpItemCode = itemJson['code'].strip().split('/')
#     fopItemCode = '/'.join(arrFpItemCode[:(len(arrFpItemCode) - 1)]) + '/'
#     fnIdEntity = arrFpItemCode[len(arrFpItemCode) - 2]
#     arrFpOriginal=itemJson['original'].strip().split('/')
#     fnNameOfPureCodeFile=arrFpOriginal[len(arrFpOriginal)-1].split('.c:')[0]+'.c'
#     beginLineInJson=(int)(arrFpOriginal[len(arrFpOriginal)-1].split('.c:')[1].split(':')[0])-1
#     if fnNameOfPureCodeFile=='code.c' or fnNameOfPureCodeFile=='pragma.c':
#         print(fnIdEntity+' issue here')
#     fpItemPureCode=fopItemCode+fnNameOfPureCodeFile
#     fpItemASTPureCode = fopItemCode + fnNameOfPureCodeFile.replace('.c','.ast.txt')
#
#     fpItemParallelFileC=fopParallelInput+fnIdEntity+'.c'
#     fpItemAugmentFileC = fopParallelOutput + fnIdEntity + '.c'
#     # if fnIdEntity!='jrk_QuakeTM_sv_send.c_2':
#     #     continue
#     if os.path.isfile(fpItemParallelFileC):
#         beginLineDummy=beginLineInJson
#         endLineDummy=0
#         try:
#             f1 = open(fpItemASTPureCode, 'r')
#             strItemAST=f1.read().strip()
#             f1.close()
#             jsonAST=ast.literal_eval(strItemAST)
#             f1=open(fpItemPureCode,'r')
#             strItemCode=f1.read().strip()
#             f1.close()
#             arrItemCode=strItemCode.split('\n')
#
#             beginForLoop=beginLineInJson
#             while(beginForLoop<len(arrItemCode)):
#                 strStrip=arrItemCode[beginForLoop].strip()
#                 if strStrip.startswith('for'):
#                     break
#                 beginForLoop=beginForLoop+1
#
#             lstFors=[]
#             findScopeOfForLoop(jsonAST,beginForLoop,lstFors)
#             indexParallel=indexParallel+1
#             strItemLogStatus = '{}\t{}\t{}\tFailed'.format(fnIdEntity,beginLineInJson+1,beginForLoop+1)
#             if len(lstFors)==1:
#                 endLineDummy=lstFors[0]['endLine']
#                 lstTabItems = []
#                 for j in range(0, lstFors[0]['startOffset']):
#                     lstTabItems.append('\t')
#                 strTabItem = '\t'.join(lstTabItems)
#                 lstNewCode = []
#                 for j in range(0, len(arrItemCode)):
#                     if j == beginLineDummy:
#                         lstNewCode.append(strTabItem + 'dummyMethod1();')
#                         lstNewCode.append(arrItemCode[j])
#                     elif j == endLineDummy:
#                         lstNewCode.append(arrItemCode[j])
#                         lstNewCode.append(strTabItem + 'dummyMethod2();')
#                     else:
#                         lstNewCode.append(arrItemCode[j])
#                 lstNewCode.append('\n\n{}\n'.format(strDummyMethod))
#                 f1 = open(fpItemAugmentFileC, 'w')
#                 f1.write('\n'.join(lstNewCode))
#                 f1.close()
#                 endLineDummy += 2
#                 beginLineDummy += 1
#                 strItemLogStatus = '{}\t{}\t{}\t{}'.format(fnIdEntity, beginLineInJson + 1, beginLineDummy,
#                                                            endLineDummy)
#
#             else:
#                 endLineDummy=beginForLoop+1
#                 strStartLeftStrip=getLeftStrip(arrItemCode[beginLineDummy])
#                 while endLineDummy<len(arrItemCode):
#                     if arrItemCode[endLineDummy].strip()!='{' and arrItemCode[endLineDummy].strip()!='' and len(getLeftStrip(arrItemCode[endLineDummy]))<=len(strStartLeftStrip):
#                         break
#                     endLineDummy+=1
#
#                 lstNewCode = []
#                 for j in range(0, len(arrItemCode)):
#                     if j == beginLineDummy:
#                         lstNewCode.append(strStartLeftStrip + 'dummyMethod1();')
#                         lstNewCode.append(arrItemCode[j])
#                     elif j == endLineDummy:
#                         lstNewCode.append(arrItemCode[j])
#                         lstNewCode.append(strStartLeftStrip + 'dummyMethod2();')
#                     else:
#                         lstNewCode.append(arrItemCode[j])
#                 lstNewCode.append('\n\n{}\n'.format(strDummyMethod))
#                 f1 = open(fpItemAugmentFileC, 'w')
#                 f1.write('\n'.join(lstNewCode))
#                 f1.close()
#                 endLineDummy += 2
#                 beginLineDummy += 1
#                 strItemLogStatus = '{}\t{}\t{}\t{} (adhoc) {}'.format(fnIdEntity, beginLineInJson + 1, beginLineDummy,
#                                                            endLineDummy,len(strStartLeftStrip))
#
#
#             lstStrJsonParseResults.append(strItemLogStatus)
#         except:
#             traceback.print_exc()
#             pass
#         # and (len(lstStrJsonParseResults) % 100 == 0 or index == len(jsonInput.keys()))
#     if len(lstStrJsonParseResults)>0 and (len(lstStrJsonParseResults) % 100 == 0 or index == len(jsonInput.keys())):
#         f1=open(fpLogAugmentStatus,'a')
#         f1.write('\n'.join(lstStrJsonParseResults)+'\n')
#         f1.close()
#         lstStrJsonParseResults=[]
#         print('Index {} {} {}'.format(index,indexParallel,fnIdEntity))
#         # break






