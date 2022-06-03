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
fopOutputScript='/home/hungphd/git/Open_OMP/script/'
fopDatabaseCodeLocation=fopInputDataset+'database/'
fpJsonDatabase=fopInputDataset+'database.json'
createDirIfNotExist(fopOutputScript)
from ForPragmaExtractor.global_parameters import PragmaForTuple

f1 = open(fpJsonDatabase, 'r')
strInputJson = f1.read().strip()
f1.close()
jsonInput = ast.literal_eval(strInputJson)
index=0
lstProjectDownloads=set()
lstProjectNoRepos=set()
lstAllOriginals=[]
setRepos=set()
strUsername='pdhung3012'
strPassword='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'
dictFileNameAndRepo={}
for key in jsonInput.keys():
    index=index+1
    itemJson = jsonInput[key]
    arrFpItemCode = itemJson['code'].strip().split('/')
    fopItemCode = '/'.join(arrFpItemCode[:(len(arrFpItemCode) - 1)]) + '/'
    fnCodeFile = arrFpItemCode[len(arrFpItemCode) - 2]
    fpOriginal=itemJson['original'].strip().replace('/home/reemh/CLPP/github-clone-all/repos_final/','')
    arrFpOriginal=itemJson['original'].strip().replace('/home/reemh/CLPP/github-clone-all/repos_final/','').split('/')
    nameAuthor=arrFpOriginal[0]
    nameRepo=arrFpOriginal[1]
    nameFile=arrFpOriginal[len(arrFpOriginal)-1]
    strKeyDictAna='{}__{}'.format(nameRepo,nameFile)
    if strKeyDictAna not in dictFileNameAndRepo.keys():
        dictFileNameAndRepo[strKeyDictAna]=[fpOriginal]
    else:
        dictFileNameAndRepo[strKeyDictAna].append(fpOriginal)
    strKey='{}__{}'.format(nameAuthor,nameRepo)
    strRemoveFolder='rm -rf {}__{}'.format(nameAuthor,nameRepo)
    strMkDir = 'mkdir {}__{}'.format(nameAuthor, nameRepo)
    strCdDir = 'cd {}__{}'.format(nameAuthor, nameRepo)
    strOutDir = 'cd ..'.format(nameAuthor, nameRepo)
    strScriptDownRepo='git clone https://{}:{}@github.com/{}/{}.git'.format(strUsername,strPassword,nameAuthor,nameRepo)
    strScriptDownNoCheckout = 'git clone --no-checkout https://{}:{}@github.com/{}/{}.git'.format(strUsername,strPassword,nameAuthor,nameRepo)
    # print(strScriptDownRepo)
    lstProjectDownloads.add(strScriptDownRepo)

    if strKey not in setRepos:
        setRepos.add(strKey)
        lstProjectNoRepos.add('{}\n{}\n{}\n{}\n{}'.format(strRemoveFolder,strMkDir,strCdDir,strScriptDownNoCheckout,strOutDir))
    lstAllOriginals.append('/'.join(arrFpOriginal))
    # print(obj.inner_node)
    # break
    # setRepos.add(nameRepo)

print('num of repo names {}'.format(len(setRepos)))
f1=open(fopOutputScript+'scriptCloneAll.sh','w')
f1.write('\n'.join(list(lstProjectDownloads)))
f1.close()
f1=open(fopOutputScript+'scriptCloneNoCheckout.sh','w')
f1.write('\n'.join(list(lstProjectNoRepos)))
f1.close()
f1=open(fopOutputScript+'allFiles.txt','w')
f1.write('\n'.join(list(lstAllOriginals)))
f1.close()
f1=open(fopOutputScript+'setRepos.txt','w')
f1.write('\n'.join(list(setRepos)))
f1.close()

lstKeys=sorted(dictFileNameAndRepo, key=lambda k: len(dictFileNameAndRepo[k]), reverse=True)
lstStrSorted=[]
for key in lstKeys:
    lstStrSorted.append('{}\t{}\t{}'.format(key,len(dictFileNameAndRepo[key]),','.join(dictFileNameAndRepo[key])))
f1=open(fopOutputScript+'dictFilenameRepoDup.txt','w')
f1.write('\n'.join(lstStrSorted))
f1.close()


listRepos=list(setRepos)
fopInputHeadFile='/home/hungphd/media/dataPapersExternal/git_parallels/cloneNoCheckout/PART1/PART2/.git/HEAD'
lstWGetCommand=[]
lstExtractCommand=[]
fopOutputZip='/home/hungphd/media/dataPapersExternal/git_parallels/cloneZip/'
fopOutputExtract='/home/hungphd/media/dataPapersExternal/git_parallels/cloneExtract/'
createDirIfNotExist(fopOutputExtract)
lstRemovedProjects=[]
dictReponameAndUsernames={}
for i in range(0,len(listRepos)):
    itemRepo=listRepos[i]
    nameAuthor=itemRepo.split('__')[0]
    nameRepo = itemRepo.split('__')[1]
    strLink='https://github.com/{}/{}/'.format(nameAuthor,nameRepo)
    if nameRepo not in dictReponameAndUsernames.keys():
        dictReponameAndUsernames[nameRepo]=[strLink]
    else:
        dictReponameAndUsernames[nameRepo].append(strLink)
    fpItemHead=fopInputHeadFile.replace('PART1',itemRepo).replace('PART2',nameRepo)
    strBranchName='master'
    isDownload=False
    fpZipLocation = fopOutputZip + itemRepo + '.zip'

    if os.path.isfile(fpItemHead):
        f1=open(fpItemHead,'r')
        strBranchName=f1.read().strip().replace('ref: refs/heads/','').strip()
        f1.close()
        isDownload=True
    # / LLNL / AutoParBench / archive / refs / heads / master.zip
    strZipDownload = 'https://github.com/{}/{}/archive/refs/heads/{}.zip'.format(nameAuthor, nameRepo, strBranchName)
    if isDownload:
        commandDownload = 'wget {} -O {}'.format(strZipDownload, fpZipLocation)
        fopLocationToExtract = fopOutputExtract + itemRepo + '/'
        createDirIfNotExist(fopLocationToExtract)
        extractComment='unzip {} -d {}'.format(fpZipLocation,fopLocationToExtract)
    else:
        lstRemovedProjects.append('https://github.com/{}/{}/'.format(nameAuthor,nameRepo))
        commandDownload = '#wget {} -O {}'.format(strZipDownload, fpZipLocation)
        extractComment = '#unzip {}'.format(fpZipLocation)
    lstWGetCommand.append(commandDownload)
    lstExtractCommand.append(extractComment)

lstKeys=sorted(dictReponameAndUsernames, key=lambda k: len(dictReponameAndUsernames[k]), reverse=True)
lstStrSorted=[]
for key in lstKeys:
    lstStrSorted.append('{}\t{}\t{}'.format(key,len(dictReponameAndUsernames[key]),','.join(dictReponameAndUsernames[key])))
f1=open(fopOutputScript+'dictRepos.txt','w')
f1.write('\n'.join(lstStrSorted))
f1.close()


f1=open(fopOutputScript+'scriptDownloadZip.sh','w')
f1.write('\n'.join(lstWGetCommand))
f1.close()
f1=open(fopOutputScript+'scriptExtractZip.sh','w')
f1.write('\n'.join(lstExtractCommand))
f1.close()
f1=open(fopOutputScript+'setRemoveProjects.txt','w')
f1.write('\n'.join(lstRemovedProjects))
f1.close()







