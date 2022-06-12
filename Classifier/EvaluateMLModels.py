import os

from lazypredict.Supervised import LazyClassifier
from UtilFunctions import createDirIfNotExist
import pandas as pd
import fasttext
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
strEL=' _EL_ '

def getVectorArr(model,strInput):
    strInput=strInput.replace('\n',strEL)
    vectorItem = model.get_sentence_vector(strInput)
    # strVector = ' '.join(map(str, vectorItem))
    return vectorItem
def convertListTextToListVector(model,lstText):
    lstVector=[]
    for item in lstText:
        vectorItem=getVectorArr(model,item)
        lstVector.append(vectorItem)
    return lstVector

fopResultPragFormer='/home/hungphd/git/Open_OMP/repResult/'
fopResultAllMLApproaches='/home/hungphd/git/Open_OMP/ensembleMLApproaches/'
createDirIfNotExist(fopResultPragFormer)
createDirIfNotExist(fopResultAllMLApproaches)
lstConfigs=['directive','private','reduction']
strBert='bert'
fpItemOverallStatistics = fopResultAllMLApproaches + 'overallStatistics.xlsx'
writer = pd.ExcelWriter(fpItemOverallStatistics,
                            engine='xlsxwriter')
for config in lstConfigs:
    dictAllPredictions = {}
    dictOracles={}
    fopStatisticsForProblems=fopResultAllMLApproaches+config+'/'
    createDirIfNotExist(fopStatisticsForProblems)

    fpItemOverallPrediction = fopStatisticsForProblems + 'overallPredictions.csv'
    dictAllPredictions[strBert]={}
    for foldIndex in range(1,11):
        nameSubFolder='{}/fold-{}/'.format(config,foldIndex)
        fpItemCSV=fopResultPragFormer+nameSubFolder+'data.csv'
        fpItemModelStatistics=fopResultPragFormer+nameSubFolder+'ensemble_models.csv'
        fpItemModelPredictions = fopResultPragFormer + nameSubFolder + 'ensemble_predictions.csv'
        fpItemBertPrediction = fopResultPragFormer + nameSubFolder + 'saved_weights_full_results.txt'
        pdEnsembleMLs = pd.read_csv(fpItemModelPredictions)

        if foldIndex==1:
            pdData=pd.read_csv(fpItemCSV)
            for i in range(0,len(pdData)):
                strKey=pdData['id'][i]
                val=pdData['label'][i]
                dictOracles[strKey]=val
            for key in dictOracles.keys():
                dictAllPredictions[strBert][key]=-1

            for i in range(0, len(pdEnsembleMLs.columns)):
                colName = pdEnsembleMLs.columns[i]
                if colName == 'realId' or colName == 'Unnamed: 0':
                    continue
                dictAllPredictions[colName] = {}
                for key in dictOracles.keys():
                    dictAllPredictions[colName][key] = -1

        lstKeysEnsemble=pdEnsembleMLs['realId'].tolist()
        for i in range(0,len(pdEnsembleMLs.columns)):
            colName=pdEnsembleMLs.columns[i]
            if colName=='realId' or colName=='Unnamed: 0':
                continue
            lstValueColumn=pdEnsembleMLs[colName].tolist()
            for j in range(0,len(lstKeysEnsemble)):
                dictAllPredictions[colName][lstKeysEnsemble[j]]=lstValueColumn[j]
                # if lstKeysEnsemble[j] == 'jhuber6_ompBLAS_parallel_for_simd_misc_messages.c_85':
                #     print('fold {}'.format(foldIndex))
                #     input('abc')

        f1=open(fpItemBertPrediction,'r')
        arrBertPrediction=f1.read().strip().split('\n')
        f1.close()
        for i in range(0,len(arrBertPrediction)):
            arrTabItems=arrBertPrediction[i].split('\t')
            # if len(arrTabItems)>=3:
            strKey=lstKeysEnsemble[i]
            # if strKey=='jhuber6_ompBLAS_parallel_for_simd_misc_messages.c_85':
            #     print('fold {}'.format(foldIndex))
            #     input('abc')
            val=(int) (arrTabItems[1])
            dictAllPredictions[strBert][strKey]=val
        print('end {}'.format(nameSubFolder))
    dfAllResults=pd.DataFrame()
    dfAllResults=dfAllResults.assign(realId=list(dictOracles.keys()))
    dfAllResults=dfAllResults.assign(label=list(dictOracles.values()))

    lstOracles = list(dictOracles.values())
    lstStrAccuracy=['"Model",Accuracy,Precison,Recall,F1']
    for keyModel in dictAllPredictions.keys():
        dictKeyModel=dictAllPredictions[keyModel]
        lstValueColumn=[]
        index=0
        for key2 in dictOracles.keys():
            # print('{} {} {}'.format(len(dictKeyModel.keys()),keyModel, key2))
            lstValueColumn.append(dictKeyModel[key2])
        strMetric='"{}",{},{},{},{}'.format(keyModel,accuracy_score(lstOracles,lstValueColumn),accuracy_score(lstOracles,lstValueColumn),precision_score(lstOracles,lstValueColumn),recall_score(lstOracles,lstValueColumn),f1_score(lstOracles,lstValueColumn))
        lstStrAccuracy.append(strMetric)
        dfAllResults[keyModel]=lstValueColumn
    dfAllResults.to_csv(fpItemOverallPrediction,index=False)
    f1=open(fpItemOverallStatistics.replace('.xlsx','.csv'),'w')
    f1.write('\n'.join(lstStrAccuracy))
    f1.close()
    pdItemStat=pd.read_csv(fpItemOverallStatistics.replace('.xlsx','.csv'))

    dfAllResults.to_excel(writer, sheet_name='{}'.format(config))
writer.save()








