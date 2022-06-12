import os

from lazypredict.Supervised import LazyClassifier
from UtilFunctions import createDirIfNotExist
import pandas as pd
import fasttext
import numpy as np

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

def getListOfForLoopAndPragma(lstRealIds,lstLabels,fopDatabase,maxLength):
    lstCodes=[]
    lstPragmas=[]
    for i in range(0,len(lstRealIds)):
        fpItemCode=fopDatabase+lstRealIds[i]+'/code.c'
        f1=open(fpItemCode,'r')
        strItemCode=f1.read().strip()
        f1.close()
        if len(strItemCode)>maxLength:
            lenSubStr=maxLength
        else:
            lenSubStr=len(strItemCode)
        strSubStr=strItemCode[:lenSubStr]
        lstCodes.append(strSubStr)
        strPragma=''
        if lstLabels[i]==1:
            f1=open(fopDatabase+lstRealIds[i]+'/pragma.c','r')
            strPragma=f1.read().strip()
            f1.close()
        lstPragmas.append(strPragma)
    return lstCodes,lstPragmas



fopResultPragFormer='/home/hungphd/git/Open_OMP/repResult/'
fopDatabasePragFormer='/home/hungphd/git/Open_OMP/database/'
fopResultAllMLApproaches='/home/hungphd/git/Open_OMP/ensembleMLApproaches/'
createDirIfNotExist(fopResultPragFormer)
createDirIfNotExist(fopResultAllMLApproaches)
lstConfigs=['directive','private','reduction']
strBert='bert'

writerOut = pd.ExcelWriter(fopResultAllMLApproaches+'overallResultsSorted.xlsx', engine='xlsxwriter')
for config in lstConfigs:
    dictAllPredictions = {}
    dictOracles={}
    fopStatisticsForProblems=fopResultAllMLApproaches+config+'/'
    createDirIfNotExist(fopStatisticsForProblems)
    fpItemOverallStatistics = fopStatisticsForProblems + 'overallAccuracy.csv'
    fpItemOverallPrediction = fopStatisticsForProblems + 'overallPredictions.csv'
    fopItemDetails=fopStatisticsForProblems+'details/'
    createDirIfNotExist(fopItemDetails)

    dfAllPredictions=pd.read_csv(fpItemOverallPrediction)
    lstModelNames=[]
    for colName in dfAllPredictions.columns:
        # print(colName)
        if colName=='realId' or colName=='label':
            continue
        lstModelNames.append(colName)
    # input('aaaa ')
    lstRealIds=dfAllPredictions['realId'].tolist()
    lstLabels=dfAllPredictions['label'].tolist()
    lstCodes,lstPragmas=getListOfForLoopAndPragma(lstRealIds,lstLabels,fopDatabasePragFormer,500)
    lstCountCorrectModels=[]
    for i in range(0,len(lstRealIds)):
        expectValue=lstLabels[i]
        countCorrectClassifiers=0
        for modelName in lstModelNames:
            predictValue=dfAllPredictions[modelName][i]
            # print('{} {} {}'.format(lstRealIds[i],expectValue,predictValue))
            if expectValue==predictValue:
                countCorrectClassifiers+=1
        lstCountCorrectModels.append(countCorrectClassifiers)
    dfCountCorrect=pd.DataFrame()
    dfCountCorrect['realId'] = lstRealIds
    dfCountCorrect['code']=lstCodes
    dfCountCorrect['pragma'] = lstPragmas
    dfCountCorrect['numOfCorrectPredicts']=lstCountCorrectModels
    dfCountCorrect['label'] = lstLabels
    dfCountCorrect=dfCountCorrect.sort_values(by=['numOfCorrectPredicts'])
    # dfCountCorrect.to_csv(fopResultAllMLApproaches+config+'/statisticAllLabels_all.csv',index=False)
    dfPositiveLabels=dfCountCorrect[dfCountCorrect['label']==1]
    dfNegativeLabels = dfCountCorrect[dfCountCorrect['label'] == 0]
    # dfPositiveLabels.to_csv(fopResultAllMLApproaches+config+'/statisticAllLabels_positive.csv',index=False)
    # dfNegativeLabels.to_csv(fopResultAllMLApproaches + config + '/statisticAllLabels_negative.csv', index=False)
    dfCountCorrect.to_excel(writerOut,sheet_name='{}_all'.format(config))
    dfPositiveLabels.to_excel(writerOut, sheet_name='{}_pos'.format(config))
    dfNegativeLabels.to_excel(writerOut, sheet_name='{}_neg'.format(config))
    for modelName in lstModelNames:
        fopItemModel=fopItemDetails+'/'+modelName+'/'
        createDirIfNotExist(fopItemModel)
        lstPredictValueOfModel=dfAllPredictions[modelName].tolist()
        dfEachModel=pd.DataFrame()
        dfEachModel['realId'] = lstRealIds
        dfEachModel['predict'] = lstPredictValueOfModel
        dfEachModel['label'] = lstLabels
        dfEachModel['numOfCorrectPredicts'] = lstCountCorrectModels
        dfEachModel['code'] = lstCodes
        dfEachModel['pragma'] = lstPragmas
        dfEachModel=dfEachModel.sort_values(by=['numOfCorrectPredicts'])
        dfEachModel.to_csv(fopItemModel+'all.csv',index=False)
        dfTP=dfEachModel[(dfEachModel['predict']==1) & (dfEachModel['label']==1)].sort_values(by=['numOfCorrectPredicts'])
        dfTN = dfEachModel[(dfEachModel['predict'] == 0) & (dfEachModel['label'] == 0)].sort_values(by=['numOfCorrectPredicts'])
        dfFP=dfEachModel[(dfEachModel['predict']==1) & (dfEachModel['label']==0)].sort_values(by=['numOfCorrectPredicts'],ascending=False)
        dfFN = dfEachModel[(dfEachModel['predict'] == 0) & (dfEachModel['label'] == 1)].sort_values(by=['numOfCorrectPredicts'],ascending=False)
        # dfTP.to_csv(fopItemModel + 'TP.csv', index=False)
        # dfTN.to_csv(fopItemModel + 'TN.csv', index=False)
        # dfFP.to_csv(fopItemModel + 'FP.csv', index=False)
        # dfFN.to_csv(fopItemModel + 'FN.csv', index=False)
        createDirIfNotExist(fopResultAllMLApproaches+'excel/'+config+'/')
        writer = pd.ExcelWriter(fopResultAllMLApproaches+'excel/'+config+'/'+modelName+'.xlsx', engine='xlsxwriter')
        dfEachModel.to_excel(writer, sheet_name='{}_all'.format(config))
        dfTP.to_excel(writer, sheet_name='{}_TP'.format(config))
        dfFP.to_excel(writer, sheet_name='{}_FP'.format(config))
        dfTN.to_excel(writer, sheet_name='{}_TN'.format(config))
        dfFN.to_excel(writer, sheet_name='{}_FN'.format(config))
        writer.save()

writerOut.save()














