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

fopResultPragFormer='/home/hungphd/git/Open_OMP/repResult/'
fopResultAllMLApproaches='/home/hungphd/git/Open_OMP/ensembleMLApproaches/'
createDirIfNotExist(fopResultPragFormer)
createDirIfNotExist(fopResultAllMLApproaches)
lstConfigs=['directive','private','reduction']
lstClassificationTechniques=[]
fpTextForFasttext=fopResultAllMLApproaches+'fasttext.txt'
fpBinForFasttext=fopResultAllMLApproaches+'fasttext.bin'
if not os.path.isfile(fpBinForFasttext):
    nameSubFolder = '{}/fold-{}/'.format(lstConfigs[0], 1)
    fpItemCSV = fopResultPragFormer + nameSubFolder + 'data.csv'
    pdItemCSV=pd.read_csv(fpItemCSV)
    lstText=pdItemCSV['code'].tolist()
    lstText=[item.replace('\n',strEL) for item in lstText]
    f1=open(fpTextForFasttext,'w')
    f1.write('\n'.join(lstText))
    f1.close()
    model = fasttext.train_unsupervised(fpTextForFasttext, model='cbow', dim=100)
    model.save_model(fpBinForFasttext)
modelFasttext=fasttext.FastText.load_model(fpBinForFasttext)
for config in lstConfigs:
    for foldIndex in range(1,11):
        nameSubFolder='{}/fold-{}/'.format(config,foldIndex)
        fpItemCSV=fopResultPragFormer+nameSubFolder+'data.csv'
        fpOutModelStatistics=fopResultPragFormer+nameSubFolder+'ensemble_models.csv'
        fpOutModelPredictions = fopResultPragFormer + nameSubFolder + 'ensemble_predictions.csv'
        # f1=open(fpItemCSV,'r')
        # arrItemCSVs=f1.read().strip().split('\n')
        # f1.close()
        # arrItemCSVs[0]='len,code,label,id,split_mark'
        # f1=open(fpItemCSV,'w')
        # f1.write('\n'.join(arrItemCSVs))
        # f1.close()

        pdAllData=pd.read_csv(fpItemCSV)
        dfTest =pdAllData[pdAllData['split_mark']=='test']
        dfTrain = pdAllData[pdAllData['split_mark'] != 'test']
        # print(len(dfTrain))
        X_train=pd.DataFrame(np.array( convertListTextToListVector(modelFasttext, dfTrain['code'].tolist())))
        X_test=pd.DataFrame(np.array( convertListTextToListVector(modelFasttext, dfTest['code'].tolist())))
        y_train=dfTrain['label'].tolist()
        y_test=dfTest['label'].tolist()
        id_train = dfTrain['id'].tolist()
        id_test = dfTest['id'].tolist()
        clf = LazyClassifier(verbose=0, ignore_warnings=True,predictions=True)
        # print('{} {} {} {}'.format(type(X_train),type(X_test),type(y_train),type(y_test)))
        modelMLs, predictionMLs = clf.fit(X_train, X_test, y_train, y_test)
        # modelMLs, predictionMLs = clf.fit(X_test, X_test, y_test, y_test)
        # print('{} {}'.format(type(modelMLs),type(predictionMLs)))
        # print('aaaa {} {}'.format(modelMLs.columns.values.tolist(), predictionMLs.columns.values.tolist()))
        predictionMLs=predictionMLs.assign(realId=id_test)
        modelMLs.to_csv(fpOutModelStatistics)
        predictionMLs.to_csv(fpOutModelPredictions)

        # print(predictionMLs)


        # clf = LazyClassifier(verbose=0, ignore_warnings=True)
        # models, predictions = clf.fit(X_train, X_test, y_train, y_test)
        # models
        print('end {}'.format(nameSubFolder))

