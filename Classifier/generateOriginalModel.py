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

class Object(object):
    pass

def generateExcelFile(fpInputJson,fopOutputCsv):
    f1 = open(fpInputJson, 'r')
    strInputJson = f1.read().strip()
    f1.close()
    jsonInput = ast.literal_eval(strInputJson)
    XDsParallel=[]
    yDsParallel=[]
    XDsPrivate = []
    yDsPrivate = []
    XDsReduction = []
    yDsReduction = []
    lstTokenSize=[]
    for key in jsonInput.keys():
        itemJson = jsonInput[key]
        arrFpItemCode=itemJson['code'].strip().split('/')
        fopItemCode='/'.join(arrFpItemCode[:(len(arrFpItemCode)-1)])+'/'
        fnCodeFile=arrFpItemCode[len(arrFpItemCode)-2]

        fpItemCode=fopItemCode+'code.c'
        fpItemPragma=fopItemCode+'pragma.c'
        yItem=0
        f1=open(fpItemCode,'r')
        strCode=f1.read().strip()
        f1.close()
        lstTokenSize.append(len(strCode.split()))
        if os.path.isfile(fpItemPragma):
            yItem=1
            f1=open(fpItemPragma,'r')
            strPragma=f1.read().strip()
            f1.close()
            yReduction=0
            if 'reduction' in strPragma:
                yReduction=1
            tupReduct=[len(XDsReduction),strCode,yReduction,fnCodeFile]
            XDsReduction.append(tupReduct)
            yDsReduction.append(yReduction)
            yPrivate=0
            if 'private' in strPragma:
                yPrivate=1
            tupPrivate=[len(XDsPrivate),strCode,yPrivate,fnCodeFile]
            XDsPrivate.append(tupPrivate)
            yDsPrivate.append(yPrivate)

        indexPar=len(XDsParallel)
        tupPar=[indexPar,strCode,yItem,fnCodeFile]
        XDsParallel.append(tupPar)
        yDsParallel.append(yItem)

    maxTokenSize=max(lstTokenSize)
    indexTokenSize=lstTokenSize.index(maxTokenSize)
    ele=XDsParallel[indexTokenSize]

    print('max of code length \n{}\n{}\n{}'.format(indexTokenSize,maxTokenSize,ele[3]))
    lstSortTokenSize=sorted(lstTokenSize)
    lstSortTokenSizeReverse=sorted(lstTokenSize,reverse=True)
    # print('mean {} median {} min {} max {}'.format(mean(lstSortTokenSize),median(lstSortTokenSize),min(lstSortTokenSize),max(lstSortTokenSize)))
    # print('top 100 \n{}'.format('\n'.join(map(str,lstSortTokenSizeReverse[:100]))))
    # input('aa ')
    X_train, X_test, y_train, y_test = train_test_split(XDsParallel, yDsParallel,
                                                        test_size=0.125, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                        test_size=0.125, shuffle=True)
    fpParallel=fopOutputCsv+'directive.csv'
    # df = pd.DataFrame()
    train=[i[1] for i in X_train]
    train_labels=[i[2] for i in X_train]
    train_ids = [i[3] for i in X_train]
    val=[i[1] for i in X_val]
    val_labels=[i[2] for i in X_val]
    val_ids = [i[3] for i in X_val]
    test=[i[1] for i in X_test]
    test_labels=[i[2] for i in X_test]
    test_ids=[i[3] for i in X_test]
    data=Object()
    data.train=train
    data.train_labels=train_labels
    data.val = val
    data.val_labels = val_labels
    data.test = test
    data.test_labels = test_labels
    data.test_ids=test_ids
    dataFull = Object()
    dataFull.train = X_train
    dataFull.val = X_val
    dataFull.test = X_test

    return data,dataFull

fopInputDataset='/home/hungphd/git/Open_OMP/'
fopOutputFolder='/home/hungphd/git/Open_OMP/repResult/'
fopDatabaseCodeLocation=fopInputDataset+'database/'
fpJsonDatabase=fopInputDataset+'database.json'
createDirIfNotExist(fopOutputFolder)


parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default=None, type=str,
                    dest='config_file', help='The file of the hyper parameters.')
parser.add_argument('--train', default=False, action="store_true",
                    dest='train', help='Train phase.')
parser.add_argument('--predict', default=False, action="store_true",
                    dest='predict', help='Predict phase.')
parser.add_argument('--save', default="", type = str,
                    dest='save', help='Save tokenize phase.')
parser.add_argument('--multiple_gpu', default=False, action = "store_true",
                    dest='multiple_gpu', help='Number of gpus')
parser.add_argument('--out', default=fopOutputFolder+"/saved_weights.pt", type=str,
                    dest='out', help='Saved model name.')

# ***********  Params for data.  **********
parser.add_argument('--data_dir', default="/home/hungphd/git/Open_OMP/", type=str,
                    dest='data_dir', help='The Directory of the data.')
parser.add_argument('--data_type', default="", type=str,
                    dest='data_type', help='The type of read.')
parser.add_argument('--max_len', default=256, type=int,
                    dest='max_len', help='The type of read.')
# parser.add_argument('--specific_directive', default="reduction", type=str,
#                     dest='max_len', help='The type of read.')
parser.add_argument('--reshuffle', dest='reshuffle',action = "store_true", default=False)

args = parser.parse_args()


data,dataFull=generateExcelFile(fpJsonDatabase,fopOutputFolder)
fpOutputParallel=fopOutputFolder+'parallel.csv'
# data=pd.read_csv(fpOutputParallel)

model_pretained_name='NTUYG/DeepSCC-RoBERTa'
batch_size=32
torch.cuda.empty_cache()
# print(torch)
device = torch.device("cuda")
model_pretained_name = "NTUYG/DeepSCC-RoBERTa" #'bert-base-uncased'
# model_pretained_name = 'bert-base-uncased'
pt_model = AutoModel.from_pretrained(model_pretained_name,cache_dir=fopOutputFolder+'cached_Roberta/')
model = BERT_Arch(pt_model)
# define the optimizer
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)

# train_ds=['a']
# print("Example of data: \n", data.train[126])
train, train_size = deepscc_tokenizer(data.train, args.max_len, model_pretained_name,fopOutputFolder+'cached_Roberta/')
val, _ = deepscc_tokenizer(data.val, args.max_len, model_pretained_name,fopOutputFolder+'cached_Roberta/')
train_seq = torch.tensor(train['input_ids'])
train_mask = torch.tensor(train['attention_mask'])
train_y = torch.tensor(data.train_labels)
indexFirst=0
# print('{}\t{}\t{}\n{}\t{}\n{}\t{}\n{}\t{}'.format(train_size,dataFull.train[0][3],dataFull.train[0][2],len(data.train[0].split()),data.train[0],len(train['input_ids'][0]),train['input_ids'][0],len(train['attention_mask'][0]),train['attention_mask'][0]))
# input('train ')
# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)
# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)
# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = batch_size)


# print(val)
val_seq = torch.tensor(val['input_ids'])
val_mask = torch.tensor(val['attention_mask'])
val_y = torch.tensor(data.val_labels)
# print(val_seq)
# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)
# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)
# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size = batch_size)


# prediction model
data_test = data.test
label_test = data.test_labels
test, _ = deepscc_tokenizer(data_test, args.max_len, model_pretained_name,fopOutputFolder+'cached_Roberta/')
maxx = len(test['input_ids'])
test_seq = torch.tensor(test['input_ids'])
test_mask = torch.tensor(test['attention_mask'])
test_y = torch.tensor(label_test)
test_data = TensorDataset(test_seq, test_mask, test_y)
test_sampler = SequentialSampler(test_seq)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
test_to_show = {'label': [], 'id': [], 'input_ids': []}
test_to_show['label'].extend(data.test_labels)
test_to_show['id'].extend(data.test_ids)
test_to_show['input_ids'].extend(test['input_ids'])

class_weights = compute_class_weight('balanced', np.unique(data.train_labels), data.train_labels)
print("Class Weights:", class_weights)
# converting list of class weights to a tensor
weights = torch.tensor(class_weights, dtype=torch.float)
# push to GPU
weights = weights.to(device)
# define the loss function
cross_entropy = nn.NLLLoss(weight=weights)

epochs = 15
# print("Summary:")
# print("Train:", len(train[0]))
# print("Valid:", len(val[0]))
trainer.train(model, epochs, train_dataloader, device, cross_entropy, optimizer, val_dataloader, args.out)
# for each epoch
predict(model, device, test_dataloader, test_y, args.out, test_to_show)