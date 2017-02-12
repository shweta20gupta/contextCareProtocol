import tensorflow as tf
import pandas as pd
import numpy as np
import cPickle
#import config
import sys
import os
import math
from sklearn import preprocessing
from pymongo import MongoClient
from mlxtend.preprocessing import minmax_scaling
client = MongoClient()
db = client.scryDb
collection = db.HeData2

inputParaDict = {}
listOfNumDict = []
listOfStrDict = []
for dict in collection.find():
    inputParaDict = dict.keys()
    numDict = {}
    strDict = {}
    dict["_id"] = str(dict["_id"])
    del dict['timestamp']
    #dict["timestamp"] = str(dict["timestamp"])
    for key in dict:
        if type(dict[key]) != unicode and type(dict[key]) != str:
            numDict[key] = dict[key]
        else:
            strDict[key] = dict[key]
    listOfNumDict.append(numDict)
    listOfStrDict.append(strDict)
numData = pd.DataFrame(listOfNumDict)
strData = pd.DataFrame(listOfStrDict)
y = numData["label"]# Training Label List
with open(r"inputParameter.pickle", "wb") as output_file:
    cPickle.dump(inputParaDict, output_file)
#print y
keyList = [str(x) for x in numData.keys().tolist()]
keyList.remove("label")
numData = minmax_scaling(numData, columns=keyList)
numData = numData.join(y)
numData = numData.sample(frac=1)
reload(sys)
sys.setdefaultencoding('utf8')

def load_train_test():
    #Read data from Mongo DB
    train_test_ratio = int(math.ceil(.2*len(numData)))
    train_data = numData[:-train_test_ratio]
    y_train = train_data['label']
    train_data.pop('label')
    test_data = numData[-train_test_ratio:]
    test_path = "./context_protocol_test.csv"
    test_data.to_csv(test_path,sep=',')
    #y = train_data['Label'].tolist()
    #train_data = train_data[['A','B','C']].as_matrix()
    encoder = preprocessing.LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    n_values = np.max(encoded_Y) + 1
    y_train = np.eye(n_values)[encoded_Y]
    return [np.asarray(train_data), np.asarray(y_train),n_values]
	


def next_batch(data, batch_size, shuffle=True):

    #Generates a batch iterator for a dataset.

    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1

    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]


def testData(csvFile):
    inputParaDict = {}
    with open(r"inputParameter.pickle", "rb") as input_file:
        inputParaDict = cPickle.load(input_file)
    df = pd.read_csv(csvFile)
    keyLst = df.keys()
    inputParaToUpdate = []
    dct = collection.find_one()
    del dct['_id']
    del dct['timestamp']
    del dct['label']
    print "#####################"
    print dct
    for k in keyLst:
        if k not in inputParaDict:
            if type(df[k][0]) is str:
                print "Updating String input parameter.............."
                collection.update({}, {'$set': {k: 'NA'}}, upsert=False, multi=True)
            else:
                print "Updating Numeric input parameter.............."
                collection.update({}, {'$set': {k: 0}}, upsert=False, multi=True)
    print "input Parameter update Done.........................................."
    for key in dct:
        if key not in df.keys():
            if type(dct[key]) == unicode:
                df[key] = "NA"
            else:
                df[key] = 0


x,y,n = load_train_test()
print len(x)
print len(y)

