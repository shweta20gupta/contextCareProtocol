import tensorflow as tf
import pandas as pd
import numpy as np
#import config
import sys
import os
import math
from sklearn import preprocessing
from pymongo import MongoClient
from mlxtend.preprocessing import minmax_scaling
client = MongoClient()

db = client.scryDb
collection = db.HeData

listOfNumDict = []
listOfStrDict = []
for dict in collection.find():
    numDict = {}
    strDict = {}
    dict["_id"] = str(dict["_id"])
    dict["timestamp"] = str(dict["timestamp"])
    for key in dict:
        if type(dict[key]) != unicode and type(dict[key]) != str:
            numDict[key] = dict[key]
        else:
            strDict[key] = dict[key]
    listOfNumDict.append(numDict)
    listOfStrDict.append(strDict)
numData = pd.DataFrame(listOfNumDict)
strData = pd.DataFrame(listOfStrDict)
numData = minmax_scaling(numData, columns=['height','rbc count','weight','age','wbc count','weight gain during pregenacy',
                                     'blood pressureL','blood pressureU','pregenacy month','platelets count'])
print numData
print strData
reload(sys)  
sys.setdefaultencoding('utf8')


def load_train_test():
	#Read data from Mongo DB
	
    data = pd.DataFrame(listOfDict)
	#data = pd.read_json(OUTPUT_FILE)
    train_test_ratio = int(math.ceil(.2*len(data)))
	train_data = data[:-train_test_ratio].as_matrix()
	test_data = data[-train_test_ratio:]
	test_path = "./context_protocol_test.csv"
	test_data.to_csv(test_path,sep=',')
	y = train_data['Label'].tolist()
	le = preprocessing.LabelEncoder()
	y = le.fit(y)	
	y=np.array(y)
	return [train_data,y]

	


def next_batch(data, batch_size, shuffle=True):

    Generates a batch iterator for a dataset.

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

#load_normalized_data('/home/admin8899/LBR/data/SubChubbFile.csv')
#load_normalized_annotated_data()
#a,b,c,d = load_train_test("Delivery services/messengers")
