#!usr/bin/env python
import codecs
import cPickle
import tensorflow as tf
import glob
import pandas as pd
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import sklearn as sk
from sklearn.metrics import confusion_matrix
import config
from pymongo import MongoClient
client = MongoClient()
db = client.scryDb
collection = db.HeData_copy

INPUT_FILE = config.INPUT_FILE
REVIEW_FILE = config.REVIEW_FILE




# Parameters
# ==================================================
# cluster specification
parameter_servers = ["localhost:2222"]
workers = ["localhost:2223"]
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
AFLAGS = tf.app.flags.FLAGS




# Eval Parameters
#tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
#tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
#tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# start a server for a specific task
server = tf.train.Server(cluster,
                          job_name=AFLAGS.job_name,
                          task_index=AFLAGS.task_index)


data = pd.read_csv(INPUT_FILE)
y_test = data['label'].tolist()
y_test = np.array(y_test).astype("float")


data.pop('label')
data.drop(data.columns[[0]], axis=1, inplace=True)
x_test = np.asarray(data)

x_test = x_test.reshape(x_test.shape[0],1,x_test.shape[1])

#Map data into vocabulary
#vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
#vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
#x_test = np.array(list(vocab_processor.transform(x_raw)))



print("\nPredicting...\n")

if AFLAGS.job_name=="ps":
    server.join()
elif AFLAGS.job_name=="worker":
	# Between-graph replication

	with tf.device(tf.train.replica_device_setter(
		worker_device="/job:worker/task:%d" % FLAGS.task_index,
       		cluster=cluster)):
	    all_predictions = []
	    checkpoint_dir = os.path.join("context_protocol","train_param","checkpoints")

	    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
	    print "Checkpoint_file",checkpoint_file
            	
            with tf.Graph().as_default() as graph:
                session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement)
                sess = tf.Session(config=session_conf)
                with tf.Session(server.target) as sess:
                	print("In graph")
                    	# Load the saved meta graph and restore variables
                    	saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                    	saver.restore(sess, checkpoint_file)
		    	print("Graph restored")

                    	# Get the placeholders from the graph by name
                    	input_x = graph.get_operation_by_name("X_train").outputs[0]
                    	#features = graph.get_operation_by_name("features").outputs[0]
                    	dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                     	# Tensors we want to evaluate
                    	predictions = graph.get_operation_by_name("output/predictions").outputs[0]
 
            	        batch_predictions = sess.run(predictions, {input_x: x_test,dropout_keep_prob: 1.0})
            	        all_predictions = np.concatenate([all_predictions, batch_predictions])
		
all_predictions = np.array(all_predictions).astype("float")
correct_predictions = float(sum(all_predictions == y_test))
print("Total number of test examples: {}".format(len(y_test)))
print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
#print ("Precision", sk.metrics.precision_score(y_test, all_predictions,average=None))
#print ("Recall", sk.metrics.recall_score(y_test, all_predictions,average=None))
#print ("f1_score", sk.metrics.f1_score(y_test,all_predictions,average=None))
#print ("confusion_matrix")
#print (sk.metrics.confusion_matrix(y_test, all_predictions))

data['Actual_Label']= y_test
data['Pred_Label']=all_predictions




#Reviewer Pipeline
reviewed_data = pd.read_csv(REVIEW_FILE)
data['Review_Recommendation']=reviewed_data['Recommendation']
print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"

di = {0:"You are on healthy track.Keep following regular diet and exercise.",
1:"You may be suffering from hypertension.Do meditation,yoga,exercise on regular basis.Avoid heavy dose medicines",
2:"You are recommended to have periodic tests,ultrasounds and more frequent health checks.You are stictly required to have healthy diet",
3:"You need to visit once in a week.It is required to quit alcohol,smoking and join some groups for the same."}

data['Pred_Label'].replace(di,inplace=True)
data['Actual_Label'].replace(di,inplace=True)

data.to_csv("./pred_data.csv",sep=",")

#Write to Mongo DB
#Augment Training Set
reviewed_data.pop('Recommendation')


listOfDict = reviewed_data.T.to_dict().values()
for dictionary in listOfDict:
	collection.insert(dictionary)


