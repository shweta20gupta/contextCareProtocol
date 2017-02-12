#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import math


# Parameters
#tf.app.flags.DEFINE_string("ps_hosts","","List of parameter servers")
#tf.app.flags.DEFINE_string("worker_hosts","","List of worker servers")
# ==================================================


# Model Hyperparameters
#tf.flags.DEFINE_integer("embedding_dim",128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,1,1", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.4, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")



# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
AFLAGS = tf.app.flags.FLAGS


# cluster specification
ps_hosts = ["localhost:2222"]
worker_hosts = ["localhost:2223"]
parameter_servers = ps_hosts
workers = worker_hosts
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})



# start a server for a specific task
server = tf.train.Server(cluster,  
                          job_name=AFLAGS.job_name,
                          task_index=AFLAGS.task_index)

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparatopn
# ==================================================

# Load data

print("Loading data for....")
x,y,n_values=data_helpers.load_train_test()


'''
 Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text["InputNarrative"].tolist()])
print("max document length",max_document_length)
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text["InputNarrative"].tolist())))
'''

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
 # TODO: This is very crude, should use cross-validation
tdr = int(math.ceil(.2*len(x_shuffled)))#train_dev_split_ratio
x_train,x_dev,y_train,y_dev = x_shuffled[:-tdr],x_shuffled[-tdr:],y_shuffled[:-tdr],y_shuffled[-tdr:] 




#x_text_train, x_num_train,x_text_dev,x_num_dev = x_text_shuffled[:-tdr], x_num_shuffled[:-tdr],x_text_shuffled[-tdr:],x_num_shuffled[-tdr:]
#y_train, y_dev = y_shuffled[:-tdr], y_shuffled[-tdr:]
#print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

if AFLAGS.job_name == "ps":
	server.join()
elif AFLAGS.job_name=="worker":
	# Between-graph replication

	with tf.device(tf.train.replica_device_setter(
		worker_device="/job:worker/task:%d"% FLAGS.task_index,
		cluster=cluster)):
   	        # count the number of updates
     		global_step = tf.get_variable('global_step', [],
                                initializer = tf.constant_initializer(0),
                                trainable = False)


		cnn = TextCNN(filter_sizes=list(map(int,FLAGS.filter_sizes.split(","))),
            	num_filters=FLAGS.num_filters, vec_shape=(1,x_train.shape[1]),l2_reg_lambda=FLAGS.l2_reg_lambda,num_classes=4)
        	# Define Training procedure
		with tf.name_scope('train'):
        		optimizer = tf.train.AdamOptimizer(0.001)
        		grads_and_vars = optimizer.compute_gradients(cnn.loss)
        		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        	# Keep track of gradient values and sparsity (optional)
        	#grad_summaries = []
        	#for g, v in grads_and_vars:
            	#	if g is not None:
                #		grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                #		sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                #		grad_summaries.append(grad_hist_summary)
                #		grad_summaries.append(sparsity_summary)
        	#grad_summaries_merged = tf.merge_summary(grad_summaries)

        	# Output directory for models and summaries
        	#timestamp = str(int(time.time()))
        	#out_dir = os.path.abspath(os.path.join(os.path.curdir,codeDescription, timestamp))
       		#print("Writing to {}\n".format(out_dir))

       	 	# Summaries for loss and accuracy
        	#loss_summary = tf.scalar_summary("loss", cnn.loss)
        	#acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        	# Train Summaries
        	#train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        	#train_summary_dir = os.path.join(out_dir, "summaries", "train")
        	#train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        	# Dev summaries
        	#dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        #	dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        #	dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        	# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        	#checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        	#checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        	#if not os.path.exists(checkpoint_dir):
            	#	os.makedirs(checkpoint_dir)
        	#saver = tf.train.Saver(tf.all_variables())

        	# Write vocabulary
        	#vocab_processor.save(os.path.join(out_dir, "vocab"))

        	# Initialize all variables
        	init_op = tf.initialize_all_variables()
		print("Variables Initialized")
		saver = tf.train.Saver(tf.all_variables())

	sv = tf.train.Supervisor(is_chief=(AFLAGS.task_index==0),global_step=global_step,init_op=init_op)
	begin_time = time.time()
  	frequency = 100
  	with sv.prepare_or_wait_for_session(server.target) as sess:
	
	# create log writer object (this will log on every machine)
		# Output directory for models and summaries
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir,"context_protocol", timestamp))
                print("Writing to {}\n".format(out_dir))

		# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                #saver = tf.train.Saver(tf.all_variables())

                # Write vocabulary
                #vocab_processor.save(os.path.join(out_dir, "vocab"))

		#train_summary_dir = os.path.join(out_dir, "summaries", "train")
                #train_summary_writer = tf.train.SummaryWriter(train_summary_dir,graph=tf.get_default_graph())
		#dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
                #dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, graph= tf.get_default_graph())


    		#writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
		def train_or_dev_step(x_train, y,dropout,epoch,train):
                        feed_dict = {
                        cnn.input_x: x_train,
                        cnn.input_y: y,
                        cnn.dropout_keep_prob:dropout
                        }
                        _, step, loss, accuracy = sess.run(
                                [train_op, global_step, cnn.loss, cnn.accuracy],
                                feed_dict)
                        time_str = datetime.datetime.now().isoformat()
                        print("{}: epoch {}, step {} ,loss {:g}, acc {:g}".format(time_str, epoch,step, loss, accuracy))
			#if train:
                        	#train_summary_writer.add_summary(summaries, step)
			#else:
				#writer.add_summary(summaries,step)

		x_dev = np.asarray(x_dev).reshape(len(x_dev),1,x_train.shape[1])
    		# perform training cycles
   		start_time = time.time()
    		for epoch in range(FLAGS.num_epochs):
			#  batches in one epoch
			if sv.should_stop():
				break
			batch_count = int(len(x_train)/FLAGS.batch_size)
			print("Total number of batches in one epoch",batch_count)
			batches = data_helpers.next_batch(list(zip(x_train,y_train)),FLAGS.batch_size) #TO DO
			count = 0
			for i,batch in enumerate(batches):
				if len(batch) == FLAGS.batch_size:
        			# Training loop. For each batch...
            				x_train_batch,y_train_batch = zip(*batch)

					x_train_batch = np.asarray(x_train_batch).reshape(FLAGS.batch_size,1,x_train.shape[1])
					print np.asarray(x_train_batch).shape
            				train_or_dev_step(x_train_batch,np.asarray(y_train_batch),FLAGS.dropout_keep_prob,epoch+1,train=True)
            				current_step = tf.train.global_step(sess, global_step)
            				if current_step % batch_count == 0:
                				print("\nEvaluation:")
                				train_or_dev_step(x_dev, np.asarray(y_dev), 1.0,epoch+1,train=False)
                				print("")
            				if current_step % FLAGS.checkpoint_every == 0:
                				path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                				print("Saved model checkpoint to {}\n".format(path))
	print("Total time taken",time.time()-start_time)
	sv.stop()
	print("done")


