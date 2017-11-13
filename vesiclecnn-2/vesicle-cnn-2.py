"""
Adaptation of VESICLE CNN (Roncal) to be fully convolutional instead of patch-based.
"""

''' Configure workspace and hardware. -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

print("-- VESICLE-CNN-2 synapse detector --\n")

# Known problems. ------------------------------------------------------------------------------------------------------

# Import stock libraries. ----------------------------------------------------------------------------------------------
import collections as col
import os
import sys
import time
import timeit
from inspect import getsourcefile
import matplotlib.pyplot as plt
import h5py
import numpy as np
import tensorflow as tf
import warnings
from multiprocessing.pool import ThreadPool
import threading
import ndio
import ndio.remote.neurodata as neurodata
import math

# Import additional libraries. -----------------------------------------------------------------------------------------
current_path = os.path.abspath(getsourcefile(lambda: 0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)
import util

# Ingest arguments. ----------------------------------------------------------------------------------------------------
run = int(sys.argv[1])
gpu = str(sys.argv[2])
hyp = float(sys.argv[3])  # For reading in hyperparameters.

# Configure GPU settings. ----------------------------------------------------------------------------------------------
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

''' Configure XVal settings. --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
performingXVal = False

slices = 200

if performingXVal:
	train_temp = range(0, slices)
	testSlices = range(run * 20, (run + 1) * 20)
	if run <= 8:
		validationSlices = range((run + 1) * 20, (run + 2) * 20)
	elif run == 9:
		validationSlices = range(0, 20)
	else:
		raise ValueError("Improper setting of RUN.")
	train_temp = list(set(train_temp) ^ set(testSlices))
	trainSlices = list(set(train_temp) ^ set(validationSlices))

else:
	trainSlices = range(0, 75)
	validationSlices = range(75, 100)
	testSlices = range(0, 100)

# Display slices for user checking.
print(trainSlices)
print(validationSlices)
print(testSlices)

# Declare storage arrays for whole datasets.
allImages = np.zeros([slices, 1024, 1024, 1])
allLabelsSyn = np.zeros([slices, 1024, 1024, 1])
imCount = 0

''' Configure network settings. -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

patchSize = [67, 67]
windowSize = [1024, 1024]
border = int(math.floor(patchSize[0]/2))
trainingSteps = 300000
batch_size = 100
finalFilterCount = 48
firstFCNeurons = 512
pos_frac = hyp

finalSize = [windowSize[0] - 2*border, windowSize[1] - 2*border]


# First layer:
firstLayerDimensions = [5, 5, 1, finalFilterCount]

# Second layer:
secondLayerDimensions = [5, 5, finalFilterCount, finalFilterCount]

# Third layer:
thirdLayerDimensions = [5, 5, finalFilterCount, finalFilterCount]

learningRate = 1e-04
valRegularity = 1000

downloadData = False
training = True
deployTrain = True
deployValidation = True
deployTest = True
load_path = ''  # Path for loading pre-trained conv net.

# Do some error catching on network parameters. ------------------------------------------------------------------------
trainingSteps += 1

''' Read in images from h5. ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# Configure file locations WITHIN h5 file.
imgLocation = '/image'
synLocation = '/synapse'

# Read in training images. ---------------------------------------------------------------------------------------------
allImages = util.load_data(slices, '../kasthuri_data/train/train.h5', '../kasthuri_data/validation/validation.h5', '../kasthuri_data/test/test.h5', imgLocation)
allLabelsSyn = util.load_data(slices, '../kasthuri_data/train/train.h5', '../kasthuri_data/validation/validation.h5', '../kasthuri_data/test/test.h5', synLocation)

''' Now separate data sets according to required partitioning -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

trainImage = allImages[trainSlices, :, :, :]
trainLabelSyn = allLabelsSyn[trainSlices, :, :, :]
trainLabels = {'SYN': trainLabelSyn.astype(np.int32)}
trainImages = trainImage.shape[0]

validateImage = allImages[validationSlices, :, :, :]
validateLabelSyn = allLabelsSyn[validationSlices, :, :, :]
validateLabels = {'SYN': validateLabelSyn.astype(np.int32)}
validationImages = validateImage.shape[0]

testImage = allImages[testSlices, :, :, :]
testLabelSyn = allLabelsSyn[testSlices, :, :, :]
testLabels = {'SYN': testLabelSyn.astype(np.int32)}
testImages = testImage.shape[0]

true_ratio_syn = float(np.sum(trainLabelSyn != 0)) / ((trainLabelSyn.shape[0] * trainLabelSyn.shape[1] * trainLabelSyn.shape[2]) - float(np.sum(trainLabelSyn != 0)))
true_ratio_syn = 10.0;
del allImages, allLabelsSyn

''' CONFIGURE OUTPUT FILE ------------------------------------------------------------------------------------------ '''

# Configure file for output.
fileOutputName = "VCNN-2_train_" + time.strftime("%Y_%m_%d") + "_" + time.strftime("%H_%M_%S_") + str(run)

if not os.path.exists(fileOutputName):
	os.makedirs(fileOutputName)

fo = open(fileOutputName + "/report.txt", "w")
fo.write("Experiment to train VESICLE-CNN-2 to predict synapses, using straight upadted vesicle architecture.\n")
fo.write("Experiment conducted at:" + time.strftime("%d/%m/%Y") + " " + time.strftime("%H:%M:%S") + ".\n\n")
fo.write("Experimental setup:\n")
if performingXVal:
	fo.write("Performing 10x cross-validation, run: %s\n" % run)
elif downloadData:
	fo.write("Downloading data on-the-fly for testing")

if not downloadData:
	fo.write("Training settings:\n")
	fo.write("\tLearning scheme: ADAM\n")
	fo.write("\tTraining steps:  %s\n" % trainingSteps)
	fo.write("\tOptimize weighted cross entropy\n")
	fo.write("\tSelect best network from F1\n\n")
	fo.write("\tTraining ratio: %f\n" % pos_frac)
	fo.write("\tTraining weight: %f\n" % true_ratio_syn)
	fo.write("\tFC units: %f\n" % firstFCNeurons)
fo.close()
# '''



# Configure TensorFlow. ------------------------------------------------------------------------------------------------

imSize = trainImage.shape
imElements = windowSize[0] * windowSize[1]
imSizeFC = [5, 5]

# Use an interactive session for debugging.
config_opt = tf.ConfigProto()
config_opt.gpu_options.allow_growth = True

sess = tf.InteractiveSession(config=config_opt)

# TensorBoard debug view.
file_writer = tf.summary.FileWriter('LOGS', sess.graph)

# Create placeholders for independent and dependant variables once batch has been selected.
with tf.name_scope('Input_Image'):
	x = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='Image')  # Independent variables.
	# Reshape to amenable shape.
	# x_image = tf.reshape(x, [-1, windowSize[0], windowSize[1], 1])
with tf.name_scope('Input_Synapse'):
	y_syn = tf.placeholder(tf.float32, shape=[None, 2])  # Target values.

with tf.name_scope('First_Layer'):
	# Create first convolutional layer. (No pooling.)
	W_conv1 = util.weight_variable(firstLayerDimensions, "w_conv_1")  # Weights in first layer.
	b_conv1 = util.bias_variable([firstLayerDimensions[3]], "b_conv_1")  # Biases in first layer.
	h_conv1 = tf.nn.relu(util.conv2d(x, W_conv1, valid=True, stride=1) + b_conv1)  # Perform convolution (with zero padding) and apply ReLU.
	h_pool1 = util.max_pool(h_conv1, 1, kernelWidth=2)

with tf.name_scope('Second_Layer'):
	# Create first convolutional layer. (No pooling.)
	W_conv2 = util.weight_variable(secondLayerDimensions, "w_conv_2")  # Weights in first layer.
	b_conv2 = util.bias_variable([secondLayerDimensions[3]], "b_conv_2")  # Biases in first layer.
	h_conv2 = tf.nn.relu(util.atrous_conv2d(h_pool1, W_conv2, valid=True, rate=2) + b_conv2)  # Perform convolution (with zero padding) and apply ReLU.
	h_pool2 = util.atrous_max_pool(h_conv2, mask_size=2, rate=2)

with tf.name_scope('Third_Layer'):
	# Create first convolutional layer. (No pooling.)
	W_conv3 = util.weight_variable(thirdLayerDimensions, "w_conv_3")  # Weights in first layer.
	b_conv3 = util.bias_variable([thirdLayerDimensions[3]], "b_conv_3")  # Biases in first layer.
	h_conv3 = tf.nn.relu(util.atrous_conv2d(h_pool2, W_conv3, valid=True, rate=4) + b_conv3)  # Perform convolution (with zero padding) and apply ReLU.
	h_pool3 = util.atrous_max_pool(h_conv3, mask_size=2, rate=4)

with tf.name_scope('fccnn_Layer'):
	# Create FC layer for final classification.
	W_fccnn1 = util.weight_variable([imSizeFC[0], imSizeFC[1], finalFilterCount, firstFCNeurons], "w_fccnn_1")  # Image patch for FC, with firstFCNeurons neurons.
	b_fccnn1 = util.bias_variable([firstFCNeurons], "b_fccnn_1")  # Biases for firstFCNeurons neurons.
	h_fccnn1 = tf.nn.relu(util.atrous_conv2d(h_pool3, W_fccnn1, valid=True, rate=8) + b_fccnn1)  # Perform convolution (with zero padding) and apply ReLU.

# Insert more FC layers here.

with tf.name_scope('Output_Layer'):
	# Now add a final sigmoid layer for prediction of 0-1 probability and readout.
	W_fccnn2 = util.weight_variable([1, 1, firstFCNeurons, 2], "w_fccnn_2")
	b_fccnn2 = util.bias_variable([2], "b_fccnn_2")
	y_syn_logit = util.conv2d(h_fccnn1, W_fccnn2, valid=True) + b_fccnn2
	y_syn_soft = tf.nn.softmax(y_syn_logit)
	y_syn_logit_flat = tf.reshape(y_syn_logit, [-1, 2])
	y_syn_soft_flat = tf.reshape(y_syn_soft, [-1, 2])
	
	
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Define cross entropy as loss function.
with tf.name_scope('XEnt'):
	cross_entropy = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=y_syn, logits=y_syn_logit_flat, pos_weight=true_ratio_syn, name='syn_Loss'))

# Calculate accuracy as an average across vector.
with tf.name_scope('Acc'):
	# Get the predictions for later evaluation.
	predictions = tf.argmax(y_syn_soft, 3)
	
	# Binary vector of correct predictions.
	correct_prediction = tf.equal(tf.argmax(y_syn_logit_flat, 1), tf.argmax(y_syn, 1))
	
	# Now calc accuracy.
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	# Calculate f1
	accuracyF1, precision, recall, fmeasure = util.tf_calculate_PR(y_syn_soft_flat, y_syn)

# Use ADAM as optimizer to minimize the cross entropy.
train_step = tf.train.AdamOptimizer(learningRate).minimize(cross_entropy)

# Join metrics to create callable output.
both = tf.stack([cross_entropy, accuracy, fmeasure], axis=0)

# Merge all the summaries and write them out.
writer = tf.summary.FileWriter(fileOutputName + '/LOGS', graph=tf.get_default_graph())

with tf.name_scope('Losses'):
	tf.summary.scalar("cost", cross_entropy)
	tf.summary.scalar("accuracy", accuracy)

with tf.name_scope('F-Metrics'):
	tf.summary.scalar("accuracy_(f1)", accuracyF1)
	tf.summary.scalar("precision", precision)
	tf.summary.scalar("recall", recall)
	tf.summary.scalar("fmeasure", fmeasure)

# Merge all summaries into a single "operation" which we can execute in a session.
summary_op = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(fileOutputName + '/LOGS')

# Initialize the variables.
sess.run(tf.global_variables_initializer())

f1s = []

trainTimes = np.zeros((trainingSteps, 1))

# Prepare locations lists.
pad = int(math.floor(patchSize[0] / 2))
pos_locs = np.where(trainLabels['SYN'][:,pad:imSize[1]-pad,pad:imSize[2]-pad,:])
neg_locs = np.where(1 - trainLabels['SYN'][:,pad:imSize[1]-pad,pad:imSize[2]-pad,:])

# ======================================================================================================= TRAIN NETWORK. ==============================================================================================

if training:
	
	for i in range(trainingSteps):
		batch = util.get_minibatch_patch(trainImage, trainLabels['SYN'], batch_size, patchSize, pos_frac=pos_frac, pos_locs=pos_locs, neg_locs=neg_locs)
		if i % valRegularity == 0:
			val_cross_entropy = np.zeros((validationImages, 8))
			val_accuracy = np.zeros((validationImages, 8))
			val_fmeasure = np.zeros((validationImages, 8))
			for j in range(validationImages):
				for k in range(8):
					valBatch = util.get_minibatch_image(validateImage, validateLabels, batch_size=1, valN=j, orientation=k, border=border)
					reshaped = np.reshape(valBatch[0], [-1, windowSize[0], windowSize[1], 1])
					val_cross_entropy[j, k], val_accuracy[j, k], val_fmeasure[j, k] = both.eval(feed_dict={x: reshaped, y_syn: valBatch[1]['SYN']})
			
			validation_accuracy = np.average(np.average(val_accuracy))
			validation_cross_entropy = np.average(np.average(val_cross_entropy))
			validation_fmeasure = np.average(np.average(val_fmeasure))
			
			print("step %d, validation accuracy %g, cross entropy %g, f1(ave) %g\n" % (
			i, validation_accuracy, validation_cross_entropy, validation_fmeasure))  # Accuracy is evaluated by a forward pass over the accuracy object, which is the final entry in a long list of 'tf.' objects. Therefore, evaluating accuracy forces the evulation of all previous elements.
			f1s.append(validation_fmeasure)
			if np.nanmax(f1s) == validation_fmeasure:
				save_path = saver.save(sess, fileOutputName + "/CNN.ckpt")
			
			# Now write the result to the output file.
			fo = open(fileOutputName + "/report.txt", "a")
			fo.write("step %d, validation accuracy %g, cross entropy %g, f1(ave) %g\n" % (i, validation_accuracy, validation_cross_entropy, validation_fmeasure))
			fo.close()
		
		startTime = timeit.default_timer()
		_, summary = sess.run([train_step, summary_op], feed_dict={x: batch[0], y_syn: batch[1]})
		writer.add_summary(summary, i)
		elapsed = timeit.default_timer() - startTime
		trainTimes[i] = elapsed
	
	av = np.sum(trainTimes) / trainingSteps
	# Now write the timings to the output file.
	fo = open(fileOutputName + "/report.txt", "a")
	fo.write("\nAverage training step time: %g s \n\n" % av)
	fo.close()
	
	# Restore the best net.
	saver.restore(sess, fileOutputName + "/CNN.ckpt")
	
	# Evaluate performance on validation set. ==============================================================================================
	
	val_accuracy = np.zeros((validationImages, 8))
	val_cross_entropy = np.zeros((validationImages, 8))
	val_fmeasure = np.zeros((validationImages, 8))
	
	for j in range(validationImages):
		for k in range(8):
			valBatch = util.get_minibatch_image(validateImage, validateLabels, batch_size=1, valN=j, orientation=k, border=border)
			reshaped = np.reshape(valBatch[0], [-1, windowSize[0], windowSize[1], 1])
			val_cross_entropy[j, k], val_accuracy[j, k], val_fmeasure[j, k] = both.eval(feed_dict={x: reshaped, y_syn: valBatch[1]['SYN']})
	
	validation_accuracy = np.average(np.average(val_accuracy))
	validation_cross_entropy = np.average(np.average(val_cross_entropy))
	validation_fmeasure = np.average(np.average(val_fmeasure))
	
	print("Validation accuracy using single best validated model, applied to whole of validation set: \n\n\t Validation error: %g\n\t Validation XEnt: %g\n\t Validation F1: %g\n\t" % (
	validation_accuracy, validation_cross_entropy, validation_fmeasure))  # Accuracy is evaluated by a forward pass over the accuracy object, which is the final entry in a long list of 'tf.' objects. Therefore, evaluating accuracy forces the evulation of all previous elements.
	
	fo = open(fileOutputName + "/report.txt", "a")
	fo.write("Validation accuracy using single best validated model, applied to whole of validation set: \n\n\t Validation error: %g\n\t Validation XEnt: %g\n\t Validation F1: %g\n\t" % (validation_accuracy, validation_cross_entropy, validation_fmeasure))
	fo.close()
	
	p = 0


# ======================================================================================== DEFINE PREDICTION SUBROUTINE.

def apply_f1(_func, _images, _labels, _channel):
	_numFrames = _images.shape[0]
	_fmeasures = []
	
	for i in range(_numFrames):
		singleIm = np.squeeze(_images[i, :, :]).astype(np.float32)
		singleImFlat = np.reshape(singleIm, [1, -1])
		catLab = np.reshape(_labels[i, :, :], (1, -1)).astype(np.int32)
		retLab = np.squeeze(np.eye(2)[catLab])
		_fmeasure = _func.eval(feed_dict={x: singleImFlat, y_syn: retLab})
		_fmeasures.append(_fmeasure)
	
	f1 = np.sum(_fmeasures) / _numFrames
	
	# Now write the timings to the output file.
	fo = open(fileOutputName + "/report.txt", "a")
	fo.write("\n" + _channel + "\n")
	fo.write("\nF1:         %g \n" % f1)
	fo.write("\n")
	fo.close()
	print("\nF1:         %g \n" % f1)
	
	return f1


def apply_classifier(_func, _images):
	global volume_prediction
	
	_numFrames = _images.shape[0]
	
	volume_prediction = np.zeros((_numFrames, finalSize[0], finalSize[1], 2))
	appTimes = np.zeros((_numFrames, 1))
	
	for i in range(_numFrames):
		singleIm = np.expand_dims(_images[i, :, :].astype(np.float32), axis=0)
		startTime = timeit.default_timer()
		predFlat = _func.eval(feed_dict={x: singleIm})
		elapsed = timeit.default_timer() - startTime
		appTimes[i] = elapsed
		
		singlePred = np.reshape(predFlat[0, :, :, 0], finalSize)
		volume_prediction[i, :, :, 0] = singlePred
		
		singlePred = np.reshape(predFlat[0, :, :, 1], finalSize)
		volume_prediction[i, :, :, 1] = singlePred
		
		print("Prediction of layer %g/%g complete." % (i + 1, _numFrames))
	
	av = np.sum(appTimes) / _numFrames
	fo = open(fileOutputName + "/report.txt", "a")
	fo.write("\nAverage application time per frame: %g s \n" % av)
	fo.close()
	
	return volume_prediction


def evaluate_f1(volume_prediction, _labels, _channel):
	precision, recall, f1 = util.calculate_PR(volume_prediction, _labels)
	
	# Now write the timings to the output file.
	fo = open(fileOutputName + "/report.txt", "a")
	fo.write("\n" + _channel + "\n")
	fo.write("\nPrecision:  %g \n" % precision)
	fo.write("\nRecall:     %g \n" % recall)
	fo.write("\nF1:         %g \n" % f1)
	fo.write("\n")
	fo.close()
	
	print("\nPrecision:  %g \n" % precision)
	print("\nRecall:     %g \n" % recall)
	print("\nF1:         %g \n" % f1)
	
	return precision, recall, f1


def deploy_to_channel(_logit_func, _image, _label, _channel, _set, _file):
	print(_channel + " " + _set + " prediction.")
	trimmed_labels = _label[:, border:finalSize[0]+border, border:finalSize[1]+border, :]
	logits = apply_classifier(_logit_func, _image)
	# Now save output and truth values sed.
	group = _file.create_group(_channel)
	group.create_dataset('zeros', data=np.squeeze(logits[:, :, :, 0]))
	group.create_dataset('ones', data=np.squeeze(logits[:, :, :, 1]))
	group.create_dataset('truth', data=np.squeeze(trimmed_labels))
	# Create P-R metrics using softmax layer.
	prediction = (logits[:, :, :, 1] > logits[:, :, :, 0]).astype(np.int8)
	precision, recall, f1 = evaluate_f1(prediction, trimmed_labels, _channel + "-" + _set)
	group.attrs['Precision'] = precision
	group.attrs['Recall'] = recall
	group.attrs['F1'] = f1


# # Application speed test.
# apptimes = np.zeros((trainingImages, 1))
# for i in range(trainingImages):
# 	singleIm = np.expand_dims(trainImage[i, :, :].astype(np.float32), axis=0)
# 	startTime = timeit.default_timer()
# 	a = sess.run(predictions, feed_dict={x: singleIm, keep_prob: 1.0})
# 	elapsed = timeit.default_timer() - startTime
# 	apptimes[i] = elapsed
#
# av = np.sum(apptimes) / trainingImages
# fo = open(fileOutputName + "/report.txt", "a")
# fo.write("\nAverage application time per frame: %g s \n" % av)
# fo.close()
# print("\nAverage application time per frame: %g s \n" % av)

if deployTrain | deployValidation | deployTest:
	
	if training:
		saver.restore(sess, fileOutputName + "/CNN.ckpt")
		load_path = fileOutputName
	elif load_path != '':
		saver.restore(sess, load_path + "/CNN.ckpt")
	else:
		load_path = fileOutputName
	
	# Now write the timings to the output file.
	fo = open(fileOutputName + "/report.txt", "a")
	fo.write("\nTESTING ALGORITHM\n")
	fo.write("Algorithm values saved at: " + load_path + "\n")
	fo.close()
	
	if deployTrain:
		print("Beginning dense application to validation set.")
		
		# Ensure we aren't stung by side effects...
		numFrames = trainImage.shape[0]
		
		# Create h5 file for saving output.
		h5f = h5py.File(fileOutputName + '/train_results.h5', 'w')
		h5f.create_dataset('image', data=trainImage)
		h5f.attrs['Creation_Date'] = time.strftime("%Y_%m_%d") + "_" + time.strftime("%H_%M_%S")
		h5f.attrs['Data_Set'] = "Training"
		h5f.attrs['Network_Location'] = load_path
		h5f.attrs['Network'] = "PSyn"
		
		deploy_to_channel(y_syn_logit, trainImage, trainLabelSyn, 'syn', 'train', h5f)  # Syn.
		
		h5f.close()
	
	if deployValidation:
		print("Beginning dense application to validation set.")
		
		# Ensure we aren't stung by side effects...
		numFrames = validateImage.shape[0]
		
		# Create h5 file for saving output.
		h5f = h5py.File(fileOutputName + '/validation_results.h5', 'w')
		h5f.create_dataset('image', data=validateImage)
		h5f.attrs['Creation_Date'] = time.strftime("%Y_%m_%d") + "_" + time.strftime("%H_%M_%S")
		h5f.attrs['Data_Set'] = "Training"
		h5f.attrs['Network_Location'] = load_path
		h5f.attrs['Network'] = "PSyn"
		
		deploy_to_channel(y_syn_logit, validateImage, validateLabelSyn, 'syn', 'validation', h5f)  # Syn.
		
		h5f.close()
	
	if deployTest:
		print("Beginning dense application to test set.")
		
		# Ensure we aren't stung by side effects...
		numFrames = testImage.shape[0]
		
		# Create h5 file for saving output.
		h5f = h5py.File(fileOutputName + '/test_results.h5', 'w')
		h5f.create_dataset('image', data=testImage)
		h5f.attrs['Creation_Date'] = time.strftime("%Y_%m_%d") + "_" + time.strftime("%H_%M_%S")
		h5f.attrs['Data_Set'] = "Training"
		h5f.attrs['Network_Location'] = load_path
		h5f.attrs['Network'] = "PSyn"
		
		deploy_to_channel(y_syn_logit, testImage, testLabelSyn, 'syn', 'test', h5f)  # Syn.
		
		h5f.close()
