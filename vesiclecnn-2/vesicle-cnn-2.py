# MIT License
#
# Copyright (c) 2017, Probabilistic Programming Group at University of Oxford
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Adaptation of VESICLE-CNN (Roncal et al 2014) to be fully convolutional instead of patch-based.
Script loads, builds, trains and then deploys a fully convolutional approximation of VESCILE-CNN.
Work is described in more detail in [include citation].
"""

# Import stock libraries. ----------------------------------------------------------------------------------------------
import os
import time
import timeit
import matplotlib.pyplot as plt
import h5py
import numpy as np
import tensorflow as tf
import math
import argparse
from shutil import copyfile

# Import additional libraries. -----------------------------------------------------------------------------------------
import utilities as util

# Parse arguments. -----------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='VESICLE-CNN-2 training and deployment framework.')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--train_new', help='Are we training a new network? (mutually exclusive with --deploy_trained).', action='store_true')
group.add_argument('--deploy_pretrained', help='File location of classifier to be deployed. (mutually exclusive with --training).', default=False)

parser.add_argument('--gpu', help='GPU ID to run computations on.', default=False)
parser.add_argument('--train_fraction', help='Fraction of training batches that are positive instances.', default=0.1)
parser.add_argument('--positive_weight', help='The balancing weight used in weighted cross entropy calculations.', default=10)
parser.add_argument('--deploy_train', help='Deploy network to train data set?', action='store_true')
parser.add_argument('--deploy_validation', help='Deploy network to validation dataset', action='store_true')
parser.add_argument('--deploy_test', help='Deploy network to test data set', action='store_true')
parser.add_argument('--deploy_unlabelled', help='Deploy network to test dataset', default=False)
args = parser.parse_args()

# Configure GPU settings. ----------------------------------------------------------------------------------------------
if args.gpu:
	os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # see issue #152
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

''' Configure network settings. -------------------------------------------------------------------------------------'''

# Configure network architecture parameters.
patchSize = [67, 67]
imSizeFC = [5, 5]
convolutionalFilters = 48
firstLayerDimensions = [5, 5, 1, convolutionalFilters]
secondLayerDimensions = [5, 5, convolutionalFilters, convolutionalFilters]
thirdLayerDimensions = [5, 5, convolutionalFilters, convolutionalFilters]
fcNeurons = 1024
fcLayerDimensions = [imSizeFC[0], imSizeFC[1], convolutionalFilters, fcNeurons]
# dropoutProb = 1  # TODO - Not using dropout. 

# Configure training parameters.
trainingSteps = 300000
batch_size = 100
pos_frac = float(args.train_fraction)
pos_weight = float(args.positive_weight)
learningRate = 1e-04
valRegularity = 1000

# Define data locations.
dataLocations = ['./../kasthuri_data/train/train.h5', './../kasthuri_data/validation/validation.h5', './../kasthuri_data/test/test.h5']
channelLocations = ['/synapse']     # Label location _within_ the H5 file.
internalLocations = ['SYN']
imgLocation = '/image'              # Image location _within_ the H5 file.

# Define experimental setup.
training = args.train_new
deployTrain = args.deploy_train
deployValidation = args.deploy_validation
deployTest = args.deploy_test
deployUnlabelled = args.deploy_unlabelled
load_path = args.deploy_pretrained

# Misc.
trainingSteps += 1

''' Read in images from h5. -----------------------------------------------------------------------------------------'''

[trainImage, trainLabels] = util.load_data(dataLocations[0], imgLocation, channelLocations, internalLocations)
[validateImage, validateLabels] = util.load_data(dataLocations[1], imgLocation, channelLocations, internalLocations)
[testImage, testLabels] = util.load_data(dataLocations[2], imgLocation, channelLocations, internalLocations)

assert trainImage.shape[1] == trainImage.shape[2]

# Configure settings pertaining to window size.
border = int(math.floor(patchSize[0]/2))
windowSize = [trainImage.shape[1], trainImage.shape[2]]
finalSize = [windowSize[0] - 2*border, windowSize[1] - 2*border]
imSize = trainImage.shape
imElements = windowSize[0] * windowSize[1]
trainImages = trainImage.shape[0]
validationImages = validateImage.shape[0]
testImages = testImage.shape[0]

# Prepare locations lists.
pad = int(math.floor(patchSize[0] / 2))
positive_locations = np.where(trainLabels['SYN'][:, pad:imSize[1]-pad, pad:imSize[2]-pad, :])
negative_locations = np.where(1 - trainLabels['SYN'][:, pad:imSize[1]-pad, pad:imSize[2]-pad, :])

''' Configure output file -------------------------------------------------------------------------------------------'''

# Configure file for output.
if not args.deploy_pretrained:
	fileOutputName = "Results/VCNN-2_" + time.strftime("%Y_%m_%d") + "_" + time.strftime("%H_%M_%S")
else:
	fileOutputName = args.deploy_pretrained

if not os.path.exists(fileOutputName):
        os.makedirs(fileOutputName)
	
reportLocation = fileOutputName + "/report.txt"
util.echo_to_file(reportLocation, "\n-- VESICLE-CNN-2 synapse detector. --\n")

util.echo_to_file(reportLocation, "Experiment to train VESICLE-CNN-2 to predict synapses.\n")
util.echo_to_file(reportLocation, "Experiment conducted at:" + time.strftime("%d/%m/%Y") + " " + time.strftime("%H:%M:%S") + ".\n\n")

if args.deploy_pretrained is not False:
	util.echo_to_file(reportLocation, "Deploying pre-trained network saved at %s" % args.deploy_pretrained)
else:
	util.echo_to_file(reportLocation, "Training new network.")
	# Copy current version of this script, as well as the makefile just to make sure we capture the experiment.
	if not os.path.exists(fileOutputName + "/backup"):
		os.makedirs(fileOutputName + "/backup")
	copyfile('./vesicle-cnn-2.py', fileOutputName + "/backup/vesicle-cnn-2.py")
	copyfile('./Makefile', fileOutputName + "/backup/Makefile")

util.echo_to_file(reportLocation, "Experimental setup:\n")
util.echo_to_file(reportLocation, "Training settings:\n")
util.echo_to_file(reportLocation, "\tLearning scheme: ADAM\n")
util.echo_to_file(reportLocation, "\tLearning rate: %f\n" % learningRate)
util.echo_to_file(reportLocation, "\tTraining steps:  %s\n" % trainingSteps)
util.echo_to_file(reportLocation, "\tOptimize weighted cross entropy\n")
util.echo_to_file(reportLocation, "\tSelect best network from F1\n\n")
util.echo_to_file(reportLocation, "\tTraining ratio: %f\n" % pos_frac)
util.echo_to_file(reportLocation, "\tTraining weight: %f\n" % pos_weight)
util.echo_to_file(reportLocation, "\tBatch size: %f\n" % batch_size)

util.echo_to_file(reportLocation, "Architecture settings:\n")
util.echo_to_file(reportLocation, "\tConvolution layer 1: %s\n" % firstLayerDimensions)
util.echo_to_file(reportLocation, "\tConvolution layer 2: %s\n" % secondLayerDimensions)
util.echo_to_file(reportLocation, "\tConvolution layer 3: %s\n" % thirdLayerDimensions)
util.echo_to_file(reportLocation, "\tFC units: %f\n" % fcNeurons)
util.echo_to_file(reportLocation, "\tDropout prob: CURRENTLY DISABLED\n")  #%f\n" % dropoutProb)  # TODO - disabled dropout.
util.echo_to_file(reportLocation, "\tInput patch size: %s\n" % patchSize)

''' Configure TensorFlow graph -------------------------------------------------------------------------------------'''

util.echo_to_file(reportLocation, "\nConfiguring network.\n")

# Use an interactive session for debugging.
#config_opt = tf.ConfigProto()
#config_opt.gpu_options.allow_growth = True
#sess = tf.InteractiveSession(config=config_opt)

# Create placeholders for independent and dependant variables.
with tf.name_scope('Input_Image'):
	x = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='Image')   # Independent variables.
with tf.name_scope('Input_Label'):
	y_syn = tf.placeholder(tf.float32, shape=[None, 2])                         # Target values.

with tf.name_scope('First_Layer'):
	# Create first convolutional layer.
	W_conv1 = util.weight_variable(firstLayerDimensions, "w_conv_1")                            # Weights in first layer.
	b_conv1 = util.bias_variable([firstLayerDimensions[3]], "b_conv_1")                         # Biases in first layer.
	h_conv1 = tf.nn.relu(util.conv2d(x, W_conv1, valid=True, stride=1) + b_conv1)               # Perform convolution (with zero padding) and apply ReLU.
	#h_pool1 = util.max_pool(h_conv1, 1)  # NOTE removed this maxpool layer as it fufilss no purpose at this level -- it just reduces the resolution, therefore just copy.
	h_pool1 = h_conv1  

with tf.name_scope('Second_Layer'):
	# Create first convolutional layer.
	W_conv2 = util.weight_variable(secondLayerDimensions, "w_conv_2")                           # Weights in second layer.
	b_conv2 = util.bias_variable([secondLayerDimensions[3]], "b_conv_2")                        # Biases in second layer.
	h_conv2 = tf.nn.relu(util.atrous_conv2d(h_pool1, W_conv2, valid=True, rate=2) + b_conv2)    # Perform atrous convolution (with zero padding) and apply ReLU.
	h_pool2 = util.atrous_max_pool(h_conv2, mask_size=2, rate=2)                                # Apply an atrous maxpool.

with tf.name_scope('Third_Layer'):
	# Create first convolutional layer.
	W_conv3 = util.weight_variable(thirdLayerDimensions, "w_conv_3")                            # Weights in third layer.
	b_conv3 = util.bias_variable([thirdLayerDimensions[3]], "b_conv_3")                         # Biases in third layer.
	h_conv3 = tf.nn.relu(util.atrous_conv2d(h_pool2, W_conv3, valid=True, rate=4) + b_conv3)    # Perform atrous convolution (with zero padding) and apply ReLU.
	h_pool3 = util.atrous_max_pool(h_conv3, mask_size=2, rate=4)                                # Apply an atrous maxpool.

with tf.name_scope('fccnn_Layer'):
	# Create FC layer for final classification.
	W_fccnn1 = util.weight_variable(fcLayerDimensions, "w_fccnn_1")                             # Weights for patch for FC.
	b_fccnn1 = util.bias_variable([fcLayerDimensions[3]], "b_fccnn_1")                          # Biases for firstFCNeurons neurons.
	h_fccnn4 = tf.nn.relu(util.atrous_conv2d(h_pool3, W_fccnn1, valid=True, rate=8) + b_fccnn1) # Perform atrous convolution (with zero padding) and apply ReLU.
	
	# Throw some dropout in there for good measure.
#	keep_prob = tf.placeholder(tf.float32)  # TODO - removed droupout.
#	h_cnnfc1_drop = tf.nn.dropout(h_fccnn4, keep_prob)  # TODO - removed dropout.

with tf.name_scope('Output_Layer'):
	# Now add a final output layer.
	W_fccnn5 = util.weight_variable([1, 1, fcLayerDimensions[3], 2], "w_fccnn_5")
	b_fccnn5 = util.bias_variable([2], "b_fccnn_5")
	y_syn_logit = util.conv2d(h_fccnn4, W_fccnn5, valid=True) + b_fccnn5  # NOTE - removed ReLU from here.
	y_syn_soft = tf.nn.softmax(y_syn_logit)
	y_syn_logit_flat = tf.reshape(y_syn_logit, [-1, 2])
	y_syn_soft_flat = tf.reshape(y_syn_soft, [-1, 2])
	
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Define cross entropy as loss function.
with tf.name_scope('XEnt'):
	cross_entropy = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=y_syn, logits=y_syn_logit_flat, pos_weight=pos_weight, name='syn_Loss'))

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

# Use an interactive session for debugging.
config_opt = tf.ConfigProto()
config_opt.gpu_options.allow_growth = True
sess = tf.Session(config=config_opt)

# Initialize the variables.
sess.run(tf.global_variables_initializer())

# Declare misc variables for storing times etc.
f1s = []
xEnts = []
accs = []
trainTimes = np.zeros((trainingSteps, 1))
gpuTimes = np.zeros((trainingSteps, 1))

''' Train network -------------------------------------------------------------------------------------------'''


# Function to automate application of a classifier to the validation volume.
def validate_network(_f1s=[], _accs=[], _xents=[], final_val=False):
	val_cross_entropy, val_accuracy, val_fmeasure = np.zeros((validationImages, 8)), np.zeros((validationImages, 8)), np.zeros((validationImages, 8))
	for j in range(validationImages):
		for k in range(8):
			val_batch = util.get_minibatch_image(validateImage, validateLabels, batch_size=1, valN=j, orientation=k, border=border)
			reshaped = np.reshape(val_batch[0], [-1, windowSize[0], windowSize[1], 1])
			val_cross_entropy[j, k], val_accuracy[j, k], val_fmeasure[j, k] = sess.run([cross_entropy, accuracy, fmeasure], feed_dict={x: reshaped, y_syn: val_batch[1]['SYN']})  # , keep_prob: 1.0}) # TODO - removed droupout.
	
	validation_accuracy = np.average(np.average(val_accuracy))
	validation_cross_entropy = np.average(np.average(val_cross_entropy))
	validation_fmeasure = np.average(np.average(val_fmeasure))

	_f1s.append(validation_fmeasure)
	_accs.append(validation_accuracy)
	_xents.append(validation_fmeasure)
	
	if (np.nanmax(f1s) == validation_fmeasure) | (f1s == []):
		saver.save(sess, fileOutputName + "/CNN.ckpt")
	
	if not final_val:
		output_string = ("step %d, validation accuracy %g, cross entropy %g, f1(ave) %g\n" % (i, validation_accuracy, validation_cross_entropy, validation_fmeasure))
	else:
		output_string = ("Validation accuracy using single best validated model, applied to whole of validation set: \n\n\t Validation error: %g\n\t Validation XEnt: %g\n\t Validation F1: %g\n\t" % (_accs[0], _xents[0], _f1s[0]))
	
	util.echo_to_file(reportLocation, output_string)
	
	return _f1s, _accs, _xents


# If we are training the network (as opposed to deploying an existing network).
if training:
	
	util.echo_to_file(reportLocation, "\nTraining network.\n")
	
	for i in range(trainingSteps):
		
		if i % valRegularity == 0:
			f1s, accs, xEnts = validate_network(f1s, accs, xEnts)
			
		startTime = timeit.default_timer()
		batch = util.get_minibatch_patch(trainImage, trainLabels['SYN'], batch_size, patchSize, pos_frac=pos_frac, pos_locs=positive_locations, neg_locs=negative_locations)
		startTimeGPU = timeit.default_timer()
		_, summary = sess.run([train_step, summary_op], feed_dict={x: batch[0], y_syn: batch[1]})  # TODO - removed droupout., keep_prob: dropoutProb})
		elapsed = timeit.default_timer() - startTime
		gpuElapsed = timeit.default_timer() - startTimeGPU
		trainTimes[i] = elapsed
		gpuTimes[i] = gpuElapsed
		writer.add_summary(summary, i)	

	av = np.sum(trainTimes) / trainingSteps
	gpu_av = np.sum(gpuTimes) / trainingSteps
	# Now write the timings to the output file.
	util.echo_to_file(reportLocation, "\nAverage training step time: %g s (%g GPU s). \n\n" % (av, gpu_av))
	
	# Restore the best net.
	saver.restore(sess, fileOutputName + "/CNN.ckpt")
	
	# Do final validation on network.
	validate_network(final_val=True)

else:
	# Odd hack required. Deployment to GPU without taking at least a single training step causes memory error. TODO fix this.
        batch = util.get_minibatch_patch(trainImage, trainLabels['SYN'], batch_size, patchSize, pos_frac=pos_frac, pos_locs=positive_locations, neg_locs=negative_locations)
        _, summary = sess.run([train_step, summary_op], feed_dict={x: batch[0], y_syn: batch[1]})


	
''' Define targets to allow for validation and deployment of network ---------------------------------------------'''


# Apply the classifier (defined by _func) to the image stack (defined by _images).
def apply_classifier(_func, _images):

	_numFrames = _images.shape[0]
	
	volume_prediction = np.zeros((_numFrames, finalSize[0], finalSize[1], 2))
	_application_times = np.zeros((_numFrames, 1))
	_gpu_times = np.zeros((_numFrames, 1))	
	
	for i in range(_numFrames):
		startTime = timeit.default_timer()
		_single_im = np.expand_dims(_images[i, :, :].astype(np.float32), axis=0)
		startTimeGPU = timeit.default_timer()
		predFlat = sess.run(_func, feed_dict={x: _single_im})  # , keep_prob: 1.0}) # TODO - dropout removed.
		elapsed = timeit.default_timer() - startTimeGPU
		_gpu_times[i] = elapsed
		
		singlePred = np.reshape(predFlat[0, :, :, 0], finalSize)
		volume_prediction[i, :, :, 0] = singlePred
		
		singlePred = np.reshape(predFlat[0, :, :, 1], finalSize)
		volume_prediction[i, :, :, 1] = singlePred
		
		elapsed = timeit.default_timer() - startTime
		_application_times[i] = elapsed

		print("Prediction of layer %g/%g complete." % (i + 1, _numFrames))
	
	av = np.sum(_application_times) / _numFrames
	gpu_av = np.sum(_gpu_times) / _numFrames
	util.echo_to_file(reportLocation, "\nAverage time application time per frame: %g s (%g GPU s). \n\n" % (av, gpu_av))

	return volume_prediction


# Evaluate F1 score for a test volume (volume_prediction) to a ground truth volume (_labels).
# Use _channel as a tag for output.
def evaluate_f1(volume_prediction, _labels, _channel):
	precision, recall, f1 = util.calculate_PR(volume_prediction, _labels)
	
	# Now write the timings to the output file.
	util.echo_to_file(reportLocation, "\n" + _channel + "\n")
	util.echo_to_file(reportLocation, "\nPrecision:  %g \n" % precision)
	util.echo_to_file(reportLocation, "\nRecall:     %g \n" % recall)
	util.echo_to_file(reportLocation, "\nF1:         %g \n" % f1)
	util.echo_to_file(reportLocation, "\n")
	
	return precision, recall, f1


# Deploy the classifier, defined by its logit function (_logit_func), to an image stack (_image).
# Which channel to apply to is defined by _channel.
# _set defines whether we are applying to train, validate or test set, and stores results inside _file.
# _label then defines the ground truth stack (may not exist for new, unlabelled data).
def deploy_to_channel(_logit_func, _image, _channel, _set, _file, _label=None):
	print(_channel + " " + _set + " prediction.")
	
	_file.create_dataset('image', data=_image)
	
	logits = apply_classifier(_logit_func, _image)
	# Now save output and truth values sed.
	group = _file.create_group(_channel)
	group.create_dataset('zeros', data=np.squeeze(logits[:, :, :, 0]))
	group.create_dataset('ones', data=np.squeeze(logits[:, :, :, 1]))
	
	if _label is not None:
		trimmed_labels = _label[:, border:finalSize[0]+border, border:finalSize[1]+border, :]
		group.create_dataset('truth', data=np.squeeze(trimmed_labels))
		# Create P-R metrics using softmax layer.
		prediction = (logits[:, :, :, 1] > logits[:, :, :, 0]).astype(np.int8)
		precision, recall, f1 = evaluate_f1(prediction, trimmed_labels, _channel + "-" + _set)
		group.attrs['Precision'] = precision
		group.attrs['Recall'] = recall
		group.attrs['F1'] = f1


# Script for testing speed of application of _func to _image.
def application_speed_test(_func, _image):
	_application_times = np.zeros((_image.size[0], 1))
	for i in range(_image.size[0]):
		_single_im = np.expand_dims(trainImage[i, :, :].astype(np.float32), axis=0)
		startTime = timeit.default_timer()
		_ = sess.run(_func, feed_dict={x: _single_im})  # , keep_prob: 1.0}) # TODO - removed dropout.
		elapsed = timeit.default_timer() - startTime
		_application_times[i] = elapsed
	
	av = np.sum(_application_times) / _image.size[0]
	util.echo_to_file(reportLocation, "\nAverage application time per frame: %g s \n" % av)


# Now lets go and deploy the algorithm to the datasets.
if deployTrain | deployValidation | deployTest | deployUnlabelled:
	
	# Load the correct classifier file.
	# If we have been training, re-load the optimally trained classifier.
	# Else, load the classifier defined by the input.
	if training:
		saver.restore(sess, fileOutputName + "/CNN.ckpt")
		load_path = fileOutputName
	else:
		saver.restore(sess, load_path + "/CNN.ckpt")
	
	util.echo_to_file(reportLocation, "\nTesting network parameters saved at: " + load_path + "\n")
	
	if deployTrain:
		util.echo_to_file(reportLocation, "Beginning dense application to training set.")
		
		# Create h5 file for saving output.
		h5f = h5py.File(fileOutputName + '/train_results.h5', 'w')
		h5f.attrs['Creation_Date'] = time.strftime("%Y_%m_%d") + "_" + time.strftime("%H_%M_%S")
		h5f.attrs['Data_Set'] = "Training"
		h5f.attrs['Network_Location'] = load_path
		h5f.attrs['Network'] = "VESICLE-CNN-2"
		
		deploy_to_channel(y_syn_logit, trainImage, 'syn', 'train', h5f, trainLabels['SYN'])  # Syn.
		
		h5f.close()
	
	if deployValidation:
		util.echo_to_file(reportLocation, "Beginning dense validation to training set.")
		
		# Create h5 file for saving output.
		h5f = h5py.File(fileOutputName + '/validation_results.h5', 'w')
		h5f.attrs['Creation_Date'] = time.strftime("%Y_%m_%d") + "_" + time.strftime("%H_%M_%S")
		h5f.attrs['Data_Set'] = "Validation"
		h5f.attrs['Network_Location'] = load_path
		h5f.attrs['Network'] = "VESICLE-CNN-2"
		
		deploy_to_channel(y_syn_logit, validateImage, 'syn', 'validation', h5f, validateLabels['SYN'])  # Syn.
		
		h5f.close()
	
	if deployTest:
		util.echo_to_file(reportLocation, "Beginning dense application to test set.")
		
		# Create h5 file for saving output.
		h5f = h5py.File(fileOutputName + '/test_results.h5', 'w')
		h5f.attrs['Creation_Date'] = time.strftime("%Y_%m_%d") + "_" + time.strftime("%H_%M_%S")
		h5f.attrs['Data_Set'] = "Test"
		h5f.attrs['Network_Location'] = load_path
		h5f.attrs['Network'] = "VESICLE-CNN-2"
		
		deploy_to_channel(y_syn_logit, testImage, 'syn', 'test', h5f, testLabels['SYN'])  # Syn.
		
		h5f.close()

	if deployUnlabelled:
		util.echo_to_file(reportLocation, "Beginning dense application to unlabelled set.")
		
		# We need to load this one.
		unlabelledLoc = dataLocations[0]
		unlabelledImage, _ = util.load_data(unlabelledLoc, imgLocation)
		
		# Create h5 file for saving output.
		h5f = h5py.File(fileOutputName + '/unlabelled_results.h5', 'w')
		h5f.create_dataset('image', data=testImage)
		h5f.attrs['Creation_Date'] = time.strftime("%Y_%m_%d") + "_" + time.strftime("%H_%M_%S")
		h5f.attrs['Data_Set'] = "Unlabelled"
		h5f.attrs['Network_Location'] = load_path
		h5f.attrs['Network'] = "VESICLE-CNN-2"
		
		deploy_to_channel(y_syn_logit, testImage, 'syn', 'unlabelled', h5f)  # Syn.
		
		h5f.close()

# Now run the MATLAB accuracy evaluation script.
# This makes a call to matlab and passes in the arguements to the evaluation script.

# Close the TF session to release the resources.
sess.close()
del sess

# Make the MATLAB call.
os.system('matlab -r "addpath(genpath(\'../evaluation\')); wrap_synapse_pr(\'./' + fileOutputName +'\' ,\'syn\'); wrap_voxel_pr(\'./' + fileOutputName +'\' ,\'syn\'); exit"')


## Finish up.
util.echo_to_file(reportLocation, "-- End of VESICLE-CNN-2 report. --")
