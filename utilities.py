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
Utility script for TensorFlow code projects.
VERY MUCH EXPERIMENTAL CODE.
"""

import tensorflow as tf
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
import sklearn.metrics as sklM
import sklearn.feature_extraction as sklF
import h5py
import timeit
from operator import xor


# Writes a string to a file, while also echoing it to the command line.
def echo_to_file(_floc, _str):
    # Print the string to the screen.
    print _str
    # Now write the result to the output file.
    fo = open(_floc, "a")
    fo.write(_str)
    fo.close()
    
    
# Function reads and loads train, validation and test datasets from disk, and loads them into
# a single, concatenated array.
def load_data(_h5_location, _img_location, _channel_locations=None, _internal_locations=None):
    
    if xor(_channel_locations==None, _internal_locations==None):
        raise IOError('_channel_locations and _internal_locations must both be either None, or take a list value.')
    
    # Open h5 file for reading.
    file_location = _h5_location
    data_file = h5py.File(file_location, 'r')
    
    # Read in images.
    images = data_file[_img_location].value
    images = images.astype('f')
    images = images.reshape(images.shape + (1,))
    image_count = images.shape[0]

    # Normalize the training batch.
    for i in range(image_count):
        img_av = np.mean(images[i, :, :, 0])
        images[i, :, :, 0] -= img_av  # Now zero mean.
        train_normalizer = np.max(np.absolute(images[i, :, :, 0]))
        images[i, :, :, 0] /= train_normalizer  # Now exists between -1 & 1.
    
    # Define a holder for storing the labels.
    labels = {}

    for loc, internal in zip(_channel_locations, _internal_locations):
        lab = data_file[loc].value
        lab = lab.reshape(lab.shape + (1,))
        labels[internal] = lab
        
    return images, labels
    

# Library functions that initialize variables to small positive values.
# ReLU neurons should be initialized with positive weights so as to not kill the gradients.
def weight_variable(shape, _name=None, _summ=None):
    initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    
    if _name is not None:
        weights = tf.Variable(initial, dtype=tf.float32)
    else:
        weights = tf.Variable(initial, name=_name, dtype=tf.float32)

    if _summ is not None:
        variable_summaries(weights)
    
    return weights


def bias_variable(shape, _name=None, _summ=None):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    if _name is not None:
        biases = tf.Variable(initial, dtype=tf.float32)
    else:
        biases = tf.Variable(initial, name=_name, dtype=tf.float32)
        
    if _summ is not None:
        variable_summaries(biases)
    
    return biases


# Library functions that implement zero padded convolution using the efficient tensor flow library.
# `SAME' padding indicates as a setting that the input is padded to retain the same output size.
# Strides set to one in each dimension means that there is no reduction in spatial extent through the convolution.
def conv2d(x, W, stride=1, valid=None):
    if valid is None:
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
    else:
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')


def atrous_conv2d(x, W, rate=1, valid=None):
    if valid is None:
        return tf.nn.convolution(x, W, dilation_rate=[rate, rate], padding='SAME')
    else:
        return tf.nn.convolution(x, W, dilation_rate=[rate, rate], padding='VALID')
    

# Pooling is used to reduce the spatial extent of the image.
# Here, the kernel is set to be the standard one batch, stride x stride across each channel.
# The padding option means that the input is zero padded, not to preserve spatial dimension,
#   but ensure that each element is used in the pooling instead of being dropped, as is the case with valid pooling.
def max_pool(x, stride, mode=None, kernelWidth=None):
    if kernelWidth is None:
        kernelWidth = stride
        
    if mode==None:
        return tf.nn.max_pool(x, ksize=[1, kernelWidth, kernelWidth, 1], strides=[1, stride, stride, 1], padding='SAME')
    else:
        return tf.nn.max_pool(x, ksize=[1, kernelWidth, kernelWidth, 1], strides=[1, stride, stride, 1], padding='VALID')


# Apply max pool but with dilation.
def atrous_max_pool(x, mask_size, rate):
    return tf.nn.pool(x, window_shape=[mask_size, mask_size], pooling_type='MAX', padding='VALID', dilation_rate=[rate, rate])


def get_minibatch_image(x, y, batch_size, valN=None, orientation=None, depth=0, border=0):
    # valN is a scalar whose value defines the image from which we are going to draw the validation image from.
    # If valN is not set, we are compiling a training set and it will be fully stochastic.
    
    imSize = x.shape
    labPatch = (imSize[1]-(2*border)) * (imSize[2]-(2*border))
    imPatch = imSize[1] * imSize[2]
    
    # Are we compiling a validation set?
    if valN==None:
        a_samp = random.sample(range(0+depth, imSize[0]-depth), batch_size)
    else:
        # Check batch size is one.
        if batch_size != 1:
            warnings.warn("Batch size for validation sets can only be one. Undefined behaviour otherwise.")
            
        if orientation == None:
            warnings.warn("Must specify orientation when creating validation image.")
            
        if ((orientation < 0) or (orientation > 7)):
            warnings.warn("Orientation not in valid range.")
        
        # Getting image for single orientation.
        a_samp = [valN]
        batch_size = 1
        a_samp = np.repeat(a_samp,batch_size)

    retIm = np.zeros((batch_size, imPatch*(2*depth+1)))
        
    tempLab = {}
    retLab = {}
    retLabFlat = {}
    
    for feat in y.keys():
        retLab[feat] = np.zeros((batch_size, labPatch, 2))

    for i in range(batch_size):
        
        a = a_samp[i]
        tempIm = x[range((a-depth),(a+depth+1)), :, :, 0]

        for feat in y.keys():
            if border != 0:
                tempLab[feat] = y[feat][a, border:-border, border:-border, 0]
            else:
                tempLab[feat] = y[feat][a, :, :, 0]
        if valN == None:
            rotIm, rotLab = rotate_image(tempIm, tempLab)  # Compiling a training set.
        else:
            rotIm, rotLab = rotate_image(tempIm, tempLab, orientation)  # Compiling a single validation image.
        
        # Create vector.

        retIm[i, :] = np.reshape(rotIm, (1, -1))
        # Creating one-hot.
        for feat in y.keys():
            catLab = np.reshape(rotLab[feat], (1, -1))
            retLab[feat][i, :] = np.eye(2)[catLab]

    # Return rotated images as flattened vector.
    for feat in y.keys():
        retLabFlat[feat] = retLab[feat].reshape(-1, retLab[feat].shape[-1])
        
    return retIm, retLabFlat


def rotate_image(tempIm, tempLab, val=None):
    
    # Val is a scalar in [0,7] that indicates transpose and rotate level.
    # Even -> No transpose. Odd -> transpose.
    # Div2 -> rotations.

    # Are we compiling a validation dataset or a training set?
    if val==None:
        # Training.
        transp = random.randint(0, 1)  # Do we transpose our image?
        rotate = random.randint(0, 3)  # How many times do we rotate?
    else:
        # Validation.
        transp = int(val % 2)
        rotate = int(math.floor(val/2))
        
    if tempIm.ndim == 2:
        t_order = [1,0]
        r_order = [0,1]
    elif tempIm.ndim == 3:
        t_order = [0,2,1]
        r_order = [1,2]

    # Do we transpose?
    if transp == 1:
        tempIm = np.transpose(tempIm,axes=t_order)
        for feat in tempLab.keys():
            tempLab[feat] = np.transpose(tempLab[feat])
    
    # Now rotate that many times.
    for i in range(0,rotate):
        tempIm = np.rot90(tempIm,axes=r_order)
        for feat in tempLab.keys():
            tempLab[feat] = np.rot90(tempLab[feat])
    
    return tempIm, tempLab


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def calculate_PR(predictions, truth):
    binaryPredications = np.reshape(np.squeeze(predictions)>0.5,[-1,1])
    binaryTruth = np.reshape(truth>0.5,[-1,1])
    precision, recall, f1, support = sklM.precision_recall_fscore_support(binaryPredications,binaryTruth, average='binary')
    return precision, recall, f1


def tf_calculate_PR(predictions, truth):
    
    predicted = tf.reshape(predictions[:,1]>0.5, [-1, 1])
    actual = tf.reshape(truth[:,1]>0.5, [-1, 1])
    
    # Count true positives, true negatives, false positives and false negatives.
    tp = tf.reduce_sum(tf.cast(tf.logical_and(predicted, actual), tf.float32))
    tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(predicted), tf.logical_not(actual)), tf.float32))
    fp = tf.reduce_sum(tf.cast(tf.logical_and(predicted, tf.logical_not(actual)), tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(predicted), actual), tf.float32))

    # Calculate accuracy, precision, recall and F1 score.
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fmeasure = (2 * precision * recall) / (precision + recall)
    
    return accuracy, precision, recall, fmeasure


def get_minibatch_patch(x, y, batch_size, window_shape, valN=None, pos_frac=None, seed=None, pos_locs=None, neg_locs=None):
    if seed != None:
        random.seed(seed)
    
    imSize = x.shape
    
    pad = int(math.floor(window_shape[0] / 2))
    
    retPatches = np.zeros((batch_size, window_shape[0], window_shape[1], 1), dtype=np.float32)
    retLabels = np.zeros((batch_size, 2), dtype=np.float32)
    
    if valN == None:
        
        pos_ex = batch_size * pos_frac
        
        for i in range(batch_size):
            
            if i < pos_ex:
                samp = random.randint(0, len(pos_locs[0]) - 1)
                z_index = pos_locs[0][samp]
                x_index = pos_locs[1][samp] + pad
                y_index = pos_locs[2][samp] + pad
                lab = 1
            else:
                samp = random.randint(0, len(neg_locs[0]) - 1)
                z_index = neg_locs[0][samp]
                x_index = neg_locs[1][samp] + pad
                y_index = neg_locs[2][samp] + pad
                lab = 0
            
            img = x[z_index, :, :, 0]
            extract_int = img[x_index - pad:x_index + pad + 1]
            extract = extract_int[:, y_index - pad:y_index + pad + 1]
            
            rotations = random.randint(0, 3)
            post_rotation = np.rot90(extract, rotations)
            transpose = random.randint(0, 1)
            if transpose == 1:
                post_rotation = np.transpose(post_rotation)
            retPatches[i, :, :] = np.expand_dims(post_rotation, 2)
            
            retLabels[i, lab] = 1
    
    else:
        
        # valN[0] is image to get patch from.
        img = np.squeeze(x[valN[0], :, :, ])
        lab = np.squeeze(y[valN[0], :, :, ])
        
        # valN[1] is x index of row to draw from
        # valN[2] is range of y indexes to draw from
        
        for i in range(int(pad), valN[2] - int(pad)):
            y_index = i
            x_index = valN[1]
            
            extract_int = img[x_index - pad:x_index + pad + 1, :]
            extract = extract_int[:, y_index - pad:y_index + pad + 1]
            
            retPatches[i - pad, :, :] = np.expand_dims(extract, axis=3)
            
            retLabels[i - pad, lab[x_index - pad, y_index - pad]] = 1
    
    return retPatches, retLabels
