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


# Function reads and loads train, validation and test datasets from disk, and loads them into
# a single, concatenated array.
def load_data(_slices, _trainLoc, _valLoc, _testLoc, _channel):
    
    # Declare storage arrays for whole datasets.
    allImages = np.zeros([_slices, 1024, 1024, 1])
    imCount = 0
    
    # Open h5 file for reading.
    fileLocation = _trainLoc
    dataFile = h5py.File(fileLocation, 'r')
    
    # Load the images.
    trainImage = dataFile[_channel].value
    trainImage = trainImage.astype('f')
    trainImage = trainImage.reshape(trainImage.shape + (1,))
    trainingImages = trainImage.shape[0]
    
    # Normalize the training batch.
    for i in range(trainingImages):
        if _channel == '/image':
            imgAv = np.mean(trainImage[i, :, :, 0])
            trainImage[i, :, :, 0] -= imgAv  # Now zero mean.
            trainNormalizer = np.max(np.absolute(trainImage[i, :, :, 0]))
            trainImage[i, :, :, 0] /= trainNormalizer  # Now exists between -1 & 1.
        allImages[imCount, :, :, 0] = trainImage[i, :, :, 0]
        imCount += 1
        
    # Read in validation images. -------------------------------------------------------------------------------------------
    
    # Open h5 file for reading.
    fileLocation = _valLoc
    dataFile = h5py.File(fileLocation, 'r')
    
    # Load the validation images.
    validateImage = dataFile[_channel].value
    validateImage = validateImage.astype('f')
    validateImage = validateImage.reshape(validateImage.shape + (1,))
    validationImages = validateImage.shape[0]
    
    # Normalize the validation batch.
    for i in range(validationImages):
        if _channel == '/image':
            imgAv = np.mean(validateImage[i, :, :, 0])
            validateImage[i, :, :, 0] -= imgAv  # Now zero mean.
            valNormalizer = np.max(np.absolute(validateImage[i, :, :, 0]))
            validateImage[i, :, :, 0] /= valNormalizer  # Now exists between -1 & 1.
        allImages[imCount, :, :, 0] = validateImage[i, :, :, 0]
        imCount += 1
    
    # Read in test images. -------------------------------------------------------------------------------------------------
    
    # Open h5 file for reading.
    fileLocation = _testLoc
    dataFile = h5py.File(fileLocation, 'r')
    
    # Load the validation images.
    testImage = dataFile[_channel].value
    testImage = testImage.astype('f')
    testImage = testImage.reshape(testImage.shape + (1,))
    testImages = testImage.shape[0]
    
    # Normalize the test batch.
    for i in range(testImages):
        if _channel == '/image':
            imgAv = np.mean(testImage[i, :, :, 0])
            testImage[i, :, :, 0] -= imgAv  # Now zero mean.
            valNormalizer = np.max(np.absolute(testImage[i, :, :, 0]))
            testImage[i, :, :, 0] /= valNormalizer  # Now exists between -1 & 1.
        allImages[imCount, :, :, 0] = testImage[i, :, :, 0]
        imCount += 1
        
    return allImages
        

# Library functions that initialize variables to small positive values.
# ReLU neurons should be initialized with positive weights so as to not kill the gradients.
def weight_variable(shape, _name=None, _summ=None):
    initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    
    if _name != None:
        weights = tf.Variable(initial, dtype=tf.float32)
    else:
        weights = tf.Variable(initial, name=_name, dtype=tf.float32)

    if _summ != None:
        variable_summaries(weights)
    
    return weights


def bias_variable(shape, _name=None, _summ=None):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    if _name != None:
        biases = tf.Variable(initial, dtype=tf.float32)
    else:
        biases = tf.Variable(initial, name=_name, dtype=tf.float32)
        
    if _summ != None:
        variable_summaries(biases)
    
    return biases


# Library routine for upsampling an image by a factor of _upsample by copying nearest neighbour.
def upsample_nn(image, _upsample):
    sizes = image.get_shape()
    new_height = sizes[1] * _upsample
    new_height = new_height.value
    new_width = sizes[2] * _upsample
    new_width = new_width.value
    up = tf.image.resize_images(image, [new_height, new_width])
    return up


# Library functions that implement zero padded convolution using the efficient tensor flow library.
# `SAME' padding indicates as a setting that the input is padded to retain the same output size.
# Strides set to one in each dimension means that there is no reduction in spatial extent through the convolution.
def conv2d(x, W, stride=1, valid=None):
    if valid == None:
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
    else:
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')


def atrous_conv2d(x, W, rate=1, valid=None):
    if valid == None:
        return tf.nn.convolution(x, W, dilation_rate=[rate, rate], padding='SAME')
    else:
        return tf.nn.convolution(x, W, dilation_rate=[rate, rate], padding='VALID')


def _conv3d(x, W, stride=1, valid=None, depth=1):
    if valid == None:
        return tf.nn.conv3d(x, W, strides=[1, stride, stride, depth, 1], padding='SAME')
    else:
        return tf.nn.conv3d(x, W, strides=[1, stride, stride, depth, 1], padding='VALID')

def deconv2d(x, W, stride, batchSize):
    in_shape = x.get_shape()
    batch_size = tf.shape(x)[0]
    filterDims = W.get_shape()
    deconv_shape = tf.stack([batch_size, in_shape[1] * 2, in_shape[2] * 2, filterDims[2]])
    return tf.nn.conv2d_transpose(x, W, deconv_shape, strides=[1, stride, stride, 1], padding='SAME')

def conv_relu_layer(input, weights, biases):
    relu = tf.nn.relu(conv2d(input, weights) + biases)  # Perform convolution (with zero padding) and apply ReLU.
    return relu

def conv_relu_layer_3d(input, weights, biases, depth):
    relu = tf.nn.relu(_conv3d(input, weights, depth=depth) + biases)  # Perform convolution (with zero padding) and apply ReLU.
    return relu

def conv_bn_relu_layer(input, weights, biases, phase=True):
    conv = conv2d(input, weights) + biases
    bn = tf.contrib.layers.batch_norm(conv,center=True, scale=True,is_training=phase)
    relu = tf.nn.relu(bn)  # Perform convolution (with zero padding) and apply ReLU.
    return relu


def conv3d(x, W, stride=1):
    return tf.nn.conv3d(x, W, strides=[1, stride, stride, 1, 1], padding='VALID')


def conv3d_relu_layer(input, weights, biases):
    relu = tf.nn.relu(conv3d(input, weights) + biases)
    return relu


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

def atrous_max_pool(x, mask_size, rate):
    return tf.nn.pool(x, window_shape=[mask_size, mask_size], pooling_type='MAX', padding='VALID', dilation_rate=[rate, rate])


# Function accepts arguments of the full data corpus, the full vector of labels, and the required batch size.
# Random examples are then drawn from the full corpus to create a minibatch.
# Returns the minibatch as a tuple containing the new images and the new labels.
# Exemplar usage: (testIm, testLab) = util.get_minibatch(xTrain, yTrain, 50)
# Optional fourth argument to specify a crop from the image.
# The image returned will then be a cropped version of the image cropped to be the size specified by the vector crop.
# Exemplar usage: (testIm, testLab) = util.get_minibatch(xTrain, yTrain, 50, (20,20)) extracts a random 20x20 crop.
def get_minibatch(x, y, batch_size, crop=None, regression=False):
    imSize = x.shape

    imPatch = 0;

    if crop is None:
        imPatch = imSize[1]*imSize[2]*imSize[3]
    else:
        imPatch = crop[0]*crop[1]*imSize[3]

    retIm = np.zeros((batch_size, imPatch))


    if regression == True:
        retLab = np.zeros((batch_size, 1))
    else:
        retLab = np.zeros((batch_size, 2))

    for i in range(0, batch_size):
        a = random.randint(0, len(x)-1)

        if crop is None:
            tempIm = x[a, :, :, :]
            
            retIm[i, :] = np.reshape(tempIm, (1, -1))
        else:
            imStart1 = random.randint(0, imSize[1]-crop[0]-1)
            imStart2 = random.randint(0, imSize[2]-crop[1]-1)
            imDim1 = range(imStart1, imStart1+crop[0])
            imDim2 = range(imStart2, imStart2+crop[1])
            imTemp = x[a, imDim1, :, :] # Export row-wise submatrix as intermediate step.
            retIm[i, :] = np.asarray(np.reshape(imTemp[:, imDim2, :], [-1, imPatch]))

        if len(y.shape) == 1:
            # The labels are a vector.
            retLab[i, y[a]] = 1  # Create one-hot vector.
        else:
            # The labels are also an image with a 1:1 mapping to the images
            offset1 = int(math.floor(crop[0]/2))
            offset2 = int(math.floor(crop[1]/2))
            lab = y[a, imStart1 + offset1, imStart2 + offset2]
            if regression == True:
                retLab[i] = lab  # Create regression vector.
            else:
                p = int(round(lab/255))
                retLab[i, p] = 1  # Create one-hot vector.


    return retIm, retLab


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
            warning.warn("Must specify orientation when creating validation image.")
            
        if ((orientation < 0) or (orientation > 7)):
            warning.warn("Orientation not in valid range.")
        
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


def get_minibatch_patch(x, y, batch_size, window_shape, valN=None, pos_frac=None,seed=None, pos_locs=None, neg_locs=None):
    
    if seed != None:
        random.seed(seed)

    imSize = x.shape
    
    pad = int(math.floor(window_shape[0]/2))

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

            rotations = random.randint(0,3)
            post_rotation = np.rot90(extract,rotations)
            transpose = random.randint(0,1)
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
        
        for i in range(int(pad),valN[2]-int(pad)):
            
            y_index = i
            x_index = valN[1]

            extract_int = img[x_index - pad:x_index + pad + 1, :]
            extract = extract_int[:, y_index - pad:y_index + pad + 1]

            retPatches[i-pad, :, :] = np.expand_dims(extract, axis=3)

            retLabels[i-pad, lab[x_index - pad, y_index - pad]] = 1
            

    return retPatches, retLabels
    
    
# Gets a segmentation of a single image.
def get_minibatch_segmentation(_image, _labels, _batch_size):
    
    numFrames = _image.shape[0]
    imSize = _image.shape
    
    fullRetLab = np.zeros((_batch_size*imSize[1]*imSize[2]*1, 2))  # TODO - make sure single channel doesnt harm.
    fullRetIm = np.zeros((_batch_size, imSize[1]*imSize[2]*1))
    fullProjectionMask = np.zeros((_batch_size, imSize[1]*imSize[2]*1))

    wideRetLab = np.zeros((_batch_size, imSize[1] * imSize[2], 2))

    one_over_ann_frac = []
    
    for i in range(_batch_size):
        imIdx = random.randint(0, numFrames - 1)
        retIm = _image[imIdx, :, :, 0]
        anFull = _labels[imIdx, :, :, 0]
        ids = np.unique(anFull)
        segs = len(ids)
        segIdx = ids[random.randint(0, segs - 1)]
        anSingle = (anFull == segIdx).astype(np.int8)
        retLab = anSingle

        # Using random seed voxel.
        # posIdx = np.nonzero(retLab)
        # posPix = len(posIdx[0])
        # selPix1 = posIdx[0][random.randint(0, posPix)]
        # selPix2 = posIdx[1][random.randint(0, posPix)]
        # projectionMask = np.zeros(retLab.shape)
        # projectionMask[selPix1, selPix2] = 1
    
        # Using centroid of object.
        posIdx = np.nonzero(retLab)
        posPix = len(posIdx[0])
        centroid = np.mean(posIdx, axis=1).astype(np.int64)
        projectionMask = np.zeros(retLab.shape)
        projectionMask[centroid[0], centroid[1]] = 1
        
        # Transpose and rotate?
        rotations = random.randint(0, 3)
        transpose = random.randint(0, 1)
        rotRetIm = np.rot90(retIm, rotations)
        rotRetLab = np.rot90(retLab, rotations)
        rotRetMask = np.rot90(projectionMask, rotations)
        
        if transpose == 1:
            rotRetIm = np.transpose(rotRetIm)
            rotRetLab = np.transpose(rotRetLab)
            rotRetMask = np.transpose(rotRetMask)
        
        # Reshape and inscribe to storage matrix.
        fullRetIm[i, :] = np.reshape(rotRetIm, [-1, imSize[1]*imSize[2]*1])
        fullProjectionMask[i, :] = np.reshape(rotRetMask, [-1, imSize[1]*imSize[2]*1])
        # Creating one-hot.
        catLab = np.reshape(rotRetLab, (1, -1))
        wideRetLab[i, :, :] = np.eye(2)[catLab]
    
        one_over_ann_frac.append(((retLab.shape[0] * retLab.shape[1] - float(np.sum(retLab != 0))) / float(np.sum(retLab != 0))))

    fullRetLab = np.reshape(wideRetLab, (-1, wideRetLab.shape[-1]))
    one_over_ann_frac = np.mean(one_over_ann_frac)

    return fullRetIm, fullRetLab, fullProjectionMask, one_over_ann_frac
    

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


