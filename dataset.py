import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np


#Helper function to load in the training set of images and resize them all to the given size

def load_train(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []

    print('Going to read training images')
    for fields in classes:   
        index = classes.index(fields)
        print('Now going to read {} files (Index: {})'.format(fields, index))
        path = os.path.join(train_path, fields, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)

    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls


#A class containing various information about the training set
class DataSet(object):

  def __init__(self, images, labels, img_names, cls):
    self._num_examples = images.shape[0]

    self._images = images
    self._labels = labels
    self._img_names = img_names
    self._cls = cls
    self._epochs_done = 0
    self._index_in_epoch = 0

  #Return the set of images
  def images(self):
    return self._images
	
  #Return the set of 1-hot class vectors
  def labels(self):
    return self._labels
	
  #Return the set of image filenames
  def img_names(self):
    return self._img_names
	
  #Return the set of class labels
  def cls(self):
    return self._cls
	
  #Return the number of examples in the training set
  def num_examples(self):
    return self._num_examples

  #Return the number of epochs that have been completed
  def epochs_done(self):
    return self._epochs_done

  #Retrieve the next batch of data to pass to the neural network
  #Inputs: 
  #batch_size: The number of training examples to return in a batch
  #Outputs:
  #the images in the next batch, the 1-hot class vectors for the next batch, the filenames in the next batch, and the class labels in the next batch
  def next_batch(self, batch_size):
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


#Code to read in training data and put it in a decent format for learning
#Inputs: 
#train_path: a string containing the path to the training data
#image_size: image size (in pixels) that each training image will be resized to. Resulting dimensions will be image_size x image_size
#classes: an array containing each of the classes. For this assignment, it would be ['pembroke', 'cardigan']
#validation_size: Float corresponding to the proportion of the training set to set aside for validation. This is different than the test set!
#Returns:
#data_sets: a DataSet object containing images, labels, 1-hot label vectors, filenames, as well as training and validation data. 
def read_train_sets(train_path, image_size, classes, validation_size):
  class DataSets(object):
    pass
  data_sets = DataSets()

  images, labels, img_names, cls = load_train(train_path, image_size, classes)
  images, labels, img_names, cls = shuffle(images, labels, img_names, cls)  

  if isinstance(validation_size, float):
    validation_size = int(validation_size * images.shape[0])

  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]
  validation_img_names = img_names[:validation_size]
  validation_cls = cls[:validation_size]

  train_images = images[validation_size:]
  train_labels = labels[validation_size:]
  train_img_names = img_names[validation_size:]
  train_cls = cls[validation_size:]

  data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
  data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

  return data_sets