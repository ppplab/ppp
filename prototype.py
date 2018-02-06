from numpy import array
from numpy import argmax
from numpy.random import laplace
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class Prototype:
  def __init__(self, epsilon=1):
    self.epsilon = epsilon

  def one_hot(self, data):
    """ Encode data with one-hot representation. """
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = label_encoder.fit_transform(data).reshape(len(data), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded
  
  def randomize(self, oh_data, private=True):
    """ Add Laplace noise. """
    for row in oh_data:
      for i in range(len(row)):
        b = laplace(scale=1.0/self.epsilon) if private else 0
        row[i] += b
    return oh_data

  def get_histogram(self, data, private=True):
    """ Return count of each unique element in data. """
    oh = self.one_hot(data)
    return np.sum(self.randomize(oh, private=private), axis=0)

