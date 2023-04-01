import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm #progress bar library
import pickle

#loading the pre-trained ResNet50 model from Keras
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#freezing all of its layers so that they are not trainable.
model.trainable = False

#adding a global max pooling layer to the model
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

