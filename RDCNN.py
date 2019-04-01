# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:10:28 2019

@author: Hesiris
"""

#import argparse
#import tensorflow as tf

from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import os

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import AveragePooling2D

'''Reinforcement learning? Subtract at most 6 instrument, the remaining sound is the 
reward function
'''

class res_net:
    def __init__(self,weights_checkpoint_filename=None,use_residuals = True):
        '''Residual net
            weights can be imported from a checkpoint, if a filename is specified
        '''

        print('Creating Model structure, Tensorflow Versions: {}'.format(tf.__version__))
        
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
            
        #44100 with 4098 buckets, leads to 44.93 samples/sec -> crop or extend to .5 sec
        #=> 20 wide, 2049 high
        input_shape_lin = (20,2049,1,)
        #after conversion to  logarithmic -- specificly to mel scale
        #256 based on the ratio of the recommended fft-mel (freq) window size
        input_shape_mel = (20,256,1,)
        
        
        #a width  of 3 would be equal to 60ms which is in line with the minimum dration
        #for instrument recognition on trickier instruments for humans
        kernel_size_lin = (3,32)
        kernel_size_mel = (3,8)
        pool_size = (2,5)
        chanDim=-1
        
        #2 channels since hearing and related components are logarithmic, 
        # but instrument physics are linear 
        #1 input with linear spacing
        #1 input with logarithmic spacing
        #Based on Phase-based Harmonic/Percussive Separation by 
        # Estefan´ıa Cano, Mark Plumbley, Christian Dittmar, phase infomration 
        #could aslo be used
        in_lin = Input(shape=input_shape_lin)
        
        p1 = in_lin
        base_f = 32
        for i in range(1,4):
            p0 = p1
            for j in range(3):
                p1 = Conv2D(base_f*i, kernel_size_lin, padding="same",activation='relu')(p1)
                p1 = BatchNormalization(axis=chanDim)(p1)
            if use_residuals:
                p1 = self._add_shortcut(p0,p1)
            p1 = MaxPooling2D(pool_size=pool_size)(p1)
            p1 = Dropout(0.25)(p1)
        p1 = Flatten()(p1)
        

        
        in_mel = Input(shape=input_shape_mel)
        p2=  in_mel
        base_f = 32
        for i in range(1,4):
            p0 = p2
            for j in range(3):
                p2 = Conv2D(base_f*i, kernel_size_mel, padding="same",activation='relu')(p2)
                p2 = BatchNormalization(axis=chanDim)(p2)
            p2 = MaxPooling2D(pool_size=pool_size)(p2)
            if use_residuals:
                p2 = self._add_shortcut(p0,p2)
            p2 = Dropout(0.25)(p2)
        p2 = Flatten()(p2)
        
        
        #p22 = Flatten()(p22)
        
        cted = Concatenate()([p1,p2])
        m = Dense(128)(cted)
        out1 = Activation("softmax")(m)
        
        model = Model(inputs = [in_lin,in_mel],outputs = out1)
        

        
        self.model = model
        
        self.model.compile(
              keras.optimizers.SGD(lr=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        
        if weights_checkpoint_filename is not None:
            print('Loading Weights...')
            self.load_weights(weights_checkpoint_filename)
    
        #self._load()
        
    def _add_shortcut(self,layer_from,layer_to):
#        def padding(sh1,sh2):
#            p = []
#            for i,j in zip(sh1,sh2):
#                p.append((int(np.floor(np.int(i-j)/2)),
#                    int(np.ceil(np.int(i-j)/2))))
#            return p
        
        sh1 = [np.int(layer_from.shape[1]),np.int(layer_from.shape[2]),np.int(layer_from.shape[3])]#layer_from.shape[1:]
        sh2 = [np.int(layer_to.shape[1]),np.int(layer_to.shape[2]),np.int(layer_to.shape[3])]
        if sh1==sh2:
            return Add()([layer_from,layer_to])
        else:
            _strides = (np.int(np.floor(sh1[0]/sh2[0])),
                      np.int(np.floor(sh1[1]/sh2[1])))
            
            intermediate = layer_from
            if sh1[2]!=sh2[2]:
                intermediate = Conv2D(np.int(layer_to.shape[-1]),kernel_size=(1,1),padding='valid')(intermediate)
            if sh1[:2] != sh2[:2]:
                intermediate = AveragePooling2D(_strides)(intermediate)
            
            return Add()([intermediate,layer_to])
            
        
    def _load(self):
        fashion_mnist = keras.datasets.fashion_mnist

        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()
        
        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0
        
        self.train_images = self.train_images.reshape(self.train_images.shape[0],28,28,1)
        self.test_images = self.test_images.reshape(self.test_images.shape[0],28,28,1)
        
        #self.train_labels = keras.utils.to_categorical(self.train_labels,10)
        #self.test_lables = keras.utils.to_categorical(self.test_labels,10)
        print('Loaded {} images with shape: {}'.format(self.train_images.shape[0],self.train_images.shape[1:]))
        
        
    def train(self,epochs=10):
        self.history = AccuracyHistory()
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        checkpoint = keras.callbacks.ModelCheckpoint('checkpoints/checkpoint{epoch:02d}.h5', monitor='val_acc', period=1)
        self.model.fit(self.train_images, self.train_labels, epochs=epochs,verbose = 1,
                       batch_size = 32, callbacks = [self.history, checkpoint])
    
    def test(self):
#        self.test_loss, self.test_acc = self.model.evaluate(self.test_images, 
#                                                            self.test_labels,verbose = 1)
#
#        print('Test accuracy:', self.test_acc)
        self.preds = self.model.predict(self.test_images)
        print(classification_report(self.test_labels, self.preds.argmax(axis=1),
                                target_names=self.class_names))
                
    def load_weights(self,filename):
        self.model.load_weights(filename)
        
    def plot_model(self,to_file='model.png'):
        keras.utils.plot_model(self.model,to_file, show_layer_names=False,
                               show_shapes=True)#, dpi=140)
        
    def show_image(self):
        plt.figure()
        plt.imshow(self.train_images[0])
        plt.colorbar()
        plt.grid(False)
        plt.show()
        
        
class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        
    