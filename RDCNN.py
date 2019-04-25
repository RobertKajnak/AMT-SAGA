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
    def __init__(self,checkpoint_dir=None,checkpoint_frequency=1000,
                 weights_checkpoint_filename=None,starting_checkpoint_index=1,
                 input_shape_lin = (20,2049,1,), input_shape_mel = (20,256,1,),
                 output_classes = 128,
                 kernel_size_lin = (3,32),       kernel_size_mel = (3,8),
                 pool_size = (2,5),
                 convolution_stack_size = 3,
                 layer_stack_count = 4,
                 use_residuals = True):
        '''Residual net
        params:    
            checkpoint_dir: The directory to save checkpoints to. If set to none,
                no operation will be performed
            checkpoint_frequency: the model weights will be saved every nth batch
            weights_checkpoint_filename: weights will be attempted to be imported from this file
            starting_checkpoint_index: specify this to continue couning from a previous point
            input_shape_lin and input_shape_mel: the input layer shapes for linear and mel spacing
                set either to None to use a single channel
            pool_size: pooling size
            convolution_stack_size: the number of convolutional laers next to each others
            layer_stack_count: how many times the convolution_stack_size structure is repeated
            use_residuals: creates residual layers over convolutional stacks
        '''

        print('Creating Model structure, Tensorflow Versions: {}'.format(tf.__version__))
        
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
            
        #44100 with 4098 buckets, leads to 44.93 samples/sec -> crop or extend to .5 sec
        #=> 20 wide, 2049 high
        
        #after conversion to  logarithmic -- specificly to mel scale
        #256 based on the ratio of the recommended fft-mel (freq) window size
        
        
        
        #a width  of 3 would be equal to 60ms which is in line with the minimum dration
        #for instrument recognition on trickier instruments for humans
        chanDim=-1
        
        #2 channels since hearing and related components are logarithmic, 
        # but instrument physics are linear 
        #1 input with linear spacing
        #1 input with logarithmic spacing
        #Based on Phase-based Harmonic/Percussive Separation by 
        # Estefan´ıa Cano, Mark Plumbley, Christian Dittmar, phase infomration 
        #could aslo be used
        
        if input_shape_lin:
            in_lin = Input(shape=input_shape_lin,batch_size=1)
            p1 = in_lin
            base_f = 32
            
            for i in range(layer_stack_count):
                p0 = p1
                for j in range(convolution_stack_size):
                    p1 = Conv2D(base_f*(i+1), kernel_size_lin, padding="same",activation='relu')(p1)
                    p1 = BatchNormalization(axis=chanDim)(p1)
                if use_residuals:
                    p1 = self._add_shortcut(p0,p1)
                p1 = MaxPooling2D(pool_size=pool_size)(p1)
                p1 = Dropout(0.25)(p1)
            p1 = Flatten()(p1)
        

        if input_shape_mel:
            in_mel = Input(shape=input_shape_mel,batch_size=1)
            p2=  in_mel
            base_f = 32
            
            for i in range(layer_stack_count):
                p0 = p2
                for j in range(convolution_stack_size):
                    p2 = Conv2D(base_f*(i+1), kernel_size_mel, padding="same",activation='relu')(p2)
                    p2 = BatchNormalization(axis=chanDim)(p2)
                if use_residuals:
                    p2 = self._add_shortcut(p0,p2)
                p2 = MaxPooling2D(pool_size=pool_size)(p2)
                p2 = Dropout(0.25)(p2)
            p2 = Flatten()(p2)
        
        
        #p22 = Flatten()(p22)
        if input_shape_lin and input_shape_mel:
            cted = Concatenate()([p1,p2])
        else:
            cted = p1 if input_shape_lin else p2
            
        m = Dense(output_classes)(cted)
        out1 = Activation("softmax")(m)
        
        if input_shape_lin and input_shape_mel: 
            model = Model(inputs = [in_lin,in_mel],outputs = out1)
        else:
            in_single = in_lin if input_shape_lin else in_mel
            model = Model(inputs = [in_single],outputs = out1)

        
        self.model = model
        
        self.model.compile(
              keras.optimizers.SGD(lr=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        
        if weights_checkpoint_filename is not None:
            print('Loading Weights...')
            self.load_weights(weights_checkpoint_filename)
            
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency
        if checkpoint_frequency<1:
            self.checkpoint_dir = None
        #Counting starts from 1, otherwise the initial state is saved as well
        self.current_batch = starting_checkpoint_index if starting_checkpoint_index is not None else 1
        
        self.metrics_train = []
        self.metrics_test = []
        
    def _add_shortcut(self,layer_from,layer_to):    
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
            
    
    def train(self,x,y):
        """Wrapper for train_on_batch. Also creates checkpoints when needed.
        Sample and class weights constant"""
        
        self.metrics_train.append(
                self.model.train_on_batch(
                        x,y,sample_weight=None, class_weight=None)
                )
        
        if self.checkpoint_dir and (self.current_batch % self.checkpoint_frequency==0):
            print('Saving checkpoint {}'.format(self.current_batch))
            self.model.save_weights(
                    os.path.join(self.checkpoint_dir,
                                 'checkpoint_{}.h5'.format(self.current_batch))
                    )
        self.current_batch += 1
    
    def test(self,x,y):
        """Wrapper for test_on_batch. Sample weights constant"""
        self.metrics_test.append(
                self.model.test_on_batch(x, y, sample_weight=None)
                )
        
    def predict(self,x):
        """Calculates the output for input x"""
        return self.model.predict_on_batch(x)
        
    def report(self, filename=None):
        """Prints the classification report for both training and test, if available.
        If the filename is specifies, the report(s) is/are saved to a csv file"""
        if self.metrics_train:
            print(classification_report(self.train, self.preds.argmax(axis=1),
                                target_names=self.class_names))
        if self.metrics_test:
            print(classification_report(self.test_labels, self.preds.argmax(axis=1),
                                target_names=self.class_names))
        #model.metrics_names
        if filename:
            raise NotImplementedError('Yet')
        
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
        
    