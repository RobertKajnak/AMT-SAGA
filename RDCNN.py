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
import csv

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
#from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import AveragePooling2D



class res_net:
    def __init__(self,
                 input_shapes = [(20,2049,1,),(20,256,1,)],
                 output_classes = 128,
                 output_range=[0,128],
                 batch_size = 1,
                 
                 kernel_sizes = [(3,32),(3,8)],
                 pool_sizes = [(2,5),(2,5)],
                 
                 convolutional_layer_count = 15,
                 feature_expand_frequency = 6,
                 pool_layer_frequency = 6,
                 residual_layer_frequencies = 2,
                 metrics = ['acc'],
                 
                 checkpoint_dir=None,checkpoint_prefix='checkpoint',
                 checkpoint_frequency=5000,
                 weights_load_checkpoint_filename=None,
                 starting_checkpoint_index=1,
                 
                 verbose = True):
        '''Residual net
        params:    
            input_shapes: the input shape of the separate input channels. The default value uses two channels.
                For a single channel use [(n,m,1)]
            output_classes: the number of output classes. specifying a value of 1 will change the network to use a
                single output with sigmoid/MSE loss instead of softmax/cross entropy
            output_range: if the output classes are classified as 1, this is used
                to scale the output from this domain to to the appropriate values

            pool_size: pooling sizes, in the same order as the input channles.
                If a single channel is present use [(n,m)]
            kernel_sizes: kernel sizes in the same order as the input channles.
                If a single channel is present use [(n,m)]

            convolutional_layer_count: total number of convolutional layers in the network, not counting size scaling
                for the residual layers and similar layers
            feature_expand_frequency: the frequency at which the number of features are doubled.
                E.g. for convolutional_layer_count==5 and feature_expand_frequency==2 the feature sizes will be
                32-32-64-64-128
            pool_layer_frequency: the frequency at which the pooling layers are inserted.
                Same logic as feature_expand_frequency. The pool layers are inserted BEFORE the feature expansion
            residual_layer_frequencies: the frequency at which the the residual shortcuts are insterted.
                Multiple frequencies can be specified e.g. [2,4] will insert a shortcut between every 2nd layer and
                separately also between every 4th
            metrics: the metrics to use. See Keras documentation
            
            checkpoint_dir: The directory to save checkpoints to. If set to none,
                no operation will be performed
            checkpoint_prefix: The filename for checkpoints. E.g. a prefix ='foo' will
                yield filenames of 'foo_1000.h5', 'foo_2000.h5' etc.
            checkpoint_frequency: the model weights will be saved every nth batch
            weights_load_checkpoint_filename: weights will be attempted to be imported from this file
            starting_checkpoint_index: specify this to continue couning from a previous point
        '''
        activation_fuction = 'sigmoid'
        self.out_func_min = 0
        self.out_func_max = 1
        self.out_func_factor = self.out_func_max - self.out_func_min
        self.output_classes = output_classes
        self.verbose = verbose
        if verbose:
            print('Creating Model structure, Tensorflow Versions: {}'.format(tf.__version__))
        
        residual_layer_frequencies = self._to_array(residual_layer_frequencies)
        #output_classes = self._to_array(output_classes)
            
        #44100 with 4098 buckets, leads to 44.93 samples/sec -> crop or extend to .5 sec
        #=> 20 wide, 2049 high
        
        #after conversion to  logarithmic -- specificly to mel scale
        #256 based on the ratio of the recommended fft-mel (freq) window size
        
        
        
        #a width  of 3 would be equal to 60ms which is in line with the minimum dration
        #for instrument recognition on trickier instruments for humans
        
        
        #2 channels since hearing and related components are logarithmic, 
        # but instrument physics are linear 
        #1 input with linear spacing
        #1 input with logarithmic spacing
        #Based on Phase-based Harmonic/Percussive Separation by 
        # Estefan´ıa Cano, Mark Plumbley, Christian Dittmar, phase infomration 
        #could aslo be used
        
        input_layers = []
        layers = []
        for iidx, input_shape in enumerate(input_shapes):
            in_lay = Input(shape=input_shape,batch_size=batch_size)
            input_layers.append(in_lay)
            p1 = in_lay
            feature_out = 32
            p0 = [p1]*len(residual_layer_frequencies)
            
            for i in range(1,convolutional_layer_count+1):
                p1 = Conv2D(feature_out, kernel_sizes[iidx], padding="same")(p1)
                p1 = BatchNormalization()(p1)
                p1 = Activation(activation_fuction)(p1)
                for ri in range(len(residual_layer_frequencies)):
                    if i%residual_layer_frequencies[ri] == 0:
                        p1 = self._add_shortcut(p0[ri],p1)
                        p1 = BatchNormalization()(p1)
                        p0[ri] = p1
                if pool_layer_frequency and i%pool_layer_frequency==0:
                    p1 = MaxPooling2D(pool_size=pool_sizes[iidx])(p1)
                    #p1 = Dropout(0.25)(p1)
                if feature_expand_frequency and i%feature_expand_frequency==0:
                    feature_out *= 2
            
            #p1 = BatchNormalization()(p1) #TODO: Only do if the previous one isn't normalization
            p1 = Flatten()(p1)
            layers.append(p1)
        
        
        #p22 = Flatten()(p22)
        if len(input_shapes)>1:
            cted = Concatenate()(layers)
        else:
            cted = layers[0]
            
        
        m = Dense(300)(cted)
        m = Activation(activation_fuction)(m)
        m = Dense(output_classes)(m)


        
        if output_classes>1:
            out1 = Activation("softmax")(m)
        elif output_classes==1:
            out1 = Activation(activation_fuction)(m)
            self.out_val_min = output_range[0]
            self.out_val_max = output_range[1]
            self.out_val_factor = output_range[1]-output_range[0]
            if verbose:
                print('Output scaling set to: [{},{}] -> [{},{}] ({})'.
                      format(self.out_val_min,self.out_val_max,
                             self.out_func_min,self.out_func_max,
                             activation_fuction))
        else:
            raise ValueError('Invalid output size {}'.format(str(output_classes)))
        
        model = Model(inputs = input_layers,outputs = out1)
        
        
        self.model = model

        #Create the default metrics and add it to the ones in the params
        custom_metrics = self._define_metrics(output_classes)
        for m in custom_metrics:
            metrics.append(m)

        if output_classes>1:
            self.model.compile(
                  keras.optimizers.SGD(lr=0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=metrics)
        else:
            self.model.compile(
                  keras.optimizers.SGD(lr=0.01),
                  loss='mean_squared_error',
                  metrics=metrics)

        
        if weights_load_checkpoint_filename is not None:
            if verbose:
                print('Loading Weights...')
            self.load_weights(weights_load_checkpoint_filename)
            
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = checkpoint_prefix
        self.checkpoint_frequency = checkpoint_frequency
        if checkpoint_frequency<1:
            self.checkpoint_dir = None
        #Counting starts from 1, otherwise the initial state is saved as well
        self.current_batch = starting_checkpoint_index if starting_checkpoint_index is not None else 1
        
        self.metrics_train = []
        self.metrics_test = []
        if output_classes==1:
            self.metrics_true_ind = self._get_metric_ind('y_true_scaled')
        else:
            self.metrics_true_ind = self._get_metric_ind('y_true')
        
        
    def _to_array(self,single):
        #As in single value, not precision
        if single is None:
            single = []
        else:
            try:
                len(single)
            except TypeError:
                if single<1:
                    single = []
                else:
                    single = [single]
        return single
        
    def _scale_output_to_activation(self,x):
        return ((x - self.out_val_min)/self.out_val_factor)* \
                self.out_func_factor + self.out_func_min
    
    def _scale_activation_to_output(self,x):
        return ((x-self.out_func_min)/self.out_func_factor) * \
                self.out_val_factor+self.out_val_min
    
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
            
            intermediate = BatchNormalization()(intermediate)
            return Add()([intermediate,layer_to])
            
    def _define_metrics(self,output_classes):
        if output_classes>1:
            def y_pred(y_true, y_pred):
                return keras.backend.argmax(y_pred)           
            def y_true (y_true, y_pred):
                return y_true
            custom_metrics = [y_pred,y_true]
        else:
            def mse(y_true, y_pred):
                return (y_pred-y_true)**2
            def mse_scaled(y_true, y_pred):
                return (self._scale_activation_to_output(y_pred)
                        - self._scale_activation_to_output(y_true))**2
            def y_pred(y_true, y_pred):
                return y_pred
            def y_true(y_true, y_pred):
                return y_true 
            def y_pred_scaled(y_true, y_pred):
                return self._scale_activation_to_output(y_pred)
            def y_true_scaled(y_true, y_pred):
                return self._scale_activation_to_output(y_true)
            custom_metrics = [mse,mse_scaled,
                              y_pred,y_true,
                              y_pred_scaled,y_true_scaled]
        return custom_metrics
    
    def train(self,x,y):
        """Wrapper for train_on_batch. 
        Rescales output when specified in the constructor.
        Creates checkpoints when needed.
        Sample and class weights constant
        Returns the scaled predicted value
        """
#        print('x format = {}'.format(x[0].shape))
#        print('y={}'.format(y))        
        if self.output_classes == 1:
            y = self._scale_output_to_activation(y)
        
        #print(y)
        #TODO: Check if it is possible to only update when the error values are small
        self.metrics_train.append(
                self.model.train_on_batch(
                        x,y,sample_weight=None, class_weight=None)
                )
        
        if self.checkpoint_dir and (self.current_batch % self.checkpoint_frequency==0):
            if self.verbose:
                print('Saving checkpoint {}'.format(self.current_batch))
                progress_str = 'Current progress over last checkpoint ({} batches): '.format(self.checkpoint_frequency)
                avgs = np.average(self.metrics_train[
                                self.current_batch-self.checkpoint_frequency:
                                self.current_batch],axis=0)
                for idx,metric_name in enumerate(self.model.metrics_names[:-2]):
                    progress_str += '{}: {:.3f} '.format(metric_name,avgs[idx])
                print(progress_str)
                
            self.model.save_weights(
                    os.path.join(self.checkpoint_dir,
                                 self.checkpoint_prefix + '_{}.h5'.format(self.current_batch))
                    )
        self.current_batch += 1
        
        return self.metrics_train[-1][self.metrics_true_ind]
    
    class _fit_metrics(Callback):
        def on_train_begin(self, logs={}):
            self.metrics_all = []
        def on_batch_end(self, batch, logs={}):
            self.metrics_all.append(logs)
    
    
    def fit_generator(self,generator,steps_per_epoch,
                      epochs=1,workers=1,use_multiprocessing=False):
        #TODO: ADD Checkpointing
        fit_metrics = self._fit_metrics()
        self.model.fit_generator(generator, steps_per_epoch=steps_per_epoch,
                                 epochs=epochs, verbose=0, 
                                callbacks=[fit_metrics], 
                                class_weight=None, 
                                max_queue_size=10, 
                                workers=workers, 
                                use_multiprocessing=use_multiprocessing)
        
        self.metrics_train += [list(metric_batch.values())[2:] for metric_batch in fit_metrics.metrics_all]
        if self.output_classes>1:
            ind_true = self._get_metric_ind('y_true')
            ind_pred = self._get_metric_ind('y_pred')
            for i in range(len(self.metrics_train)):
                self.metrics_train[i][ind_true] = int(self.metrics_train[i][ind_true])
                self.metrics_train[i][ind_pred] = int(self.metrics_train[i][ind_pred])
                
    
    def test(self,x,y):
        """Wrapper for test_on_batch. Sample weights constant
        Returns the scaled predicted value"""
        
        if self.output_classes == 1:
            y = self._scale_output_to_activation(y)
        
        #print(y)
        self.metrics_test.append(
                self.model.test_on_batch(x, y, sample_weight=None)
                )
        
        return self.metrics_test[-1][self.metrics_true_ind]
        
    def predict(self,x):
        """Calculates the output for input x"""
        return self.__scale_activation_to_output(self.model.predict_on_batch(x))
    
    def _get_metric_ind(self,metric_string):
        for ind,m_name in enumerate(self.model.metrics_names):
            if metric_string==m_name:
                break;
        else:
            return None
        return ind
    
    def get_metric_train(self,metric):
        """Returns an array containing the datapoints for the metric
        params:
            metric: can be either a string e.g. 'loss' or numerical e.g. 0 
        """
        if isinstance(metric, str):
            ind = self._get_metric_ind(metric)
        else:
            ind = metric
        if ind == None:
            return None
        else:
            return np.array([val[ind] for val in self.metrics_train])
            
    def get_metric_test(self,metric):
        """Returns an array containing the datapoints for the metric
        params:
            metric: can be either a string e.g. 'loss' or numerical e.g. 0 
        """
        if isinstance(metric, str):
            ind = self._get_metric_ind(metric)
        else:
            ind = metric
        if ind == None:
            return None
        else:
            return np.array([val[ind] for val in self.metrics_test])

    def report(self, class_names = None,
               training=False, test = True,
               filename_training=None, filename_test = None):
        """Prints the training and test report 
        For single class networks, only the test is considered"""
        
        if self.output_classes==1:
            mse_scaled = np.mean(self.get_metric_test('mse_scaled'))
            mse_not_scaled = np.mean(self.get_metric_test('mse'))
            print('MSE for test data: scaled {:.3f}, not scaled: {}'.
                  format(mse_scaled,mse_not_scaled))
            if filename_test:
                f = open(filename_test, "w")
                f.write(str(mse_scaled) + '\n' + str(mse_not_scaled))
                f.close()
        else:
            def save_report(fn,rep):
                 with open(fn, 'w', newline='') as csvfile:
                    fieldnames = [''] +list(rep[list(rep.keys())[0]].keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                    writer.writeheader()
                    for key, value in rep.items():
                        d = {'':key}
                        d = {**d,**value}
                        writer.writerow(d)
                    
            if training:
                print('\nTraining Results:  ')
                true = self.get_metric_train('y_true')
                pred = self.get_metric_train('y_pred')
                print(classification_report(true,pred))
                if filename_training:
                    save_report(filename_training,
                                classification_report(true,pred,output_dict = True,
                                                      target_names = class_names))
                
            if test:
                print('\nTest Results:  ')
                true = self.get_metric_test('y_true')
                pred = self.get_metric_test('y_pred')
                print(classification_report(true,pred))
                if filename_test:
                    save_report(filename_test,
                                classification_report(true,pred,output_dict = True,
                                                      target_names = class_names))
            
        
        
    def plot(self,metrics_to_plot=[0,1],moving_average_window=10,
             filename_training=None, filename_test=None):
        """Prints the classification report for both training and test, if available.
        If the filename is specifies, the report(s) is/are saved to a csv file"""

        def moving_average(x):        
            r_list = []
            ma_sum = np.average(x[:moving_average_window])
            for i in range(moving_average_window):
                #ma_sum += x[i]
                #r_list.append(ma_sum/(i+1))
                r_list.append(ma_sum)
            ma_sum *= moving_average_window
            for i in range(moving_average_window,len(x)):
                ma_last = x[i-moving_average_window]
                ma_sum = ma_sum - ma_last+x[i]
                r_list.append(ma_sum/moving_average_window)
            return r_list

        titles = ['test','training'] #It's backwards because .pop is used
        for metric in [self.metrics_train,self.metrics_test]:
            N = len(metrics_to_plot)
            plt.figure(figsize=(9, 4*N))
            title = titles.pop()#Kinky, I know
            if metric:
                for i,m in enumerate(metrics_to_plot):
                    plt.subplot(N,1,i+1)
                    if moving_average_window>1:
                        ma = moving_average([met[m] for met in metric])
                        ma = moving_average(ma)
                    else:
                        ma = [met[m] for met in metric]
                    plt.plot(ma)
                    plt.xlabel('Batch')
                    plt.title(self.model.metrics_names[m] + ' - ' + title)
                
                plt.tight_layout()        
                if title == 'training' and filename_training:
                    plt.savefig(filename_training, dpi=300)
                if title == 'test' and filename_test:
                    plt.savefig(filename_test, dpi=300)
            
            
        
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
        

        
    