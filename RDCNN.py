# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:10:28 2019

@author: Hesiris
"""

#import argparse
#import tensorflow as tf

from __future__ import absolute_import, division, print_function

import os
import logging
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.metrics import get as get_metric_by_name

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
                 metrics = ['sparse_categorical_accuracy'],
                 
                 checkpoint_dir=None,checkpoint_prefix='checkpoint',
                 checkpoint_frequency=5000,
                 metrics_prefix='metrics_logs',
                 weights_load_checkpoint_filename=None,
                 starting_checkpoint_index=1,
                 
                 logging_parent=None):
        '''Residual net
        params:    
            input_shapes: the input shape of the separate input channels. 
                The default value uses two channels.
                For a single channel use [(n,m,1)]
            output_classes: the number of output classes. specifying a value 
                of 1 will change the network to use a single output with 
                sigmoid/MSE loss instead of softmax/cross entropy
            output_range: if the output classes are classified as 1, this is 
                used to scale the output from this domain to to the 
                appropriate values

            pool_size: pooling sizes, in the same order as the input channles.
                If a single channel is present use [(n,m)]
            kernel_sizes: kernel sizes in the same order as the input channles.
                If a single channel is present use [(n,m)]

            convolutional_layer_count: total number of convolutional layers 
                in the network, not counting size scaling for the residual 
                layers and similar layers
            feature_expand_frequency: the frequency at which the number of 
                features are doubled. E.g. for convolutional_layer_count==5 
                and feature_expand_frequency==2 the feature sizes will be
                32-32-64-64-128
            pool_layer_frequency: the frequency at which the pooling layers 
                are inserted. Same logic as feature_expand_frequency. The 
                pool layers are inserted BEFORE the feature expansion
            residual_layer_frequencies: the frequency at which the the 
                residual shortcuts are insterted. Multiple frequencies can be 
                specified e.g. [2,4] will insert a shortcut between every 
                2nd layer and separately also between every 4th
            metrics: the metrics to use. See Keras documentation. 
                Use full name, aliases may not work
            
            checkpoint_dir: The directory to save checkpoints to. If set to 
                none, no operation will be performed
            checkpoint_prefix: The filename for checkpoints. 
                E.g. a prefix ='foo' will yield filenames of 
                'foo_1000.h5', 'foo_2000.h5' etc.
            checkpoint_frequency: the model weights will be saved every 
                nth batch
            weights_load_checkpoint_filename: weights will be attempted to be
                imported from this file
            starting_checkpoint_index: specify this to continue batch 
                numbering from a previous point
                
            logging_parent: sets the verbosity of the network. 
                a None, 1 or 2 will attach a logger to stdout, if no hanlders
                are attached yet. If there are, their level will be modified
                None == debug, 1==info, 2==error
                specifying a string will attempt to add a logger to 
                {logginging_parent}.res_net
                
        '''
        
        if logging_parent is None or \
            logging_parent == 1 or logging_parent==2:
            self.logger = logging.getLogger('res_net')
            self.logger.setLevel(logging.DEBUG)
            if self.logger.hasHandlers():
                ch = self.logger.handlers[0]
            else:
                ch = logging.StreamHandler()
                self.logger.addHandler(ch)
            
            if logging_parent == 1:
                ch.setLevel(logging.INFO)
            elif logging_parent == 2:    
                ch.setLevel(logging.ERROR)
            else:    
                ch.setLevel(logging.DEBUG)
                
        else:
            self.logger = logging.getLogger(logging_parent + '.res_net')
        
        
        self.batch_size = batch_size
        activation_fuction = 'sigmoid'
        self.out_func_min = 0
        self.out_func_max = 1
        self.out_func_factor = self.out_func_max - self.out_func_min
        self.output_classes = output_classes
        
        
        self.logger.debug('Creating Model structure, Tensorflow Versions: {}'.format(tf.__version__))
        for i in range(len(metrics)):
            if isinstance(metrics[i],str):
                metrics[i] = get_metric_by_name(metrics[i])
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
            
            self.logger.debug('Output scaling set to: [{},{}] -> [{},{}] ({})'.
                  format(self.out_val_min,self.out_val_max,
                         self.out_func_min,self.out_func_max,
                         activation_fuction))
        else:
            raise ValueError('Invalid output size {}'.format(str(output_classes)))
        
        model = Model(inputs = input_layers,outputs = out1)
        
        
        self.model = model

        #Create the default metrics and add it to the ones in the params
        custom_metrics = self._define_metrics(output_classes)
        self.metrics_custom = custom_metrics
        for m in custom_metrics:
            metrics.append(m)

        self.metrics_names = None
        if output_classes>1:
            self.model.compile(
                  keras.optimizers.Adagrad(),
                  loss='sparse_categorical_crossentropy',
                  metrics=metrics)
        else:
            self.model.compile(
                  keras.optimizers.Adagrad(),
                  loss='mean_squared_error',
                  metrics=metrics)

        
        if weights_load_checkpoint_filename is not None:
            
            self.logger.info('Loading Weights from {}'.format(weights_load_checkpoint_filename))
            self.load_weights(weights_load_checkpoint_filename)
            
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = checkpoint_prefix
        self.checkpoint_frequency = checkpoint_frequency
        self.metrics_prefix = metrics_prefix
        if checkpoint_frequency<1:
            self.checkpoint_dir = None
        #Counting starts from 1, otherwise the initial state is saved as well
        self.current_batch = starting_checkpoint_index if starting_checkpoint_index is not None else 1
        
        self.metrics_train = []
        self.metrics_test = []
        if self.metrics_names is None:
            self.metrics_names = self.model.metrics_names
#                self.metrics_names = ['loss'] + [m.__name__ for m in metrics]
            
        if output_classes==1:
            self.metrics_true_ind = self._get_metric_ind('y_true_scaled')
        else:
            self.metrics_true_ind = self._get_metric_ind('y_true')
            
        self.logger.debug('Using metrics:')
        self.logger.debug(self.metrics_names)
        oname = str(self.model.optimizer)
        oname = oname[oname.find('optimizers')+11:oname.find('object')-1]
        self.logger.debug('Optimizer: ' + oname)
        
    def _to_array(self,single):
        """Converts a single value (e.g. int, float, str) into an array 
        containing only that value. E.g. str-> [str]"""
        
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
        """Creates a shortcut between the two layers. 
        Layers with differing shapes can be added. For smaller layers, average
        pooling will be used, for bigger ones, values will be duplicated
        Roughly follows Deep Residual Learning for Image Recognition by
        Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

        """
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
        """Returs some pre-determined basic metrics, depending on wether 
        the output is categorical or numerical"""
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
    
    def save_metrics(self,directory='.',prefix='metrics_logs', index=None,
                     training=True, test = True, use_csv=True):
        """ Saves the metrics until now to a file.
            params:
                directory: directory to save to
                prefix: 'training.format' and 'test.format' will be added to this
                index: appended after prefix
                training: save the training metrics
                test: save the testing metrics
                use_csv: If True, format=csv (numpy.savetxt) is used. 
                    If False format=npy (numpy.save) is used
        """
        prefix+='_'
        traintest = []
        if training:
            traintest.append(('training',self.metrics_train))
        if test:
            traintest.append(('test',self.metrics_test))
            
        if index is None:
            index=''
        else:
            index=str(index) + '_'
            
        for tt in traintest:
            if use_csv:
                np.savetxt(os.path.join(directory,prefix + index + tt[0]
                            + '.csv'), tt[1], delimiter=',', 
                            header = ','.join(self.metrics_names))
            else:
                np.savez_compressed(os.path.join(directory,prefix + index + tt[0]
                        + '.npz'),metrics=tt[1],names=self.metrics_names)
        
    def load_metrics(self,directory='.',prefix='metrics_logs', index = None,
                     training=True, test = True, use_csv=True, 
                     load_metric_names=False):
        """Replaces in this instance withthe metrics stored in the 
            specified directories.
            params:
                directory: directory to load from
                prefix: files should be of format prefix + 'training.format' and 
                    prefix+'test.csv'
                index: appears after prefix
                training: load the training metrics
                test: save the testing metrics
                use_csv: If True format = csv (numpy.gendomtxt).
                    If false format = npy (numpy.load)
                load_metric_names: replace the metrics_names in the instance
                    with the header of the file
        """
        prefix += '_'
        if index is None:
            index=''
        else:
            index=str(index) + '_'
            
        if training:
            if use_csv:
                self.metrics_train = \
                    np.genfromtxt(os.path.join(directory,prefix + index + 
                                               'training.csv'),
                           delimiter=',')
                if load_metric_names:
                    with open(os.path.join(directory,prefix + index + 
                                               'training.csv'),'r') as f:
                        self.metrics_names = f.readline()[2:].split(',')
                        self.metrics_names[-1] = self.metrics_names[-1][:-1]
            else:
                npz = np.load(os.path.join(directory, prefix +
                                                             index +
                                                             'training.npz'))
                self.metrics_train = npz['metrics']
                if load_metric_names:
                    self.metrics_names = list(npz['names'])
            self.metrics_train = [list(m) for m in self.metrics_train]
        if test:
            if use_csv:
                self.metrics_test = \
                    np.genfromtxt(os.path.join(directory,prefix + index + 
                                               'test.csv'),
                           delimiter=',')
                if load_metric_names:
                        with open(os.path.join(directory,prefix + index + 
                                                   'training.csv'),'r') as f:
                            self.metrics_names = f.readline()[2:].split(',')
                            self.metrics_names[-1] = self.metrics_names[-1][:-1]
            else:
                npz = np.load(os.path.join(directory,prefix + 
                                                         index + 'test.npz'))
                self.metrics_test = npz['metrics']
                if load_metric_names:
                    self.metrics_names = list(npz['names'])
            self.metrics_test = [list(m) for m in self.metrics_test]
    
    def save_checkpoint(self):
            self.logger.info('Saving {} {}'
                             .format(self.checkpoint_prefix,
                                     self.current_batch))
            if self.current_batch-self.checkpoint_frequency>0:
                batch_start = self.current_batch-self.checkpoint_frequency
            else:
                batch_start = 0
                
            progress_str = ('{} avaraged over the last '
                            '{} batches: ').format(self.metrics_prefix,
                                    self.current_batch-batch_start)
            
            avgs = np.average(self.metrics_train[
                                batch_start:self.current_batch],axis=0)
            metr_inds = []
            for idx,mn in enumerate(self.metrics_names):
                if 'y_pred' not in mn and 'y_true' not in mn:
                    metr_inds.append((idx,mn))
            for idx,metric_name in metr_inds:
                progress_str += '{}: {:.3f} '.format(metric_name,avgs[idx])
            self.logger.info(progress_str)
                
            self.model.save_weights(
                    os.path.join(self.checkpoint_dir,
                                 self.checkpoint_prefix + 
                                 '_{}.h5'.format(self.current_batch))
                    )
            #Saving to a single file and making a backup each time to avoid 
            #possible corruption is an alternative, but considering the size
            #difference between the weights file and logs, this isn't so bad
            #either (10 MB vs 100Kb)
            self.save_metrics(self.checkpoint_dir, prefix=self.metrics_prefix, 
                              index=self.current_batch,
                              training=True, test = False, use_csv=False)
    
    def train(self,x,y):
        """Wrapper for train_on_batch. 
        Rescales output when specified in the constructor.
        Creates checkpoints when needed.
        Sample and class weights constant
        Returns the scaled predicted value
        """
#        self.logger.debug('x format = {}'.format(x[0].shape))
#        self.logger.debug('y={}'.format(y))        
        if self.output_classes == 1:
            y = self._scale_output_to_activation(y)
        
        #self.logger.debug(y)
        #TODO: Check if it is possible to only update when the error values are small
        self.metrics_train.append(
                self.model.train_on_batch(
                        x,y,sample_weight=None, class_weight=None)
                )
            
        if self.checkpoint_dir and (self.current_batch % self.checkpoint_frequency==0):
            self.save_checkpoint()
        self.current_batch += 1
        
        return self.metrics_train[-1][self.metrics_true_ind]
    
    class _fit_metrics(Callback):
        """For use with fit_generator"""
        def on_train_begin(self, logs={}):
            self.metrics_all = []
        def on_batch_end(self, batch, logs={}):
            self.metrics_all.append(logs)
    
    
    def fit_generator(self,generator,steps_per_epoch,
                      epochs=1,workers=1,use_multiprocessing=False):
        """ Starts training using a generator provided
        """
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
                
    
    def test(self,x,y, use_predict=False):
        """Wrapper for test_on_batch. Sample weights constant
        Returns the scaled predicted value
        params:
            use_predict: if set to true the model will use predict_on_batch 
                instead of train_on_batch. Does not affect logs. Can be useful
                to circumvent batch size requirement in tests"""
        
        if self.output_classes == 1:
            y = self._scale_output_to_activation(y)
        
        #self.logger.debug(y)
        if use_predict:
            y_pred = self.model.predict_on_batch(x)
            for i in range(y.shape[0]):
                nm = []
                for m in self.metrics_names:
                    if m=='y_pred':
                        nm.append(np.argmax(y_pred[i]))
                    elif m=='y_true':
                        nm.append(y[i])
                    else:
                        nm.append(0)
                self.metrics_test.append(nm)
#                self.metrics_test.append([m(x,sample) for m in self.metrics_custom])
        else:
            self.metrics_test.append(
                    self.model.test_on_batch(x, y, sample_weight=None)
                    )
        
        return self.metrics_test[-1][self.metrics_true_ind]
        
    def predict(self,x):
        """Calculates the output for input x"""
        y = self.model.predict(x)
        if self.output_classes == 1:
            return self._scale_activation_to_output(y)
        else:
            return y
    
    def _get_metric_ind(self,metric_string):
        """Return the index (in the name list) of the metric requested.
        Returns None if metric is not found"""
        for ind,m_name in enumerate(self.metrics_names):
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
        """Plots the metrics over the training and test data, where available.
            params:
                metrics_to_plot: the indices of the metrics to use. The 
                    metrics_names property contains the names. See also 
                    get_metric_test, get_metric_test and _get_metric_ind 
                    functions
                moving_average_window: the window to use to smooth the data
                    1 will result in no smoothing
                filename_training. If specified a png of the training data 
                    will be saved to the specified file
                filename_test. If specified a png of the test data 
                    will be saved to the specified file
        """

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
                    plt.title(self.metrics_names[m] + ' - ' + title)
                
                plt.tight_layout()        
                if title == 'training' and filename_training:
                    plt.savefig(filename_training, dpi=300)
                if title == 'test' and filename_test:
                    plt.savefig(filename_test, dpi=300)
            
            
        
    def load_weights(self,filename):
        """Replaces current weights with the ones in the file specified.
            Should be an hd5 file, created by the checkpointing part.
            Class should still be defined using the correct parameters."""
        self.model.load_weights(filename)
        
    def plot_model(self,to_file='model.png'):
        """Plots the structure of the model. If filename is specified, it is
        also saved to a the specified path/filename"""
        keras.utils.plot_model(self.model,to_file, show_layer_names=False,
                               show_shapes=True)#, dpi=140)
               
    