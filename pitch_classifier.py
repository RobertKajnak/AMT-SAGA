#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:30:51 2019

@author: hesiris
"""

import numpy as np
from RDCNN import res_net


class pitch_classifier(res_net):
    def __init__(self,params,checkpoint_prefix = 'checkpoint_pitch',metrics_prefix='metrics_pitch'):
        super().__init__(input_shapes=[(params.pitch_bands, params.pitch_frames, 1)],
                         kernel_sizes=params.kernel_size_pitch, pool_sizes=params.pool_size_pitch,
                         output_classes=1,
                         output_range = [params.pitch_low,params.pitch_high],
                         batch_size = params.batch_size,

                         convolutional_layer_count=params.convolutional_layer_count,
                         pool_layer_frequency=params.pool_layer_frequency,
                         feature_expand_frequency=params.feature_expand_frequency,
                         residual_layer_frequencies=params.residual_layer_frequencies,
                         
                         checkpoint_dir=params.checkpoint_dir, 
                         checkpoint_frequency=params.checkpoint_frequency,
                         checkpoint_prefix= checkpoint_prefix,
                         metrics_prefix = metrics_prefix,
                         metrics=[],
                         
                         logging_parent = 'AMT-SAGA')

        self.params = params
        
    def classify(self, spec, pitch_gold = None, test_phase = False):
        """ Trains, test and classifies the provided sample.
        params:
            spec: sample
            instrument_gold: if None, prediction is performed, without checkking
                the correctness of the result
            test_phase: if set to true, testing is done, otherwise training
        """
        if isinstance(spec,list):
            spec_shape = spec[0].shape
        else:
            spec_shape = spec.shape
        if spec_shape != (self.params.pitch_bands,self.params.pitch_frames):
            raise ValueError('Invalid Input shape. Expected: {} . Got: {}'.
                             format((self.params.pitch_bands,
                                     self.params.frames),spec_shape))

        if isinstance(spec,list):
            cb_x=np.zeros([0]+list(spec[0].shape)+[1])
            cb_y=np.zeros([0,1])
            for specs,label in zip(spec,pitch_gold):
                expanded_x = specs[np.newaxis,:,:,np.newaxis]
                expanded_y = np.expand_dims([label],axis=0)
                cb_x=np.concatenate((cb_x,expanded_x))
                cb_y=np.concatenate((cb_y,expanded_y))
                
            expanded = cb_x
            gold_expanded = cb_y
        else:
            expanded = spec[np.newaxis,:,:,np.newaxis]
            gold_expanded = np.expand_dims(pitch_gold,axis=0)
            
        if pitch_gold is None:
            pitch_pred = self.predict(expanded)
        else:
            if test_phase:
                pitch_pred = self.test(expanded, gold_expanded)
            else:
                pitch_pred = self.train(expanded, gold_expanded)

        return pitch_pred