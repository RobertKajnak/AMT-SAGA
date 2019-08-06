#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:30:51 2019

@author: hesiris
"""

from RDCNN import res_net
from util_train_test import check_shape, list_to_nd_array

class pitch_classifier(res_net):
    def __init__(self,params,checkpoint_prefix = 'checkpoint_pitch',
                 metrics_prefix='metrics_pitch'):
        super().__init__(input_shapes=[(params.pitch_bands, params.pitch_frames, 1)],
                         kernel_sizes=params.kernel_size_pitch, 
                         pool_sizes=params.pool_size_pitch,
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
            pitch_gold: if None, prediction is performed, without checkking
                the correctness of the result
            test_phase: if set to true, testing is done, otherwise training
        """
        check_shape(spec,self.params.pitch_bands,self.params.pitch_frames)
        
        expanded, gold_expanded = list_to_nd_array(spec,pitch_gold)
        
        if pitch_gold is None:
            pitch_pred = self.predict(expanded)
        else:
            if test_phase:
                pitch_pred = self.test(expanded, gold_expanded, use_predict=True)
            else:
                pitch_pred = self.train(expanded, gold_expanded)

        return pitch_pred