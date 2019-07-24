#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:30:51 2019

@author: hesiris
"""

from RDCNN import res_net
from util_train_test import check_shape, list_to_nd_array

class VelocityClassifier(res_net):
    def __init__(self,params,checkpoint_prefix = 'checkpoint_velocity',
                 metrics_prefix='metrics_velocity'):
        super().__init__(input_shapes=[(params.instrument_bands, params.instrument_frames, 1)],
                         kernel_sizes=params.kernel_size_instrument, 
                         pool_sizes=params.pool_size_instrument,
                         output_classes=1,
                         output_range = [params.velocity_min,params.velocity_max],
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
        
    def classify(self, spec, velocity_gold = None, test_phase = False):
        """ Trains, test and classifies the provided sample.
        params:
            spec: sample
            velocity_gold: if None, prediction is performed, without checkking
                the correctness of the result
            test_phase: if set to true, testing is done, otherwise training
        """
        check_shape(spec,self.params.pitch_bands,self.params.pitch_frames)
        
        expanded, gold_expanded = list_to_nd_array(spec,velocity_gold)
        
        if velocity_gold is None:
            velocity_pred = self.predict(expanded)
        else:
            if test_phase:
                velocity_pred = self.test(expanded, gold_expanded)
            else:
                velocity_pred = self.train(expanded, gold_expanded)

        return velocity_pred