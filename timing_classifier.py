#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:25:19 2019

@author: hesiris
"""

from RDCNN import res_net
from util_train_test import check_shape, list_to_nd_array


class timming_classifier(res_net):
    def __init__(self, params,checkpoint_prefix = 'checkpoint_timing',
                 metrics_prefix='metrics_timing'):
        super().__init__(input_shapes=[(params.timing_bands, params.timing_frames, 1)],
                         kernel_sizes=params.kernel_size_timing, 
                         pool_sizes=params.pool_size_timing,
                         output_classes=1,
                         output_range = [0,params.timing_frames],
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

    def classify(self, spec, timing_gold = None, test_phase = False):
        """ Trains, test and classifies the provided sample.
        params:
            spec: sample
            timing_gold: if None, prediction is performed, without checkking
                the correctness of the result
            test_phase: if set to true, testing is done, otherwise training
        """
        check_shape(spec,self.params.timing_bands,self.params.timing_frames)

        expanded, gold_expanded = list_to_nd_array(spec,timing_gold)
            
        if timing_gold is None:
            timing_pred = self.predict(expanded)
        else:
            if test_phase:
                timing_pred = self.test(expanded, gold_expanded, use_predict=True)
            else:
                timing_pred = self.train(expanded, gold_expanded)

        return timing_pred