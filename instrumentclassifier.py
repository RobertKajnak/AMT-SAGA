#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:46:18 2019

@author: hesiris
"""

from RDCNN import res_net
from util_train_test import check_shape, list_to_nd_array

class InstrumentClassifier(res_net):
    INSTRUMENT = 'instrument'
    INSTRUMENT_FOCUSED = 'instrument_focused'
    INSTRUMENT_FOCUSED_CONST = 'instrument_focused_const'
    INSTRUMENT_DUAL = 'instrument_dual'
    def __init__(self, params, variant, prefix = None):
        if variant == InstrumentClassifier.INSTRUMENT:
            input_shape = [(params.instrument_bands, params.instrument_frames, 1)]
            kernel_size = params.kernel_size_instrument
            pool_size = params.pool_size_instrument
        elif variant == InstrumentClassifier.INSTRUMENT_FOCUSED or \
        variant == InstrumentClassifier.INSTRUMENT_FOCUSED_CONST:
            input_shape = [(params.instrument_bands, params.instrument_frames, 1)]
            kernel_size = params.kernel_size_instrument
            pool_size = params.pool_size_instrument
        elif variant == InstrumentClassifier.INSTRUMENT_DUAL:
            input_shape = [(params.instrument_bands, params.instrument_frames, 1),
                           (params.instrument_bands, params.instrument_frames, 1)]
            #The *2 duplicates it [(2,4)]->[(2,4),(2,4)]*
            kernel_size = params.kernel_size_instrument*2
            pool_size = params.pool_size_instrument*2
        else:
            raise ValueError('Invalid Variant Selected')
        
        if prefix is None:
            checkpoint_prefix = 'checkpoint_' + variant
            metrics_prefix = 'metrics_' + variant
        else:
            checkpoint_prefix = 'checkpoint_' + prefix
            metrics_prefix = 'metrics_' + prefix
            
        super().__init__(input_shapes=input_shape,
                         kernel_sizes=kernel_size, pool_sizes=pool_size,
                         output_classes=params.instrument_classes,
                         batch_size = params.batch_size,

                         convolutional_layer_count=params.convolutional_layer_count,
                         pool_layer_frequency=params.pool_layer_frequency,
                         feature_expand_frequency=params.feature_expand_frequency,
                         residual_layer_frequencies=params.residual_layer_frequencies,
                         checkpoint_dir=params.checkpoint_dir, checkpoint_frequency=params.checkpoint_frequency,
                         checkpoint_prefix= checkpoint_prefix,
                         metrics_prefix = metrics_prefix,
                         metrics = [],
                         
                         logging_parent = 'AMT-SAGA')

        self.params = params

    def classify(self, spec, instrument_gold=None, test_phase = False):
        """ Trains, test and classifies the provided sample.
        params:
            spec: sample
            instrument_gold: if None, prediction is performed, without checkking
                the correctness of the result
            test_phase: if set to true, testing is done, otherwise training
        """

        check_shape(spec, self.params.instrument_bands,
                    self.params.instrument_frames)

        expanded, gold_expanded = list_to_nd_array(spec,instrument_gold)
            
        if instrument_gold is None:
            instrument_pred = self.predict(expanded)
        else:
            if test_phase:
                instrument_pred = self.test(expanded, gold_expanded, use_predict=True)
            else:
                instrument_pred = self.train(expanded, gold_expanded)
            
        return instrument_pred