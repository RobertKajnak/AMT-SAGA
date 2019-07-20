#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:46:18 2019

@author: hesiris
"""

import numpy as np
from RDCNN import res_net


class InstrumentClassifier(res_net):
    INSTRUMENT = 'instrument'
    INSTRUMENT_FOCUSED = 'instrument_focused'
    INSTRUMENT_DUAL = 'instrument_dual'
    def __init__(self, params, variant):
        if variant == InstrumentClassifier.INSTRUMENT:
            input_shape = [(params.instrument_bands, params.instrument_frames, 1)]
            kernel_size = params.kernel_size_instrument
            pool_size = params.pool_size_instrument
        elif variant == InstrumentClassifier.INSTRUMENT_FOCUSED:
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
        
        checkpoint_prefix = 'checkpoint_' + variant
        metrics_prefix = 'metrics_' + variant
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
        if isinstance(spec,list) or isinstance(spec,tuple):
            if isinstance(spec[0],list) or isinstance(spec[0],tuple):
                spec_shape = spec[0][0].shape
            else:
                spec_shape = spec[0].shape
        else:
            spec_shape = spec.shape
        if spec_shape != (self.params.instrument_bands,self.params.instrument_frames):
            raise ValueError('Invalid Input shape. Expected: {} . Got: {}'.
                             format((self.params.instrument_bands,
                                     self.params.instrument_frames),spec_shape))

        if isinstance(spec,list) or isinstance(spec,tuple):
            if isinstance(spec[0],list) or isinstance(spec[0],tuple):
                cb_xa = [np.zeros([0]+list(spec[0][0].shape)+[1]) for _ in range(len(spec[0]))]
                cb_y=np.zeros([0,1])
                
                for sp,label in zip(spec,instrument_gold):
                    expanded_y = np.expand_dims([label],axis=0)
                    cb_y=np.concatenate((cb_y,expanded_y))
                    for ind,chan in enumerate(sp):
                        expanded_x = chan[np.newaxis,:,:,np.newaxis]
                        cb_xa[ind]=np.concatenate((cb_xa[ind],expanded_x))
                        
                expanded = cb_xa
                gold_expanded = cb_y
            else:
                cb_x=np.zeros([0]+list(spec[0].shape)+[1])
                cb_y=np.zeros([0,1])
                for specs,label in zip(spec,instrument_gold):
                    expanded_x = specs[np.newaxis,:,:,np.newaxis]
                    expanded_y = np.expand_dims([label],axis=0)
                    cb_x=np.concatenate((cb_x,expanded_x))
                    cb_y=np.concatenate((cb_y,expanded_y))
                    
                expanded = cb_x
                gold_expanded = cb_y
        else:
            expanded = spec[np.newaxis,:,:,np.newaxis]
            gold_expanded = np.expand_dims(instrument_gold,axis=0)
            
        if instrument_gold is None:
            instrument_pred = self.predict(expanded)
        else:
            if test_phase:
                instrument_pred = self.test(expanded, gold_expanded)
            else:
                instrument_pred = self.train(expanded, gold_expanded)
            
        return instrument_pred