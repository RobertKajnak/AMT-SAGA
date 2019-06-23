#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:46:18 2019

@author: hesiris
"""

import numpy as np
from RDCNN import res_net


class InstrumentClassifier(res_net):
    def __init__(self, params):
        super().__init__(input_shapes=[(params.N / 2, params.pitch_input_shape, 1)],
                         kernel_sizes=params.kernel_sizes, pool_sizes=params.pool_sizes,
                         output_classes=127,
                         batch_size = params.batch_size,

                         convolutional_layer_count=params.convolutional_layer_count,
                         pool_layer_frequency=params.pool_layer_frequency,
                         feature_expand_frequency=params.feature_expand_frequency,
                         residual_layer_frequencies=params.residual_layer_frequencies,
                         checkpoint_dir=params.checkpoint_dir, checkpoint_frequency=params.checkpoint_frequency,
                         checkpoint_prefix='checkpoint_onset_',
                         metrics = [],
                         
                         logging_parent = 'AMT-SAGA')

        self.params = params

    def classify(self, ac, instrument_gold=None):
        if isinstance(ac,list):
            ac_shape = ac[0].shape
        else:
            ac_shape = ac.shape
        if ac_shape != (self.params.N/2,self.params.pitch_input_shape):
            raise ValueError('Invalid Input shape. Expected: {} . Got: {}'.
                             format((int(self.params.N/2),
                                     self.params.pitch_input_shape),ac_shape))

        if isinstance(ac,list):
            cb_x=np.zeros([0]+list(ac[0].shape)+[1])
            cb_y=np.zeros([0,1])
            for acs,label in zip(ac,instrument_gold):
                expanded_x = acs.mag[np.newaxis,:,:,np.newaxis]
                expanded_y = np.expand_dims([label],axis=0)
                cb_x=np.concatenate((cb_x,expanded_x))
                cb_y=np.concatenate((cb_y,expanded_y))
                
            expanded = cb_x
            gold_expanded = cb_y
        else:
            expanded = ac.mag[np.newaxis,:,:,np.newaxis]
            gold_expanded = np.expand_dims(instrument_gold,axis=0)
            
        if instrument_gold is None:
            instrument_pred = self.predict(expanded)
        else:
            instrument_pred = self.train(expanded, gold_expanded)
            
        return instrument_pred