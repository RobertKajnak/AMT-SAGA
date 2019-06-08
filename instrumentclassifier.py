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

                         convolutional_layer_count=params.convolutional_layer_count,
                         pool_layer_frequency=params.pool_layer_frequency,
                         feature_expand_frequency=params.feature_expand_frequency,
                         residual_layer_frequencies=params.residual_layer_frequencies,
                         checkpoint_dir=params.checkpoint_dir, checkpoint_frequency=params.checkpoint_frequency,
                         checkpoint_prefix='checkpoint_onset_',
                         metrics = ['acc'],
                         verbose=False)

        self.params = params

    def classify(self, ac, instrument_gold=None):
        if ac.mag.shape != (self.params.N / 2, self.params.pitch_input_shape):
            raise ValueError('Invalid Input shape. Expected: {} . Got: {}'.format(
                (int(self.params.N / 2), self.params.pitch_input_shape), ac.mag.shape))

        expanded = ac.mag[np.newaxis,:,:,np.newaxis]
        if instrument_gold is None:
            instrument_pred = self.predict([expanded])
        else:
            instrument_pred = self.train([expanded], [instrument_gold])
        # return int(np.random.rand() * 127)

        return instrument_pred