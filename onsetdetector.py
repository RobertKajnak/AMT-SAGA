#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:25:19 2019

@author: hesiris
"""

import numpy as np
from RDCNN import res_net


class OnsetDetector(res_net):
    def __init__(self, params):
        super().__init__(input_shapes=[(params.N / 2, params.timing_input_shape, 1)],
                         kernel_sizes=params.kernel_sizes, pool_sizes=params.pool_sizes,
                         output_classes=1,
                         output_range = [0,params.timing_input_shape],

                         convolutional_layer_count=params.convolutional_layer_count,
                         pool_layer_frequency=params.pool_layer_frequency,
                         feature_expand_frequency=params.feature_expand_frequency,
                         residual_layer_frequencies=params.residual_layer_frequencies,
                         checkpoint_dir=params.checkpoint_dir, checkpoint_frequency=params.checkpoint_frequency,
                         checkpoint_prefix='checkpoint_onset_',
                         metrics=[],
                         verbose=False)

        self.params = params

    def detect(self, ac, start_gold=None):
        if ac.mag.shape != (self.params.N / 2, self.params.timing_input_shape):
            raise ValueError('Invalid Input shape. Expected: {} . Got: {}'.
                             format((int(self.params.N / 2), self.params.timing_input_shape),
                                    ac.mag.shape))
        # start = np.random.rand() * ac.F.shape[1]
        # duration = np.random.rand() * (ac.F.shape[1] / 42 - start)
        expanded = ac.mag[np.newaxis,:,:,np.newaxis]
        if start_gold is None:
            start = self.predict([expanded])
        else:
            start = self.train([expanded], np.expand_dims(start_gold,axis=0))

        return start
