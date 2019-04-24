#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:30:51 2019

@author: hesiris
"""

import numpy as np

class pitch_classifier:
    def __init__(self,params):
        self.params = params
        
    def classify(self,arg,pitch_gold):
        if arg.shape != (self.params.N/2,self.params.pitch_input_shape):
            raise ValueError('Invalid Input shape. Expected: {} . Got: {}'.format((int(self.params.N/2),self.params.pitch_input_shape),arg.shape))
        return int(np.random.rand()*107)