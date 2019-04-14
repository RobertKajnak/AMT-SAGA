#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:25:19 2019

@author: hesiris
"""

import numpy as np

class onset_detector:
    def __init__(self,params):
        self.params = params
    def detect(self, ac):
        start = np.random.rand()*ac.F.shape[1]
        duration = np.random.rand()*(ac.F.shape[1]-start)
        
        return start,duration