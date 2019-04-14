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
        
    def classify(self,arg):
        return np.random.rand()*107