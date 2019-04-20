#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:46:18 2019

@author: hesiris
"""

import numpy as np

class instrument_classifier:
    def __init__(self,params):
        self.params = params
        
    def classify(self,arg,instrument_gold=None):
        return int(np.random.rand()*127)