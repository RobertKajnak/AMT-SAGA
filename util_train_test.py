#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 15:41:33 2019

@author: hesiris
"""

import numpy as np

class Hyperparams:
    def __init__(self, path, sf_path, 
                 N=4096, sr=44100, H=None, window_size_note_time=None,
                 batch_size = 8, synth_worker_count =1,
                 parallel_train=False):
        self.N = N
        self.sr = sr
        self.H = np.int(N / 4) if H is None else H
        self.window_size_note_time = 6 if window_size_note_time is None else window_size_note_time
        self.pitch_input_shape = 20
        self.timing_input_shape = 258
        self.batch_size = batch_size = batch_size

        self.kernel_sizes = [(32, 3), (32,3)]
        self.pool_sizes = [(5, 2), (5, 2)]
#        self.kernel_sizes_pitch = [(3, 32), (3, 8)]
#        self.pool_sizes_pitch = [(2, 5), (2, 5)]

        self.checkpoint_dir = './data/checkpoints'
        self.checkpoint_frequency = 200
        self.convolutional_layer_count = 33
        self.pool_layer_frequency = 12
        self.feature_expand_frequency = 12  # pool_layer_frequency
        self.residual_layer_frequencies = [2]

        self.parallel_train = parallel_train
        self.synth_worker_count = synth_worker_count
        self.path = path
        self.sf_path = sf_path
        


def relevant_notes(sequence, offset, duration):
    notes_w, notes_target = sequence.get_notes(offset, offset + duration)
    minus = notes_w.start_first

    notes_w = notes_w.clone()

    notes_target = notes_target.clone()
    notes_target.shift(-offset)
    notes_w.shift(-offset + minus)
    return notes_target, notes_w

class note_sample:
    def __init__(self,filename, audio, pitch, instrument,
                 onset_s, duration_s):
        self.filename = filename
        self.audio = audio
        self.pitch = pitch
        self.instrument = instrument
        self.onset_s = onset_s
        self.duration_s = duration_s