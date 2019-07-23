#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 15:41:33 2019

@author: hesiris
"""

import numpy as np

PATH_MODEL_META = 'model_meta'
PATH_NOTES = 'notes'
PATH_CHECKPOINTS = 'checkpoints'

class Hyperparams:
    def __init__(self, path_data, sf_path, path_output='.', 
                 N=4096, sr=44100, H=None, window_size_note_time=None,
                 bins_per_tone = 4,
                 batch_size = 8, synth_worker_count =1,
                 parallel_train=False,
                 checkpoint_dir ='./data/checkpoints',
                 checkpoint_frequency = 200,
                 note_save_freq=0,
                 autoload = False,
                 use_precise_note_count = True):
        self.N = N
        self.sr = sr
        self.H = np.int(N / 4) if H is None else H
        self.window_size_note_time = 6 if window_size_note_time is None else window_size_note_time
        
        self.timing_frames = int(self.window_size_note_time * self.sr/self.H)
        self.timing_bands = 40 * bins_per_tone
        self.kernel_size_timing = [(2*bins_per_tone, 16)]
        self.pool_size_timing = [(2*bins_per_tone, 8)]
        
        self.pitch_frames = 8
        self.pitch_bands = 87
        self.pitch_low = 21
        self.pitch_high = 108
        self.kernel_size_pitch = [(6, 2)]
        self.pool_size_pitch = [(6, 2)]
        
        self.instrument_frames = self.pitch_frames
        self.instrument_bins_per_tone = bins_per_tone
        self.instrument_bands = self.instrument_bins_per_tone*self.pitch_bands
        self.instrument_classes = 112
        self.kernel_size_instrument = [(4*bins_per_tone, 2)]
        self.pool_size_instrument = [(6*bins_per_tone, 2)]
        
        self.batch_size = batch_size = batch_size


        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency
        self.convolutional_layer_count = 33
        self.pool_layer_frequency = 12
        self.feature_expand_frequency = 12  # pool_layer_frequency
        self.residual_layer_frequencies = [2]

        self.parallel_train = parallel_train
        self.synth_worker_count = synth_worker_count
        self.path_data = path_data
        self.sf_path = sf_path
        self.path_output = path_output
        self.note_save_freq = note_save_freq

        self.autoload = autoload
        
        self.use_precise_note_count = use_precise_note_count

def relevant_notes(sequence, offset, duration):
    notes_w, notes_target = sequence.get_notes(offset, offset + duration,
                                               include_drums = False)
    minus = notes_w.start_first

    notes_w = notes_w.clone()

    notes_target = notes_target.clone()
    notes_target.shift(-offset)
    notes_w.shift(-offset + minus)
    return notes_target, notes_w

def check_shape(spec,bands,frames):
    """ Checks if the spectrogram has the correct shape.
    params:
        spec: spectrogram or other 2D input
        bands: shape along 0th dimension
        frames: shape along 1st dimension
    returns:
        throws an exception if evaluation fails
    """
    if isinstance(spec,list) or isinstance(spec,tuple):
        if isinstance(spec[0],list) or isinstance(spec[0],tuple):
            spec_shape = spec[0][0].shape
        else:
            spec_shape = spec[0].shape
    else:
        spec_shape = spec.shape
    if spec_shape != (bands,frames):
        raise ValueError('Invalid Input shape. Expected: {} . Got: {}'.
                 format((bands,
                         frames),spec_shape))
        
def list_to_nd_array(spec,label):
    """ Converts a list to the NN good input. Returns the NN matrix or list of 
    matrices and the matrix witht the labels"""
    if isinstance(spec,list) or isinstance(spec,tuple):
        if isinstance(spec[0],list) or isinstance(spec[0],tuple):
            cb_xa = [np.zeros([0]+list(spec[0][0].shape)+[1]) for _ in range(len(spec[0]))]
            cb_y=np.zeros([0,1])
            
            for sp,label in zip(spec,label):
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
            for specs,label in zip(spec,label):
                expanded_x = specs[np.newaxis,:,:,np.newaxis]
                expanded_y = np.expand_dims([label],axis=0)
                cb_x=np.concatenate((cb_x,expanded_x))
                cb_y=np.concatenate((cb_y,expanded_y))
                
            expanded = cb_x
            gold_expanded = cb_y
    else:
        expanded = spec[np.newaxis,:,:,np.newaxis]
        gold_expanded = np.expand_dims(label,axis=0)
        
    return expanded, gold_expanded

def validate_note(note, params):
    """Returns None if no errors found, otherwise returns a string with the
    error message. 
    params:
        params must include the following fields: instrument_classes,
        pitch_low, pitch_high"""
    
    if note.program>=params.instrument_classes:
        return 'Program out of range: {}'.format(note.program)
    elif note.pitch<params.pitch_low or note.pitch>params.pitch_high:
        return 'Pitch out of range: {}'.note.pitch
    elif note.is_drum:
        return 'Note is percussive'

class note_sample:
    def __init__(self,filename, 
                 C_timing, 
                 sw_C_pitch, sw_C_inst, sw_F_inst_foc,sw_F_inst_foc_const,
                 pitch, instrument,
                 time_start, time_end):
        self.filename = filename
        self.C_timing = C_timing
        self.sw_C_pitch = sw_C_pitch
        self.sw_C_inst = sw_C_inst
        self.sw_F_inst_foc = sw_F_inst_foc
        self.sw_F_inst_foc_const = sw_F_inst_foc_const
        self.pitch = pitch
        self.instrument = instrument
        self.time_start = time_start
        self.time_end = time_end