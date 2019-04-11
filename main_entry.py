# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 17:04:08 2019

@author: Hesiris
"""

#from magenta.models.onsets_frames_transcription import data
#from magenta.music import midi_io
#from magenta.protobuf import music_pb2
#
#
#import matplotlib.pyplot as plt
#
#import librosa
#import librosa.display
#import soundfile
#import scipy
#

#from util import midi_from_file
import numpy as np

import RDCNN
import util_dataset
from util_audio import audio_from_file
from util_audio import note_sequence


if __name__ == '__main__':
    #%% Hyperparams
    N=4096
    fs =44100
    H = np.int(N/4)
    
    window_size = 3*fs/H #in FS
    window_th = 0.2*fs/H
    training_session = True
    
    #%% Prepare data, training and test
    dm = util_dataset.DataManager()
    dm.switch_type(0)
    
    synth = sf_synthetiser()
    
    onset_detector = None
    pitch_detector = None
    instrument_classifier = None
    
    audio_name, midi_name = dm.next_pair()
    while audio_name is not None or midi_name is not None:
       mid = note_sequence(midi_name)
       aud = audio_from_file(audio_name)
       
       sheet = note_sequence()
       
       offset = 0
       no_more_detectable_notes = False
       while not no_more_detectable_notes:
           notes_w = mid.get_notes(offset,offset+window_size)
           audio_w = sf.render_notes_fourier(notes_w,offset,window_size)
           
           onset_s,duration_s = Onset_detector.detect(audio_w)
           if onset_s>window_th:
               offset+=onset_s
               continue
           
           pitch_s = pitch_detector.detect(audio_w)
           
           audio_sw = audio_util.resize(audio_w,onset_s,duration_s,pitch_s)
           instrument_sw = instrument_classifier.classify(audio_sw)
           
           sheet.add(None,instrument_sw,pitch_s,onset_s+offset,onset_s+offset+duration_s)
           
           
        
       sheet.save(midi_name[:-4] + '_t.mid') 
       audio_name, midi_name = dm.next_pair()
    
    #%% Create Model    
    rnet = RDCNN.res_net()