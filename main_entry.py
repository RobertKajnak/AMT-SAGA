# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 17:04:08 2019

@author: Hesiris
"""

#from magenta.models.onsets_frames_transcription import data
#from magenta.music import midi_io
#from magenta.protobuf import music_pb2
#
#import numpy as np
#import matplotlib.pyplot as plt
#
#import librosa
#import librosa.display
#import soundfile
#import scipy
#
#from util import note_sequence as nsequence
#from util import midi_from_file


import RDCNN
import util_dataset
from util_audio import audio_from_file
from util_adio import note_sequence

if __name__ == '__main__':
    #%% Hyperparams
    window_size = 3
    
    #%% Prepare data, training and test
    dm = util_dataset.DataManager()
    dm.switch_type(0)
    
    
    audio_name, midi_name = dm.next_pair()
    while audio_name is not None or midi_name is not None:
       mid = note_sequence(midi_name)
       aud = audio_from_file(audio_name)
       
       offset = 0
       mid.get_notes(offset,offset+window_size)
       
        
       audio_name, midi_name = dm.next_pair()
    
    #%% Create Model    
    rnet = RDCNN.res_net()