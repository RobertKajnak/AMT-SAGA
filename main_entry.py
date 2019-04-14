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

#import RDCNN
import onset_detector as OnsetDetector
import pitch_classifier as PitchClassifier
import instrument_classifier as InsrumentClassifier
import util_dataset
from util_audio import audio_from_file
from util_audio import note_sequence
from util_audio import audio_complete

class DummyNetwork:
    def __init__(self,params):
        self.params = params
    def detect(self,arg):
        return 0
    def classify(self,arg):
        return 0
    
class hyperparams:
    def __init__(self,N=4096,sr =44100,H=None,window_size = None,window_th = None):
        self.N=N
        self.sr =sr
        self.H = np.int(N/4) if H is None else H
        self.window_size = 3*sr/H  if window_size is None else window_size#in FS
        self.window_th = 0.2*sr/H if window_th is None else window_th
        self.pitch_input_shape = 20
        
def pre_train(path,sf_path,params):
    """ Prepare data, training and test"""
    dm = util_dataset.DataManager(path,sets=['training','test'],types=['midi'])
    onset_detector = OnsetDetector(params)
    pitch_classifier = PitchClassifier(params)
    instrument_classifier = InsrumentClassifier(params)
    DEBUG = True

    dm.set_type('training')        
    for fn in dm:
       mid = note_sequence(fn)
       
       sheet = note_sequence()
       
       offset = 0
       no_more_detectable_notes = False
       while not no_more_detectable_notes:
           notes_w = mid.get_notes(offset,offset+params.window_size)
           audio_w = audio_complete(notes_w.render_notes(sf_path),params.H)
           
           onset_s,duration_s = onset_detector.detect(audio_w)
           
           #use correct value to move window
           if onset_s>params.window_th:
               offset+=onset_s
               continue
           
           pitch_s = pitch_classifier.classify(audio_w)
           
           audio_sw = audio_util.resize(audio_w,onset_s,duration_s,pitch_s,params.pitch_input_shape)
           instrument_sw = instrument_classifier.classify(audio_sw)
           
           sheet.add(instrument_sw,instrument_sw,pitch_s,onset_s+offset,onset_s+offset+duration_s)
           
       sheet.save(fn[:-4] + '_t.mid') 
    


if __name__ == '__main__':
    # Hyperparams
    path = './data/'
    sf_path = '/home/hesiris/Documents/Thesis/GM_soundfonts.sf2'
    p = hyperparams()
    
    pre_train(path,sf_path,p)
    
    training_session = True
    

    # Create Model    
    rnet = RDCNN.res_net()