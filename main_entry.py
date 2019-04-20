# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 17:04:08 2019

@author: Hesiris
"""

import numpy as np
import os

from onset_detector import onset_detector as OnsetDetector
from pitch_classifier import pitch_classifier as PitchClassifier
from instrument_classifier import instrument_classifier as InsrumentClassifier
import util_dataset
from util_audio import note_sequence
from util_audio import audio_complete


class hyperparams:
    def __init__(self,N=4096,sr =44100,H=None,window_size = None,window_th = None):
        self.N=N
        self.sr =sr
        self.H = np.int(N/4) if H is None else H
        self.window_size = 3  if window_size is None else window_size#in FS
        self.window_th = 0.2 if window_th is None else window_th
        self.pitch_input_shape = 20
        
def relevant_notes(sequence, offset, duration,N):
    
    notes_w,notes_target = sequence.get_notes(offset,offset+duration)
    minus = notes_w.start_first
    
    notes_w = notes_w.clone()
    notes_w.shift(-minus)
    audio = audio_complete(notes_w.render(sf_path),N)
    print(offset,minus)
    audio.wf = audio.wf[int(offset-minus)*audio.sr:]
    
    notes_target = notes_target.clone()
    notes_target.shift(-offset)
    notes_w.shift(-offset+minus)
    return audio, notes_target, notes_w
    
def pre_train(path,sf_path,params):
    """ Prepare data, training and test"""
    dm = util_dataset.DataManager(path,sets=['training','test'],types=['midi'])
    onset_detector = OnsetDetector(params)
    pitch_classifier = PitchClassifier(params)
    instrument_classifier = InsrumentClassifier(params)
    DEBUG = 1
    frametime = params.H / params.sr

    dm.set_set('training')        
    for fn in dm:
       mid = note_sequence(fn[0])
       
       sheet = note_sequence()
       
       offset = 0
       audio_w, notes_target, notes_w = relevant_notes(mid,offset,params.window_size,params.H)
       while offset<mid.duration:
           
           if DEBUG:
               fn_base = os.path.join(path,'debug',os.path.split(fn[0])[-1][:-4]+str(DEBUG))
               print(fn_base)
               audio_w.save(fn_base+'.flac')
               notes_target.save(fn_base+'_target.mid')
               notes_w.save(fn_base+'_inclusive.mid')
               DEBUG += 1
           
           note_gold = notes_target.pop(lowest=True,threshold = frametime)
           print(note_gold) 
           #training
           if note_gold is not None:
               onset_gold = note_gold.start_time
               duration_gold = note_gold.end_time-note_gold.start_time
               pitch_gold =     note_gold.pitch
               instrument_gold = note_gold.program
           else:
               onset_gold = params.window_th+offset
           onset_s,duration_s = onset_detector.detect(audio_w,onset_gold,duration_gold)
           
           onset_s = onset_gold
           #use correct value to move window
           if onset_s>=params.window_th+offset:
               offset+=onset_s
               audio_w, notes_target, notes_w = relevant_notes(mid,offset,params.window_size,params.H)
               continue
           
           pitch_s = pitch_classifier.classify(audio_w,pitch_gold)
           
           audio_sw = audio_w.resize(onset_s,duration_s,pitch_s,params.pitch_input_shape)
           instrument_sw = instrument_classifier.classify(audio_sw,instrument_gold)
           
           #subtract correct note for training:
#           note_guessed = note_sequence()
#           note_guessed.add_note(instrument_gold, instrument_gold, pitch_gold, 
#                                 onset_gold, onset_gold+duration_gold, velocity=100, 
#                                 is_drum=False)
#           audio_w.subtract(note_guessed.render(),params.N)
           
           sheet.add_note(instrument_sw,instrument_sw,pitch_s,onset_s+offset,onset_s+offset+duration_s)
           
       sheet.save(fn[:-4] + '_t.mid') 
    


if __name__ == '__main__':
    # Hyperparams
    path = './data/'
    sf_path = '/home/hesiris/Documents/Thesis/GM_soundfonts.sf2'
    p = hyperparams()
    p.path = path
    
    pre_train(path,sf_path,p)
    
    training_session = True
    

    # Create Model    
    rnet = RDCNN.res_net()