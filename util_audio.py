# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 17:05:15 2019

@author: Hesiris
"""

from magenta.music import midi_io
from magenta.protobuf import music_pb2
import rtmidi
import rtmidi.midiconstants as consts
import pyaudio
from sf2utils.sf2parse import Sf2File
import librosa

import soundfile
import numpy as np
import matplotlib.pyplot as plt

from threading import Timer

import wave

#https://www.midi.org/specifications-old/item/table-3-control-change-messages-data-bytes-2
#http://people.csail.mit.edu/hubert/pyaudio/
#https://github.com/SpotlightKid/python-rtmidi/blob/master/examples/advanced/midiwrapper.py



class entity:
    def __init__(self,waveform,F_H,F_window_size=None,sample_rate=44100):
        self.wf = waveform
        self._F = None
        self._mag = None
        self._ph = None
        self._D = None
        self.ref_mag = np.max(self.mag)
        
        self.sr = sample_rate
        self.H = F_H
        self.F_ws = F_window_size

    @property
    def F(self):
        if self._F is None:
            self._F = librosa.stft(self.wf)
        return self._F
    
    @property
    def mag(self):
        if self._mag is None:
            self._mag,self._ph = librosa.core.magphase(self.F)
        return self._mag
    @property
    def ph(self):
        if self._ph is None:
            self._mag,self._ph = librosa.core.magphase(self.F)
        return self._ph

    @property
    def D(self):
        if self._D is None:
            self._D = librosa.amplitude_to_db(self.mag,ref=self.ref_mag)
        return self._D
    
    def subtract(self,subtrahend):
        """ self is the minuend and the subtrahend is provided. 
            It can be either of the same class or a magnitude
        """
        pass
    
    def plot_spec(self,width=12,hegith = 5):
        
        plt.figure(figsize=(10, 5))

        plt.subplot(1,1,1)
        librosa.display.specshow(self.D, y_axis='log',x_axis='time',sr=self.sr)
        plt.ylabel('Log DB')
        plt.colorbar(format='%+2.0f dB')

    def save(self,filename):
        audio_to_flac(waveform = self.wf,
                      filename=filename,sr=self.sr)
        
#    def close(self):
#        pass
#        
#    def __enter__(self):
#        return self
#
#    def __exit__(self, exc_type, exc_value, traceback):
#        self.close()

class note_sequence:
    #'D:/Soundfonts/HD.sf2'
    #'/home/hesiris/Documents/Thesis/GM_soundfonts.sf2'
    def __init__(self, filename= None):
        
        if filename is None:    
            self.sequence = music_pb2.NoteSequence()
        else:
            self.sequence = midi_io.midi_file_to_note_sequence(filename)
            
            
    def add_note(self, instrument, program, pitch, start, end, velocity=100,  is_drum=False):
        '''Adds a note to the current note sequence'''
        note = self.sequence.notes.add()
        note.instrument = instrument
        note.program = program
        note.start_time = start
        note.end_time = end
        note.pitch = pitch
        note.velocity = velocity
        note.is_drum = is_drum
        
    def get_notes(self,start, end):
        ''' returns all the notes that have sound the specified period
        '''
        subsequence = music_pb2.NoteSequence()
        notes_to_add = []
        for note in self.sequence.notes:
            if (note.start_time>start and note.start_time<end) or \
                    (note.end_time>start and note.end_time<end):
                notes_to_add.append(note)
                
        subsequence.notes.extend(notes_to_add)
        return subsequence
    
    def render(self, soundfont_filename=None,max_duration = None,sample_rate=44100):
        mid = midi_io.note_sequence_to_pretty_midi(self.sequence)
        wf = mid.fluidsynth(fs=sample_rate, sf2_path=soundfont_filename)
        
        if max_duration is None:
            return wf
        else:
            return wf[:max_duration*sample_rate]
        
        
    def save(self,file_name):
        '''Saves the MIDi file to the specified filename'''
        midi_io.note_sequence_to_midi_file(self.sequence, file_name)
        
        
def plot_specs(spec_list,sr=44100,width =12,height_per_plot=5):

    N = len(spec_list)
    if N>20:
        raise ValueError('Too many spectrograms {}>20. ' +\
                         'To plot a single graph use spec_list=[spec]'.format(N))
    plt.figure(figsize=(width, height_per_plot*N))

    for i in range(N):
        plt.subplot(N,1,i+1)
        librosa.display.specshow(spec_list[i], y_axis='log',x_axis='time',sr=sr)
        plt.ylabel('Log DB')
        plt.colorbar(format='%+2.0f dB')
        
def audio_from_file(audio_filename):
    return librosa.load(audio_filename,sr=None)    
   
def audio_to_flac(waveform,filename,sr=44100): 
    soundfile.write(filename, waveform, sr, format='flac', subtype='PCM_24')

def midi_from_file(file_name):
    '''Returns a note sequence structure loaded from the specified file name'''
    return midi_io.midi_file_to_note_sequence(file_name)