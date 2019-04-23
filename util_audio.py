# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 17:05:15 2019

@author: Hesiris
"""

from magenta.music import midi_io
from magenta.protobuf import music_pb2
import librosa

import soundfile
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

import copy

#https://www.midi.org/specifications-old/item/table-3-control-change-messages-data-bytes-2
#http://people.csail.mit.edu/hubert/pyaudio/
#https://github.com/SpotlightKid/python-rtmidi/blob/master/examples/advanced/midiwrapper.py

class audio_complete:
    def __init__(self,waveform,n_fft,hop_length=None,center=True,sample_rate=44100):
        self.wf = waveform
        self._F = None
        self._mag = None
        self._ref_mag = None
        self._ph = None
        self._D = None
        
        self.sr = sample_rate
        self.N = n_fft
        self.center = center
        self.hl = hop_length if hop_length is not None else int(np.floor(n_fft/4))
        
        
    def clone(self):
        """Copies the parameters and a deepcopy of the waveform"""
        return audio_complete(copy.deepcopy(self.wf),self.N,self.hs,self.sr)
    
    @property
    def F(self):
        if self._F is None:
            self._F = librosa.stft(self.wf,n_fft = self.N,
                                   hop_length=self.hl,center=self.center)
        return self._F
    @F.setter
    def F(self,value):
        self._D = None
        self._ref_mag = None
        self._mag = None
        self._ph = None
        self._F = value
        self.wf = librosa.istft(self._F,
                                   hop_length=self.hl,center=self.center)
    
    @property
    def mag(self):
        if self._mag is None:
            self._mag,self._ph = librosa.core.magphase(self.F)
        return self._mag
    @mag.setter
    def mag(self,val):
        self._D = None
        self._ref_mag = None
        self._mag = val
        self._F = self._mag * self._ph
        self.wf = librosa.istft(self._F)
        
    @property
    def ph(self):
        if self._ph is None:
            self._mag,self._ph = librosa.core.magphase(self.F)
        return self._ph
    @ph.setter
    def ph(self,val):
        self._ph = val
        self._F = self._mag*self._ph
        self.wf = librosa.istft(self._F)

    @property
    def ref_mag(self):
        if self._ref_mag is None:
            self._ref_mag = np.max(self.mag)
        return self._ref_mag

    @property
    def D(self):
        if self._D is None:
            self._D = librosa.amplitude_to_db(self.mag,ref=self.ref_mag)
        return self._D
    @D.setter
    def D(self, val):
        self._D = val
        #self._ref_mag = None
        self._mag = librosa.db_to_amplitude(self._D,ref = self.ref_mag)
        self._F = self._mag*self._ph
        self.wf = librosa.istft(self._F)
    
    def subtract(self,subtrahend, offset=0,attack_compensation=0,
                     normalize = True, relu = True, overkill_factor = 1):
        """ self is the minuend and the subtrahend is provided
            params:
                subtrahend: It can be either of the same class or a magnitude
                offset: offset expressed in seconds
                attack_compensation: when generating a midi file, the attack may be handled differently
                    when starting from 0 time or not
                normalize: True=> subtrahend *= np.max(minuend)/np.max(subtrahend)
                relu: values below 0 are set to 0 on the result
                overkill_factor: the sutrahend is multiplied by this value to increase it's effect
                    it is adviseable to also set 'relu=True'
                
                
        """
        
        if isinstance(subtrahend,type(self)):
            mag_sub = copy.deepcopy(subtrahend.mag)
            if normalize:
                mag_sub *= self.ref_mag / subtrahend.ref_mag
        else:
            mag_sub = copy.deepcopy(subtrahend)
            ref_max_sub = np.max(mag_sub)
            if normalize:
                mag_sub *= self.ref_mag / ref_max_sub
                
        mag_sub *= overkill_factor
        
        total_s = self.wf.shape[0]/self.sr
        offset = np.int(np.floor( self.mag.shape[1]/total_s * offset) )- attack_compensation
        self.mag -= np.concatenate(
                         (np.zeros((self.mag.shape[0],offset)) ,
                         mag_sub,
                         np.zeros((self.mag.shape[0],self.mag.shape[1]-offset-mag_sub.shape[1]))),
                         axis = 1)
        if relu:
            self.mag = np.maximum(self.mag,0,self.mag)
        
    def _seconds_to_frames(self,time):
        """Converts a from seconds to frames in fourier"""
        return int(np.floor(time * self.F.shape[1]*self.sr / self.wf.shape[0]))
    
    def _frames_to_seconds(self,frames):
        return frames/self.F.shape[1]/ self.sr * self.wf.shape[0]

    def section(self, start,end):
        """Creates a copy of a section and returns it. The D, F et.c will remain calculated
        Start and end specified in seconds. They will be rounded down to the nearest
        Fourier frame"""
        tfs = self._seconds_to_frames(start)
        tfe = self._seconds_to_frames(end)
        
        nac = audio_complete(copy.deepcopy(self.wf[int(np.floor(self._frames_to_seconds(tfs) * self.sr)):\
                                    int(np.floor(self._frames_to_seconds(tfe) * self.sr))]),
                             self.N,hop_length=self.hl,center=self.center,sample_rate=self.sr)
        
        def cc(f):
            if f is not None:
                return copy.deepcopy(f[:,tfs:tfe])
    
        nac._F = cc(self._F)
        nac._ref_mag = self._ref_mag
        nac._mag = cc(self._mag)
        nac._ph =cc(self._ph)
        nac._D = cc(self._D,)
        
        return nac

    def resize(self,start,duration,pitch,NN_input_shape):
        """ A new array is returned that is the resized the audio snippet to 
            a size good for NN input. Does not modify the instance content
        """
        t = self._seconds_to_frames(duration)
        s = self._seconds_to_frames(start)
        F = self.F[:,s:t] #If it is longer, it will take a shorter section starting from s
        t = self.F.shape[1]
        
        if t==0:
            return np.zeros((F.shape[0],NN_input_shape))
        elif t==NN_input_shape:
            resd = F
        elif t<3:
            resd = np.tile(F[:,-1:],NN_input_shape)
        elif t<NN_input_shape:
            lim = np.min((4,int(np.round(t/3))))
            l_t = int(np.floor((NN_input_shape - 2 * lim)/(t-2*lim)))
            tiled = np.tile(F[:,lim:-lim],l_t)
            resd = np.concatenate((F[:,:lim],
                                   tiled,
                                   F[:,-(NN_input_shape-tiled.shape[1]-lim):]),
                                   axis=1)
        else:
            lim = np.int(np.min([4,NN_input_shape/3]))
            cent = int(np.floor(F.shape[1]/2))
            cl = cent - int(np.floor(NN_input_shape/2)) + lim
            ch = cent + int(np.ceil(NN_input_shape/2)) - lim 
            resd = np.concatenate((F[:,:lim],F[:,cl:ch],F[:,-lim:]),axis=1)
        return resd
        
    def plot_spec(self,width=12,height = 5):
        
        plt.figure(figsize=(width, height))

        plt.subplot(1,1,1)
        librosa.display.specshow(self.D, y_axis='log',x_axis='time',sr=self.sr,hop_length = self.hl)
        plt.ylabel('Log DB')
        plt.colorbar(format='%+2.0f dB')

    def save(self,filename):
        audio_to_flac(waveform = self.wf,
                      filename=filename,sr=self.sr)
        

class note_sequence:
    #'D:/Soundfonts/HD.sf2'
    #'/home/hesiris/Documents/Thesis/GM_soundfonts.sf2'
    def __init__(self, filename= None):
        """A wrapper class around the magenta midi wrapper that goes around pretty_midi
        """
        if filename is None:    
            self.sequence = music_pb2.NoteSequence()
            self.duration = 0
            self.start_first = 0
        else:
            self.sequence = midi_io.midi_file_to_note_sequence(filename)
            self.duration = self.sequence.notes[-1].end_time
            self.start_first = self.sequence.notes[0].start_time
        self.prev_octave = 4
            
    _notes = {'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,'F#':6,
             'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'Hb':10,
                 'B':11,'H':11}  
    
    def append(self, note, copy=False):
        """If copy is set to true, the note is cloned first. Otherwise reference is passed
        """
        if copy:
            #Double-checked manually, these are all primitives
            note2 = self.sequence.notes.add()
            note2.instrument = note.instrument
            note2.program = note.program
            note2.start_time = note.start_time
            note2.end_time = note.end_time
            note2.pitch = note.pitch
            note2.velocity = note.velocity
            note2.is_drum = note.is_drum
            note = note2
        else:
            self.sequence.notes.extend([note])
        self.duration = np.max((self.duration,note.end_time))
        self.start_first = np.min((self.start_first,note.start_time))
    
    def add_note(self, instrument, program, pitch, start, end, velocity=100, is_drum=False):
        """Adds a note to the current note sequence
        args:
            instrument: instrument track (int)
            program: instrument (0-127) e.g. 0==Grand Piano. 
                Check e.g. http://www.ccarh.org/courses/253/handout/gminstruments/
                for more details
            pitch: pitch either expressed in midi note or string form. 
                int: Middle C=60.
                string: 'C', 'c','c#','Db' etc. If a digit is specified after the character, 
                    it specifies octave octave settings, e.g. C#2 or D2 will 
                    be in the second octave. If no number is specified, the last 
                    used octave is taken. Initialized to 4.
                    Visit https://newt.phys.unsw.edu.au/jw/notes.html for more details
            start: starting time in seconds
            end: end time in seconds. Decay is performed after this
            velocity: strength of pressing the note
            is_drum: specifies wether it is a drum track
            
        """
        if isinstance(pitch,str):
            try:
                octave = int(pitch[-1])
                pitch_lit = pitch[:-1]                    
            except:
                octave = self.prev_octave
                pitch_lit = pitch
            try:
                modifier = pitch_lit[1]
                pitch_lit = pitch_lit[0].upper()
            except:
                modifier = ''
                pitch_lit = pitch_lit.upper()                        
            pitch = self._notes[pitch_lit+modifier] + (octave+1) * 12
        
        
        note = self.sequence.notes.add()
        note.instrument = instrument
        note.program = program
        note.start_time = start
        note.end_time = end
        note.pitch = pitch
        note.velocity = velocity
        note.is_drum = is_drum
        self.duration = np.max((self.duration,end))
        self.start_first = np.min((self.start_first,start))

    def get_notes(self,start, end):
        ''' Returns:
                all notes that have sound inside the specified period
                all notes that have both start and end points within spec. period
        '''
        subsequence_inclusive = music_pb2.NoteSequence()
        subsequence_exclusive = music_pb2.NoteSequence()
        notes_to_add_inclusive = []
        notes_to_add_exclusive = []
        for note in self.sequence.notes:
            if (note.start_time>start and note.end_time<end):
                notes_to_add_exclusive.append(note)
                notes_to_add_inclusive.append(note)
            elif (note.end_time>start and note.start_time<end):
                notes_to_add_exclusive.append(note)
                
                
        ns_exc = note_sequence()
        if len (notes_to_add_exclusive) != 0:
            subsequence_exclusive.notes.extend(notes_to_add_exclusive)
            ns_exc.sequence = subsequence_exclusive
            ns_exc.duration = subsequence_exclusive.notes[-1]
        
        ns_inc = note_sequence()
        if len(notes_to_add_inclusive) !=0:
            subsequence_inclusive.notes.extend(notes_to_add_inclusive)
            
            ns_inc.sequence = subsequence_inclusive
            ns_inc.duration = subsequence_inclusive.notes[-1]
            
        return ns_inc,ns_exc 
    
    def clone(self):
        s_clone = note_sequence()
        s_clone.prev_octave = self.prev_octave
        
        for note in self.sequence.notes:
            s_clone.append(note,copy=True)
        return s_clone
    
    def shift(self, time):
        for note in self.sequence.notes:
            note.start_time+=time
            note.end_time+=time
        
        self.duration += time
        self.start_first += time
    
    def pop(self,lowest = True, threshold=0.05):
        """Returns and removes the first element from the note sequence
        The order is determined by the parameters
        args:
            lowst: if True the secondary sorting criteria is pitch, lowest being first
        """
        if len(self.sequence.notes) == 1:
            first = self.sequence.notes[0]
            self.sequence.notes.remove(first)
            return first
        elif len(self.sequence.notes) == 0:
            return None
        
        if lowest:
            def l(x,y):
                if x.start_time-threshold < y.start_time:
                    return x
                elif x.start_time+threshold > y.start_time:
                    return y
                else:
                    if x.pitch<=y.pitch:
                        return x
                    else:
                        return y
        else: #Saves computational time
            def l(x,y):
                if x.start_time-threshold < y.start_time:
                    return x
                elif x.start_time+threshold > y.start_time:
                    return y
                else:
                    if x.pitch>=y.pitch:
                        return x
                    else:
                        return y
        first = reduce(l,self.sequence.notes)
        self.sequence.notes.remove(first)
        
        if len(self.sequence.notes)==1:
            self.start_first = self.sequence.notes[0].start_time
            self.duration  = self.sequence.notes[0].end_time
        else:
            if first.start_time==self.start_first:
                self.start_first = np.min((self.start_first,reduce(lambda x,y: x if x.start_time<y.start_time else y,self.sequence.notes).start_time))            
            if first.end_time==self.duration:
                self.duration = np.max((self.duration,reduce(lambda x,y: x if x.end_time>y.end_time else y,self.sequence.notes).end_time))
        
        return first
    
    def render(self, soundfont_filename=None,max_duration = None,sample_rate=44100):
        """Generate waveform for the stored sequence"""
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
    """ Plots a list of spectrograms or entities
    params:
        spec_list: can either be a list of db power spectra or entities
        sr: sampling rate
        width: subplot width
        height_per_plot: height for subplot
    """

    N = len(spec_list)
    if N>20:
        raise ValueError('Too many spectrograms {}>20. ' +\
                         'To plot a single graph use spec_list=[spec]'.format(N))
    plt.figure(figsize=(width, height_per_plot*N))

    for i in range(N):
        plt.subplot(N,1,i+1)
        spec = spec_list[i]
        if isinstance(spec,audio_complete):
            spec = spec_list[i].D           
        librosa.display.specshow(spec, y_axis='log',x_axis='time',sr=sr)
        plt.ylabel('Log DB')
        plt.colorbar(format='%+2.0f dB')

def audio_from_file(audio_filename):
    """Load the waveform from an audio file. Various extensions supported"""
    return librosa.load(audio_filename,sr=None)    
   
def audio_to_flac(waveform,filename,sr=44100): 
    """Save the waveform to a PCM-24 flac file"""
    soundfile.write(filename, waveform, sr, format='flac', subtype='PCM_24')

def midi_from_file(file_name):
    '''Returns a note sequence structure loaded from the specified file name'''
    return midi_io.midi_file_to_note_sequence(file_name)