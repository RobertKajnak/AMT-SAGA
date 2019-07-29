# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 17:05:15 2019

@author: Hesiris
"""


import copy
import os

import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import bisect

from magenta.music import midi_io
from magenta.protobuf import music_pb2
import librosa
import soundfile

try:
    import fluidsynth
    _HAS_FLUIDSYNTH = True
except ImportError:
    _HAS_FLUIDSYNTH = False

#https://www.midi.org/specifications-old/item/table-3-control-change-messages-data-bytes-2
#http://people.csail.mit.edu/hubert/pyaudio/
#https://github.com/SpotlightKid/python-rtmidi/blob/master/examples/advanced/midiwrapper.py

class audio_complete:
    def __init__(self,waveform,n_fft,hop_length=None,center=True,sample_rate=44100):
        """ A Class that holds both waveform and spectral information. 
            This provides a lighweight wrapper around the librosa library and
            provides some additional functionality.
            
            All properties are claculated only when called. If a property is 
            modified any other property that becomes out of synch with the new
            information is removed and recalculated when the respective 
            property is called.
            params:
                waveform: Can be set to None, if only spectral information is
                    intended to be used later.
                n_fft: Fast Furier Transform window size
                hop_length: number audio of frames between STFT columns. 
                    If unspecified, defaults win_length / 4.
                center: If True, the signal y is padded so that frame D[:, t] 
                    is centered at y[t * hop_length].
                    If False, then D[:, t] begins at y[t * hop_length]
                sample_rate: sample rate of the waveform
                
            For more information visit librosa.github.io
        """
        self._wf = waveform
        self._F = None
        self._mag = None
        self._ref_mag = None
        self._ph = None
        self._D = None
        
        self.sr = sample_rate
        self.N = n_fft
        self.center = center
        self.hl = hop_length if hop_length is not None else int(np.floor(n_fft/4))
        
        self._fft_freq = librosa.core.fft_frequencies(sample_rate,n_fft)
        
    def clone(self):
        """Copies the parameters and a deepcopy of the available properties"""
        wf = copy.deepcopy(self._wf)
        ac = audio_complete(wf, sample_rate = self.sr, n_fft = self.N,
                              center = self.center, hop_length = self.hl)
        
        F = copy.deepcopy(self._F)
        mag = copy.deepcopy(self._mag)
        ref_mag = self._ref_mag
        ph = copy.deepcopy(self._ph)
        D = copy.deepcopy(self._D)
           
        ac._F = F
        ac._mag = mag
        ac._ref_mag = ref_mag
        ac._ph = ph
        ac._D = D
        
        return ac
    @property
    def wf(self):
        if self._wf is None:
            if self._F is not None:
                self._wf = librosa.istft(self._F,
                                       hop_length=self.hl,center=self.center)
            elif self._mag is not None and self._ph is not None:
                self._F = self._mag * self._ph
                self._wf = librosa.istft(self._F,
                                       hop_length=self.hl,center=self.center)
            elif self._D is not None and self._ph is not None:
                if self._ref_mag is None:
                    self._ref_mag = 1.0;
                self._mag = librosa.db_to_amplitude(self._D,ref = self._ref_mag)
                self._F = self._mag * self._ph
                self._wf = librosa.istft(self._F,
                                       hop_length=self.hl,center=self.center)
                
        return self._wf
    @wf.setter
    def wf(self,value):
        self._D = None
        self._ref_mag = None
        self._mag = None
        self._ph = None
        self._F = None
        self._wf = value
    
    @property
    def F(self):
        if self._F is None:
            if self._mag is not None and self._ph is not None:
                self._F = self._mag * self._ph
            elif self._D is not None and self._ph is not None:
                if self._ref_mag is None:
                    self._ref_mag = 1.0;
                self._mag = librosa.db_to_amplitude(self._D,ref = self._ref_mag)
                self._F = self._mag * self._ph
            elif self.wf is not None:
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
        self._wf = None
    
    @property
    def mag(self):
        if self._mag is None:
            if self._D is not None and self._ph is not None:
                if self._ref_mag is None:
                    self._ref_mag = 1.0;
                self._mag = librosa.db_to_amplitude(self._D,ref = self._ref_mag)
            else:
                self._mag,self._ph = librosa.core.magphase(self.F)
        return self._mag
    @mag.setter
    def mag(self,val):
        self._D = None
        self._ref_mag = None
        self._mag = val
        if self._ph is not None and self._ph.shape != val.shape:
            self._ph = None
        self._F = None
        self._wf = None
        
    @property
    def ph(self):
        if self._ph is None:
            self._mag,self._ph = librosa.core.magphase(self.F)
        return self._ph
    @ph.setter
    def ph(self,val):
        self._ph = val
        self._F = None
        self._wf = None

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
        if self._ph is not None and self._ph.shape != val.shape:
            self._ph = None
        #self._ref_mag = None -- If only the D is modified, the reference is
        #probably still the same
        self._mag = None
        self._F = None
        self._wf = None
    
    def _P(self,name):
        """Returns the raw attribute, by name. 
        Example: name=='wf' => return self._wf
        """
        if name == 'wf':
            return self._wf
        elif name == 'F':
            return self._F
        elif name == 'mag':
            return self._mag
        elif name == 'ph':
            return self._ph
        elif name == 'D':
            return self._D
        else:
            raise ValueError('Requested attribute does not exist')
    
    @property
    def shape(self):
        if self._mag is not None:
            return self._mag.shape
        if self._ph is not None:
            return self._mag.shape
        if self._D is not None:
            return self._D.shape
        #if nothing esle is set, F needs to be calculated from wf
        return self.F.shape
    
    
    def subtract(self,subtrahend, offset=0,attack_compensation=0,
                     normalize = True, relu = True, overkill_factor = 1):
        """ Subtracts the magnitude information of the provided subtrahend from
            the current class' information. 
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
        
        offset = max(self._seconds_to_frames(offset) - attack_compensation, 0)
        
        if mag_sub.shape[1]+offset>self.mag.shape[1]:
            mag_sub = mag_sub[:,:(self.mag.shape[1]-offset)]
        
        self.mag -= np.concatenate(
                         (np.zeros((self.mag.shape[0],offset)) ,
                         mag_sub,
                         np.zeros((self.mag.shape[0],self.mag.shape[1]-offset-mag_sub.shape[1]))),
                         axis = 1)
        if relu:
            self.mag = np.maximum(self.mag,0,self.mag)
        
    def _seconds_to_frames(self,time):
        """Converts a from seconds to frames in fourier. Window is compensated,
            but a value of 0 always returns 0"""
        return int(np.floor(time * self.shape[1]*self.sr / self.wf.shape[0]))
#        return max(0,librosa.core.time_to_frames(time,sr=self.sr, 
#                                                 hop_length=self.hl, 
#                                                 n_fft=self.N))
    
    def _frames_to_seconds(self,frames):
        """Converts a from fourier frames to seconds. Window is compensated,
            but a value of 0 always returns 0"""
        return frames/self.shape[1]/ self.sr * self.wf.shape[0]
#        if frames == 0:
#            return 0
#        return librosa.core.frames_to_time(frames,sr=self.sr, hop_length=self.hl, 
#                                           n_fft=self.N)
    
    def midi_tone_to_FFT(self, tone):
        """ Calculates the closest FFT bin (rounded down) to the midi tone
        specified"""
        f = librosa.core.midi_to_hz(tone)
        ind = bisect.bisect_right(self._fft_freq,f) - 1
        ind = 0 if ind==0 else ind-1
        return ind
    
    def section(self, start, end, duration_in_frames=None):
        """Creates a copy of a section and returns it. The D, F et.c will 
        remain calculated. If the section specified is bigger than the array, 
        the remainder is padded with 0s
        params:
            start: start time in seconds, rounded down to the nearest Fourier frame
            end: end time in seconds, rounded down to the nearest Fourier frame
            duration_in_frames: duration of the slice, expressed in frames. 
                Overwrites end time if speicified"""
        tfs = self._seconds_to_frames(start)
        if duration_in_frames is None:
            tfe = self._seconds_to_frames(end)
        else:
            tfe = tfs+duration_in_frames
        
        if self._wf is not None:
            wav_start = int(np.floor(self._frames_to_seconds(tfs) * self.sr))
            wav_end = int(np.floor(self._frames_to_seconds(tfe) * self.sr))
            wav_cp = copy.deepcopy(self.wf[wav_start:wav_end])
            if wav_cp.shape[0]<wav_end-wav_start:
                wav_cp = np.concatenate((wav_cp,np.zeros(wav_end - wav_cp.shape[0])))
        else:
            wav_cp = None
        nac = audio_complete(wav_cp,
                             self.N,hop_length=self.hl,center=self.center,sample_rate=self.sr)
        
        def cc(f):
            if f is not None:
                cpd = copy.deepcopy(f[:,tfs:tfe])
                if f.shape[1]>=tfe:
                    return cpd
                else:
                    return np.concatenate((cpd,np.zeros(
                                            (f.shape[0],tfe-f.shape[1])))
                                          ,axis=1)
    
        nac._F = cc(self._F)
        nac._ref_mag = self._ref_mag
        nac._mag = cc(self._mag)
        nac._ph =cc(self._ph)
        nac._D = cc(self._D)
        
        return nac
    
    def section_power(self,name,band_min,band_max):
        """Returns a an array containing a copy of the part of the spectra. 
        if the maximmum band is outside the psecified range, the result will
        be padded with 0s
        params:
            name: string. Can be one of the following: 'F','mag','ph','D'
            band_min: lowest band, inclusive
            band_max: highest band, exclusive
        """
        P = self._P(name)
        h = P.shape[0]
        cpd = copy.deepcopy(P[band_min:band_max,:])
        if band_max>h:
            cpd = np.concatenate((cpd,np.zeros((band_max-h,P.shape[1])))
                                          ,axis=0)
        return cpd
    
    def slice(self,start_in_frames, end_in_frames):
        """Takes a slice from both power and waveform, where available.
        The data is not copied, the class is modified. Section outside are lost.
        """
        if self._wf is not None:
            self._wf = self._wf[int(self._frames_to_seconds(start_in_frames)*self.sr):
                                int(self._frames_to_seconds(end_in_frames)*self.sr)]
        if self._F is not None:
            self._F = self._F[:,start_in_frames:end_in_frames]
        if self._mag is not None:
            self._mag = self._mag[:,start_in_frames:end_in_frames]
        if self._ph is not None:
            self._ph = self._ph[:,start_in_frames:end_in_frames]
        if self._D is not None:
            self._D = self._D[:,start_in_frames:end_in_frames]
            

    def _concus(self,dest,src,axis=1):
        if src is None or dest is None:
            return None
        else:
            return np.concatenate((dest,src),axis=axis)
    
    def concat(self,ac):
        """Concatenates two ac's and waveforms, where available
        The data is not copied"""
        
        self._wf = self._concus(self._wf,ac._wf,axis=0)
        self._F = self._concus(self._F,ac._F)
        self._mag = self._concus(self._mag,ac._mag)
        self._ph = self._concus(self._ph,ac._ph)
        self._D = self._concus(self._D,ac._D)

    @staticmethod
    def _resize(P,target_frame_count):
        """Generic funciton to resize any property"""
        t = P.shape[1]
        if t==0:
            return np.zeros((P.shape[0],target_frame_count))
        elif t==target_frame_count:
            resd = P
        elif t<3:
            resd = np.concatenate((P[:,:1],np.tile(P[:,-1:],target_frame_count-1)),axis=1)
        elif t<target_frame_count:
            lim = np.min((1,int(np.round(t/3))))
            l_t = int(np.floor((target_frame_count - 2 * lim)/(t-2*lim)))
            tiled = np.tile(P[:,lim:-lim],l_t)
            resd = np.concatenate((P[:,:lim],
                                   tiled,
                                   P[:,-(target_frame_count-tiled.shape[1]-lim):]),
                                   axis=1)
        else:
#            lim = np.int(np.min([4,target_frame_count/3]))
#            cent = int(np.floor(P.shape[1]/2))
#            cl = cent - int(np.floor(target_frame_count/2)) + lim
#            ch = cent + int(np.ceil(target_frame_count/2)) - lim 
#            resd = np.concatenate((P[:,:lim],P[:,cl:ch],P[:,-lim:]),axis=1)
            resd = P[:,:target_frame_count]
        return resd
    
    def slice_C(self,start,duration,target_frame_count,
                magnitude_only = True,
                bins_per_tone = 1,filter_scale=2,
                highest_note = 'C8', lowest_note='A0',nbins=None):
        """
        Calculates the CQT and slices a part of it. 
        If nbins is specified, the highest note is ignored
        """
        
        if nbins is None:
            nbins = int((librosa.note_to_midi(highest_note) - 
                     librosa.note_to_midi(lowest_note))*bins_per_tone)
            
        C = librosa.cqt(self.wf,sr=self.sr,fmin=librosa.note_to_hz(lowest_note),
                        n_bins=nbins,bins_per_octave=int(12*bins_per_tone),
                        filter_scale=2,hop_length = self.hl)
        
        if magnitude_only:
            C=np.abs(C)
        
        t = self._seconds_to_frames(start+duration)
        s = self._seconds_to_frames(start)     

        return self._resize(C[:,s:t],target_frame_count)
    
    @staticmethod
    def compress_bands(spectrum, bands=80, log=True):
        """ Compresses the frequency bands by averaging neighbouring bands.
        Naive approach. Will cause artefacts. Use with caution
        params:
            spectrum: the power spectrum to compress
            bands: number of bands desired at the output. If there is 
                a remainder, the last ones are ignored
                #the last will be summed with the previous ones.
                #Example: [1,2,3,4,5] -> 2 -> [1+2],[3+4+5]
        """
        
        ns = np.zeros((bands,spectrum.shape[1]))
        
        if log:
            ind = np.geomspace(1,spectrum.shape[0],bands+1,dtype=np.int)
            ind[0] = 0
            for i in range(0,bands):
                sub = ind[i+1]-ind[i]
                if sub<1:
                    ind[i+1] += -sub + 1
                        
            for j in range(spectrum.shape[1]):
                for i in range(bands):
                    ns[i,j] = np.mean(spectrum[int(ind[i]):int(ind[i+1]),j])
        else:
            r = spectrum.shape[0] // bands
            for i in range(bands):
                for j in range(spectrum.shape[1]):
                    ns[i,j] = np.mean(spectrum[r*i:r*(i+1),j]) 
        return ns
        
    
    def resize(self,start,duration,target_frame_count, attribs=['F']):
        """ A new ac is returned that has a single data attribute set, 
            specified as a parameter. Hyperparameters copied. ref_mag copied 
            if available. Does not modify the instance content.
            params:
                start: start time in seconds for the subsection
                duration: duration in seconds of the subsection. Can be larger,
                    smaller or equal to the target. The content is adjusted 
                    based on the relative sizes
                target_frame_count: the number of frames expected at the output
                attribs: The attributes to perform the transformation on.
                    Possibilities: 'F', 'mag', 'ph' (suggested to use together),
                        'D'. Otherwise an exception will be thrown
                    If the attribute is missing from the current instance, it
                    will be calculated
                
        """
        nac = audio_complete(None,
                     self.N,hop_length=self.hl,center=self.center,
                     sample_rate=self.sr)
        if self._ref_mag is not None:
            nac._ref_mag = self._ref_mag
        
        t = self._seconds_to_frames(start+duration)
        s = self._seconds_to_frames(start)        
        #If it is longer, it will take a shorter section starting from s
        for attrib in attribs:
            if attrib=='F':
                nac.F = self._resize(self.F[:,s:t],target_frame_count)
            elif attrib=='mag':
                nac.mag = self._resize(self.mag[:,s:t],target_frame_count)
            elif attrib =='ph':
                nac.ph = self._resize(self.ph[:,s:t],target_frame_count)
            elif attrib =='D':
                nac.D = self._resize(self.ph[:,s:t],target_frame_count)
            else:
                raise ValueError('Invalid attribute requested')
                
        return nac
        
    def plot_spec(self,width=12,height = 5):
        "Plots the logarithmic spectrogram. Caclutes D if unavailable."""
        
        plt.figure(figsize=(width, height))

        plt.subplot(1,1,1)
        librosa.display.specshow(self.D, y_axis='log',x_axis='time',sr=self.sr,hop_length = self.hl)
        plt.ylabel('Log DB')
        plt.colorbar(format='%+2.0f dB')

    def save(self,filename):
        """Saves the waveform to the specified flac filename with FLAC codec"""
        audio_to_flac(waveform = self.wf,
                      filename=filename,sr=self.sr)
        

class note_sequence:
    #'D:/Soundfonts/HD.sf2'
    #'/home/hesiris/Documents/Thesis/GM_soundfonts.sf2'
    fl = None
    sfid = None
    sf_path = None
    
    _notes = {'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,'F#':6,
             'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'Hb':10,
                 'B':11,'H':11}  
    
    def __init__(self, filename= None, sf2_path = None):
        """A wrapper class around the magenta midi wrapper that goes around pretty_midi
        filename: The class will attampt to load the midi file in the specified filename
        sf2_path: If the class is intended to be used for midi rendering,
            The soundfont file can be loaded here. If left None, the file will
            be loaded on first call to render. The soundfont file will be 
            loaded and shared across all instances of the class. If a new 
            filename is specified, the new one will be loaded.
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
        
        if sf2_path is not None:
            note_sequence._init_fluidsynth(sf2_path,fs=44100)      
            note_sequence.sf_path = sf2_path
        
    def clone(self):
        """Returnes a copy of the contents of the class"""
        s_clone = note_sequence()
        s_clone.prev_octave = self.prev_octave
        
        for note in self.sequence.notes:
            s_clone.append(note,copy=True)
        return s_clone
    
    def append(self, note, copy=False):
        """If copy is set to true, the note is cloned first. 
        Otherwise reference is passed
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

    def get_notes(self,start, end, include_drums = True):
        ''' Returns:
                all notes that have sound inside the specified period
                all notes that have both start and end points within spec. period
        '''
        subsequence_inclusive = music_pb2.NoteSequence()
        subsequence_exclusive = music_pb2.NoteSequence()
        notes_to_add_inclusive = []
        notes_to_add_exclusive = []
        for note in self.sequence.notes:
            if (note.start_time>start and note.end_time<end) and \
            (include_drums or note.is_drum == include_drums):
                notes_to_add_exclusive.append(note)
                notes_to_add_inclusive.append(note)
            elif (note.end_time>start and note.start_time<end) and \
            (include_drums or note.is_drum == include_drums):
                notes_to_add_inclusive.append(note)
                
                
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
    
    def pop(self,lowest = True, threshold=0.05):
        """Returns and removes the first element from the note sequence
        The order is determined by the parameters
        args:
            lowest: if True the secondary sorting criteria is pitch, lowest being first
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
    
    def shift(self, time):
        """Shifts all notes by the specified amount. Positive amount adds 
            delay, negative value makes the notes be played earlier"""
        for note in self.sequence.notes:
            note.start_time+=time
            note.end_time+=time
        
        self.duration += time
        self.start_first += time
        
    def merge_duplicates(self,temporal_resolution = 100):
        """ Merges notes that are of the same instrument and are at least partially
        overlapping, i.e. start1"""
        
        #TODO For each instrument create a binary piano roll, check overlaps and 
        #modify notes accordingly
        pass
        
    def change_program_for_instrument(self,instrument,program_target):
        """ Replaces the program of the instrument with the new one.
        WARNING: Manual verification is advised, sometimes track metadata may
        cause different results than expected"""
        for note in self.sequence.notes:
            if note.instrument == instrument:
                note.program = program_target
                
    def change_program(self,original,target):
        """All notes of the specified program will be changed to the new one
        WARNING: Manual verification is advised, sometimes track metadata may
        cause different results than expected"""
        for note in self.sequence.notes:
            if note.program == original:
                note.program = target
    
    def render(self, sf2_path=None, max_duration = None,sample_rate=44100):
        """Generate waveform for the stored sequence
        params:
            sf2_path: needs to be specified on first run. To run without the 
                ability to render set to None The soundfont file will be 
                loaded and shared across all instances of the class. If a new 
                filename is specified, the new one will be loaded.
        """
        
        if note_sequence.fl is None or note_sequence.sfid is None:
            if sf2_path is None:
                raise ValueError("sf2_path needs to specified on first \
                                 instance creation or class method \
                                 _init_fluidsynth should be called first.")
            elif note_sequence.sf_path != sf2_path:
                note_sequence._init_fluidsynth(sf2_path,fs=44100)
                note_sequence.sf_path = sf2_path
                
        mid = midi_io.note_sequence_to_pretty_midi(self.sequence)
        wf = self._fluidsynth_midi_stateful(mid, fs=sample_rate)
        
        if max_duration is None:
            return wf
        else:
            return wf[:max_duration*sample_rate]
    def __str__(self):
        return self.sequence.__str__()
        
    def save(self,file_name):
        '''Saves the MIDi file to the specified filename'''
        midi_io.note_sequence_to_midi_file(self.sequence, file_name)
        
    @classmethod
    def _init_fluidsynth(cls,sf2_path,fs=44100):
        """ Initializes fluidsynth object and adds loads the soundfont. 
            If a soundfont is already present, the old one is discarded.
        """
        if not _HAS_FLUIDSYNTH:
            raise ImportError("fluidsynth() was called but pyfluidsynth "
                              "is not installed.")

        if not os.path.exists(sf2_path):
            raise ValueError("No soundfont file found at the supplied path "
                             "{}".format(sf2_path))
            
        if cls.fl is not None:
            cls._close_fluidsynth()
        # Create fluidsynth instance
        cls.fl = fluidsynth.Synth(samplerate=fs)
        # Load in the soundfont
        cls.sfid = cls.fl.sfload(sf2_path)
        
    @classmethod
    def _close_fluidsynth(cls):
        """Close fluidsynth"""
        cls.fl.delete()
        
    def _fluidsynth_instrument_stateful(self,instrument_preddy_midi, fs=44100):
        """Taken from github.com/craffel/pretty-midi/ 
        The key change is that sf2_path can now be a loaded file, 
        thus reducing disk load. Default file load removed.
        
        def fluidsynth(self, fs=44100, sf2_path=None):
        Synthesize using fluidsynth.
        Parameters
        ----------
        intrument_pretty_midi: object
            instnce of the pretty_midi_instrument
        fs : int
            Sampling rate to synthesize.
        Returns
        -------
        synthesized : np.ndarray
            Waveform of the MIDI data, synthesized at ``fs``.
        """

        # If the instrument has no notes, return an empty array
        if len(instrument_preddy_midi.notes) == 0:
            return np.array([])


        if instrument_preddy_midi.is_drum:
            channel = 9
            # Try to use the supplied program number
            res = note_sequence.fl.program_select(channel, note_sequence.sfid, 128, instrument_preddy_midi.program)
            # If the result is -1, there's no preset with this program number
            if res == -1:
                # So use preset 0
                note_sequence.fl.program_select(channel, note_sequence.sfid, 128, 0)
        # Otherwise just use channel 0
        else:
            channel = 0
            note_sequence.fl.program_select(channel, note_sequence.sfid, 0, instrument_preddy_midi.program)
        # Collect all notes in one list
        event_list = []
        for note in instrument_preddy_midi.notes:
            event_list += [[note.start, 'note on', note.pitch, note.velocity]]
            event_list += [[note.end, 'note off', note.pitch]]
        for bend in instrument_preddy_midi.pitch_bends:
            event_list += [[bend.time, 'pitch bend', bend.pitch]]
        for control_change in instrument_preddy_midi.control_changes:
            event_list += [[control_change.time, 'control change',
                            control_change.number, control_change.value]]
        # Sort the event list by time, and secondarily by whether the event
        # is a note off
        event_list.sort(key=lambda x: (x[0], x[1] != 'note off'))
        # Add some silence at the beginning according to the time of the first
        # event
        current_time = event_list[0][0]
        # Convert absolute seconds to relative samples
        next_event_times = [e[0] for e in event_list[1:]]
        for event, end in zip(event_list[:-1], next_event_times):
            event[0] = end - event[0]
        # Include 1 second of silence at the end
        event_list[-1][0] = 1.
        # Pre-allocate output array
        total_time = current_time + np.sum([e[0] for e in event_list])
        synthesized = np.zeros(int(np.ceil(fs*total_time)))
        # Iterate over all events
        for event in event_list:
            # Process events based on type
            if event[1] == 'note on':
                note_sequence.fl.noteon(channel, event[2], event[3])
            elif event[1] == 'note off':
                note_sequence.fl.noteoff(channel, event[2])
            elif event[1] == 'pitch bend':
                note_sequence.fl.pitch_bend(channel, event[2])
            elif event[1] == 'control change':
                note_sequence.fl.cc(channel, event[2], event[3])
            # Add in these samples
            current_sample = int(fs*current_time)
            end = int(fs*(current_time + event[0]))
            samples = note_sequence.fl.get_samples(end - current_sample)[::2]
            synthesized[current_sample:end] += samples
            # Increment the current sample
            current_time += event[0]

        return synthesized
    
    def _fluidsynth_midi_stateful(self, midi_object, fs=44100,normalize= True):
        """Taken from github.com/craffel/pretty-midi/ 
        The key change is that sf2_path can now be a loaded file, 
        thus reducing disk load. Default file load removed.
        Synthesize using fluidsynth.
        Parameters
        ----------
        fs : int
            Sampling rate to synthesize at.
        sf2_path : str
            Path to a .sf2 file.
            Default ``None``, which uses the TimGM6mb.sf2 file included with
            ``pretty_midi``.
        Returns
        -------
        synthesized : np.ndarray
            Waveform of the MIDI data, synthesized at ``fs``.
        """
        # If there are no instruments, or all instruments have no notes, return
        # an empty array
        if len(midi_object.instruments) == 0 or all(len(i.notes) == 0
                                             for i in midi_object.instruments):
            return np.array([])
        # Get synthesized waveform for each instrument
        waveforms = [self._fluidsynth_instrument_stateful(i,fs=fs,
                                  ) for i in midi_object.instruments]
        
        # Allocate output waveform, with #sample = max length of all waveforms
        synthesized = np.zeros(np.max([w.shape[0] for w in waveforms]))
        # Sum all waveforms in
        for waveform in waveforms:
            synthesized[:waveform.shape[0]] += waveform
        # Normalize
        if normalize:
            synthesized /= np.abs(synthesized).max()
        return synthesized
        
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