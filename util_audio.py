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

from threading import Timer

import wave

#https://www.midi.org/specifications-old/item/table-3-control-change-messages-data-bytes-2
#http://people.csail.mit.edu/hubert/pyaudio/
#https://github.com/SpotlightKid/python-rtmidi/blob/master/examples/advanced/midiwrapper.py


class audio_io:
    def __init__(self, channel_name = None):
        ''' If a channel name is specified, it also connects to it
        '''
        self.PA = pyaudio.PyAudio()
        self.RATE = 0
        
        if channel_name is not None:
            self.connect_to_channel(channel_name)
        
    def list_input_devices(self):
        '''Lists all devices that have at least 1 available output channel
        '''
        for i in range(self.PA.get_device_count()):
            inf = self.PA.get_device_info_by_index(i)
            if inf['maxInputChannels'] > 0:
                print('{}: {}'.format(inf['index'],inf['name']))
            
    def list_output_devices(self):
        '''Lists all devices that have at least 1 available output channel
        '''
        for i in range(self.PA.get_device_count()):
            inf = self.PA.get_device_info_by_index(i)
            if inf['maxOutputChannels'] > 0:
                print('{}: {}'.format(inf['index'],inf['name']))
    
    def get_channel_index_by_name(self,name):
        ''' Returns the channel index for the channel that contains the name
            substring. If no channel is found -1 is returned
        '''
        for i in range(self.PA.get_device_count()):
            inf = self.PA.get_device_info_by_index(i)
            if name.upper() in inf['name'].upper() :
                return inf['index']
        return -1
    
    def connect_to_channel(self,name,index = None, channels=2,rate=-1):
        '''
        args:
            index: if index is specified, it overrides name
        '''
        if index is None:
            inf = self.PA.get_device_info_by_index(self.get_channel_index_by_name(name))
        else:
            inf = self.PA.get_device_info_by_index(index)
        print('Connected to: {}'.format(inf['name']))
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = channels
        self.RATE = int(inf['defaultSampleRate']) if rate==-1 else rate
    
        
    def record(self,duration):
        ''' Synchronous record, blocking
        '''
        if self.RATE == 0:
            raise ValueError('No channel connected. Use connect_to_channel()')
        
        audio_frames=[] 
        
        stream = self.PA.open(format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK)     
           
        for i in range(0, int(self.RATE / self.CHUNK * duration)):
            data=stream.read(self.CHUNK)
            audio_frames.append(data)

        stream.close()
        
        wf = wave.open('sample.wav', 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.PA.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(audio_frames))
        wf.close()
        
        return audio_frames
        
    def save(self, filename, frames):
        soundfile.write(filename, 
            np.frombuffer(b''.join(frames),dtype=np.int16), 
            self.RATE, format='flac', subtype='PCM_24')
    
    def close(self):
        self.PA.terminate()

class sf_synthethizer:
    #'D:/Soundfonts/HD.sf2'
    def __init__(self,soundfont_filename):
        with open(soundfont_filename,'rb') as sf2_file:
            self.sf = Sf2File(sf2_file)
#        self.all_presets = {}
#        for pres in self.sf.presets:
#            if pres.name != 'EOP':
#                if pres.preset not in self.all_presets:
#                self.all_presets[pres.preset] =[]
#                self.all_presets[pres.preset].append(pres)
#            else:
#                self.all_presets[pres.preset].append(pres)
        self.all_presets = [[] for i in range(128)]
        for pres in self.sf.presets:
            if pres.name !='EOP':
                self.all_presets[pres.preset].append(pres)
        #keys_sorted = sorted(all_pres.keys())
        
    def get_sample():
        return None
    def render_notes(self,notes,start,duration):
        #waw = np.zeros(,dtype.np.double)
        return None
        
    def generate(self,pitch, duration, raw_audio=None, extend = False ,start=0, 
                 instrument=None,velocity=127):
        '''Synthetize the audio based on the soundfont file
            args:
                The generated sound will be added to the raw_audio variable. 
                A new array is generated if None is specified 
            returns:
                (raw_audio,sample_rate)    
        '''
        
        
        
        #soundfile.write(path + 'sample.wav', 
        #            np.frombuffer(sf.samples[0].raw_sample_data,dtype=np.int16), 
        #            sf.samples[0].sample_rate, format='flac', subtype='PCM_24')
        
        #program == preset[i].perset
        
        
    def close(self):
        #self.sf2_file.close()
        pass
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        
class synthetizer:
    def __init__(self, open_port=-1,record_audio = True):
        '''
        args:
            open_port: -1: open last installed port
                        None: do not open port. Port will need to be opened using open_port
        '''
        self.midiout = rtmidi.MidiOut()
        self.available_ports = self.midiout.get_ports()
        self.prev_instrument = -1
        
        if open_port == -1:
            self.open_port(len(self.available_ports)-1)
        elif open_port is not None:
            self.open_port(open_port)
        
        if record_audio:
            self.PA = audio_io()

        
    def list_ports(self):
        '''returns a list of all the available MIDI ports
        '''
        return self.available_ports
    
    #TODO not very good, can lead to accidentally opening more ports
    def open_port(self,port_id):
        '''Opens the port specified. To list available ports see list_ports()'''
        self.port_id = port_id
        if self.available_ports:
            self.midiout.open_port(port_id)
        else:
            self.midiout.open_virtual_port("My virtual output")
        
    #TODO this does not work, disabled it from the API
    def set_reverb(self):
        '''Attempts to switch off reverb'''
        self.midiout.send_message([91,0])
        self.midiout.send_message([consts.CONTROL_CHANGE,91,0])
        
        
    def play(self, pitch, duration,instrument=None,velocity=127):
        ''' Generates a note with the parameters specified.
        params:
            pitch: pitch in midi value
            velocity: ~
            duration: in seconds
            instrument: if None, the previously used instrument is used.
                        otherwise the switch will be made to the specified program code
        '''
        if instrument is not None and instrument!=self.prev_instrument:
            self.midiout.send_message([consts.PROGRAM_CHANGE,instrument])
        self.prev_instrument = instrument
        
        note_on = [consts.NOTE_ON, pitch, velocity] # channel 1, middle C, velocity 112
        note_off = [consts.NOTE_OFF, pitch, 0]
        
#        audio_frames=[]     
#        stream = p.open(format=self.PA.FORMAT,
#                channels=self.PA.CHANNELS,
#                rate=self.PA.RATE,
#                input=True,
#                frames_per_buffer=self.PA.CHUNK)

        self.midiout.send_message(note_on)
        
        for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            audio_frames.append(stream.read(self.CHUNK))

        self.midiout.send_message(note_off)
        stream.close()
        
        #time.sleep(duration)
        #self.midiout.send_message(note_off)
#        Timer(duration,self.midiout.send_message,[note_off]).start()
#        return audio_frames
    
    
    def silence(self):
        '''Sends a sound off signal to all pitches'''
        self.midiout.send_message([consts.ALL_SOUND_OFF])
        
    def close(self):
        '''closes the opened port. with [...] structure can be used instead'''
        self.midiout.close_port()
        if self.PA:
            self.PA.close()
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        
class note_sequence:
    def __init__(self, filename= None):
        
        if filename is None:    
            self.sequence = music_pb2.NoteSequence()
        else:
            self.sequence = midi_io.midi_file_to_note_sequence(filename)
            
            
    def add_note(self, instrument, program, pitch, start, end, velocity=127,  is_drum=False):
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
                
    def save(self,file_name):
        '''Saves the MIDi file to the specified filename'''
        midi_io.note_sequence_to_midi_file(self.sequence, file_name)
        
def audio_from_file(audio_filename):
    return librosa.load(audio_filename,sr=None)        

def midi_from_file(file_name):
    '''Returns a note sequence structure loaded from the specified file name'''
    return midi_io.midi_file_to_note_sequence(file_name)