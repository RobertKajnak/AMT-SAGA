#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 14:27:53 2019

@author: hesiris
"""


import rtmidi
import rtmidi.midiconstants as consts
import pyaudio


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
        
        
class sf_synthethizer:
    #'D:/Soundfonts/HD.sf2'
    def __init__(self,soundfont_filename,cache_path):
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
    
    
#    fluid_synth_write_s16 = cfunc('fluid_synth_write_s16', c_void_p,
#                                  ('synth', c_void_p, 1),
#                                  ('len', c_int, 1),
#                                  ('lbuf', c_void_p, 1),
#                                  ('loff', c_int, 1),
#                                  ('lincr', c_int, 1),
#                                  ('rbuf', c_void_p, 1),
#                                  ('roff', c_int, 1),
#                                  ('rincr', c_int, 1))
    def _fluid_synth_write_s16_stereo(self,synth, len):
        """Return generated samples in stereo 16-bit format
        
        Return value is a Numpy array of samples.
        
        """
        import numpy
        buf = create_string_buffer(len * 4)
        self.fluid_synth_write_s16(synth, len, buf, 0, 2, buf, 1, 2)
        return numpy.frombuffer(buf[:], dtype=numpy.int16)
    
    def render_notes(self,notes,start=0,end=-1):
        
        
        
        return None
    
    def generate_note(self,pitch, duration, raw_audio=None, extend = False ,start=0, 
                 instrument=None,velocity=127):
        '''Synthetize the audio based on the soundfont file
            args:
                The generated sound will be added to the raw_audio variable. 
                A new array is generated if None is specified 
            returns:
                (raw_audio,sample_rate)    
        '''
        pass
        
        
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

        