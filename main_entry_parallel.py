# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 17:04:08 2019

@author: Hesiris

Dataset: https://colinraffel.com/projects/lmd/
"""

import numpy as np
import os
import argparse
from threading import Lock
from multiprocessing import Pipe, Process, Value, Queue

#from onsetdetector import OnsetDetector as OnsetDetector
#from durationdetector import DurationDetector as DurationDetector
from pitch_classifier import pitch_classifier as PitchClassifier
from instrumentclassifier import InstrumentClassifier as InstrumentClassifier
import util_dataset
from util_audio import note_sequence
from util_audio import audio_complete
import ProgressBar as PB

class Hyperparams:
    def __init__(self, path, sf_path,
                 N=4096, sr=44100, H=None, window_size_note_time=None,
                 parallel_synth=False):
        self.N = N
        self.sr = sr
        self.H = np.int(N / 4) if H is None else H
        self.window_size_note_time = 6 if window_size_note_time is None else window_size_note_time
        self.pitch_input_shape = 20
        self.timing_input_shape = 258
        self.batch_size = 8

        self.kernel_sizes = [(32, 3), (32,3)]
        self.pool_sizes = [(5, 2), (5, 2)]
#        self.kernel_sizes_pitch = [(3, 32), (3, 8)]
#        self.pool_sizes_pitch = [(2, 5), (2, 5)]

        self.checkpoint_dir = './data/checkpoints'
        self.checkpoint_frequency = 5000
        self.convolutional_layer_count = 33
        self.pool_layer_frequency = 12
        self.feature_expand_frequency = 12  # pool_layer_frequency
        self.residual_layer_frequencies = [2]

        self.parallel_synth = parallel_synth
        self.path = path
        self.sf_path = sf_path

def relevant_notes(sequence, offset, duration):
    notes_w, notes_target = sequence.get_notes(offset, offset + duration)
    minus = notes_w.start_first

    notes_w = notes_w.clone()
    print(offset, minus)

    notes_target = notes_target.clone()
    notes_target.shift(-offset)
    notes_w.shift(-offset + minus)
    return notes_target, notes_w

class note_sample:
    def __init__(self,filename, audio, pitch, instrument,
                 onset_s, duration_s):
        self.filename = filename
        self.audio = audio
        self.pitch = pitch
        self.instrument = instrument
        self.onset_s = onset_s
        self.duration_s = duration_s

def thread_classification(model_name,params,q_samples, training_finished,DEBUG=False):
    if model_name == 'pitch':
        if DEBUG:
            print('Loading Pitch Classifier')
        model = PitchClassifier(params)
        model.plot_model(os.path.join(params.path,'model_meta','pitch'+'.png'))
    if model_name == 'instrument':
        if DEBUG:
            print('Loading Instrument Classifier')
        model = InstrumentClassifier(params)
        model.plot_model(os.path.join(params.path,'model_meta','instrument'+'.png'))

    while training_finished.value == 0:
        sample = q_samples.get()
        print('Starting {} Classification'.format(model_name))
        model.classify(sample[0],sample[1])
    

def thread_training(samples_q, params,training_finished, DEBUG = False):
    if DEBUG:
        b_i=0
    
    #Technically pipes may be better, but the ease of use outweighs the 
    #performance penalty, especially compared to audio generation and training
    
    q_pitch = Queue(1)
    q_inst = Queue(1)
    
    proc_pitch = Process(target=thread_classification, 
                            args=('pitch',params, q_pitch,
                                  training_finished,
                                  DEBUG))
    proc_inst = Process(target=thread_classification, 
                        args=('instrument',params, q_inst,
                              training_finished,
                              DEBUG))
    proc_pitch.start()
    proc_inst.start()
    
    while training_finished.value == 0:
        pitch_x,pitch_y = [],[]
        instrument_x,instrument_y = [],[]
#        print('Waiting for samples...')
        for i in range(params.batch_size):
            sample = samples_q.get()
#            print('Sample {} processed'.format(i))
            pitch_x.append(sample.audio)
            pitch_y.append(sample.pitch)
            instrument_x.append(sample.audio)
            instrument_y.append(sample.instrument)
        if DEBUG:
            print('Sending Batch {}'.format(b_i))
            b_i += 1
        
        q_pitch.put((pitch_x,pitch_y))
        q_inst.put((instrument_x,instrument_y))
    
    proc_pitch.join()
    proc_inst.join()

# noinspection PyShadowingNames
def pre_train(params):
    """ Prepare data, training and test"""
    DEBUG = True
    dm = util_dataset.DataManager(params.path, sets=['training', 'test'], types=['midi'])
#    onset_detector = OnsetDetector(params)
#    onset_detector.plot_model(os.path.join(path,'model_meta','onset'+'.png'))
#    duration_detector = DurationDetector(params)
#    duration_detector.plot_model(os.path.join(path,'model_meta','duration'+'.png'))

    frametime = params.H / params.sr
    halfwindow_frames = int(params.timing_input_shape/2)
    halfwindow_time = int(params.window_size_note_time/2)

    if not params.parallel_synth:
        synthesis_lock = Lock()
        
    samples_q = Queue(params.batch_size)

    #Thread that does a training cycle each time q is full
    training_finished = Value('b',0)
    proc_training = Process(target=thread_training, args=(samples_q,params,
                                                          training_finished,
                                                          DEBUG))
    proc_training.start()

    dm.set_set('training') 
    for fn in dm:
        mid = note_sequence(fn[0])

        
        if DEBUG:
            DEBUG = 1
        if DEBUG:
            audio_w_last = None
        offset = 0

        if DEBUG:
            print('Generating wav for midi')
        
        if not params.parallel_synth:
            synthesis_lock.acquire()
        mid_wf = audio_complete(mid.render(params.sf_path), params.N - 2)
        if not params.parallel_synth:
            synthesis_lock.release()
        # TODO - Remove drums - hopefully it can learn to ignore it though
        # TODO -  Add random instrument shifts
        
        if DEBUG:
            print('Creating first cut')
            
        audio_w = mid_wf.section(offset, None, params.timing_input_shape)
        notes_target, notes_w = relevant_notes(mid, offset, 
                                               params.window_size_note_time)
        while offset < mid.duration:

            if DEBUG:
#                fn_base = os.path.join(path, 'debug', os.path.split(fn[0])[-1][:-4] + str(DEBUG))
#                print('Saving to {}'.format(fn_base))
#                if audio_w is not audio_w_last: #only print when new section emerges
#                   audio_w.save(fn_base+'.flac')
#                    if not params.parallel_synth:
#                        synthesis_lock.acquire()
##                   audio_complete(notes_target.render(params.sf_path),params.N-2).save(fn_base + '_target.flac')# -- tested, this works as intended
#                   if not params.parallel_synth:
#                        synthesis_lock.relese()     
#                   audio_w_last = audio_w
#                   notes_target.save(fn_base+'_target.mid')
#                   notes_w.save(fn_base+'_inclusive.mid')
                DEBUG += 1

            note_gold = notes_target.pop(lowest=True, threshold=frametime)
            # training
            if note_gold is not None:
                print('Offset/Note start/end time = {:.3f} / {:.3f} / {:.3f}'.
                      format(offset, note_gold.start_time,note_gold.end_time))
                
                onset_gold = note_gold.start_time
                duration_gold = note_gold.end_time - note_gold.start_time
                pitch_gold = note_gold.pitch
                instrument_gold = note_gold.program
            else:
                onset_gold = offset + halfwindow_time

            # use correct value to move window
            if onset_gold >= halfwindow_time:                
                offset += halfwindow_time
                
                audio_w_new = mid_wf.section(offset+halfwindow_time,
                                             None, halfwindow_frames)
                audio_w_new.mag # Evaluate mag to prime it for the NN. Efficiency trick
                #Otherwise the F would be calculated for both
                audio_w.slice_power(halfwindow_frames, 2*halfwindow_frames)
                audio_w.concat_power(audio_w_new)
                
                notes_target, notes_w = relevant_notes(mid, offset, 
                                                       params.window_size_note_time)
                continue

            audio_sw = audio_w.resize(onset_gold, duration_gold, 
                                      params.pitch_input_shape,
                                      attribs=['mag','ph'])

            sample = note_sample(fn, audio_sw, pitch_gold, instrument_gold,
                                 onset_gold, duration_gold)


#            samples_send.send(sample)            
            samples_q.put(sample)
            
            # subtract correct note for training:
            note_guessed = note_sequence()
            note_guessed.add_note(instrument_gold, instrument_gold, pitch_gold,
                                  0, duration_gold, velocity=100,
                                  is_drum=False)
            
            if not params.parallel_synth:
                synthesis_lock.acquire()
            ac_note_guessed = audio_complete(note_guessed.render(params.sf_path), params.N - 2)
            if not params.parallel_synth:
                synthesis_lock.release()
            audio_w.subtract(ac_note_guessed, offset=onset_gold)
            
#            if DEBUG:
#                print(onset_gold)
            #               ac_note_guessed.save(fn_base+'_guess.flac')
#                note_last = note_gold
 



    proc_training.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AMT-SAGA entry point.')
    parser.add_argument('-soundfont_path',nargs='?',
                        default=os.path.join('..','soundfonts','GM_soundfonts.sf2'),
                        help = 'The path to the soundfont file')
    parser.add_argument('-data_path',
                        default=os.path.join('.','data'),
                        help = 'The directory containing the files for training, \
                        testing etc.')
    parser.add_argument('-parsynth',
                        default = False,
                        help = 'Setting to true will let midi to audio \
                        synthesis be run in parallel. If the background service \
                        doesn\'t suppport it, it may cause unknown behaviour')
    args = vars(parser.parse_args())
    # Hyperparams
    path_data = args['data_path']
    path_sf = args['soundfont_path']
    par_synth = args['parsynth']
    #mp.set_start_method('spawn')
    p = Hyperparams(path_data, path_sf, parallel_synth = par_synth,
                    window_size_note_time=6)
#    try:
    pre_train(p)
#    except KeyboardInterrupt:
#        print("Keyboard Interrupted")
        

