# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 17:04:08 2019

@author: Hesiris

Dataset: https://colinraffel.com/projects/lmd/
"""

import numpy as np
import os
import argparse

from onsetdetector import OnsetDetector as OnsetDetector
from durationdetector import DurationDetector as DurationDetector
from pitch_classifier import pitch_classifier as PitchClassifier
from instrumentclassifier import InstrumentClassifier as InstrumentClassifier
import util_dataset
from util_audio import note_sequence
from util_audio import audio_complete
import ProgressBar as PB

class Hyperparams:
    def __init__(self, N=4096, sr=44100, H=None, window_size_note_time=None):
        self.N = N
        self.sr = sr
        self.H = np.int(N / 4) if H is None else H
        self.window_size_note_time = 6 if window_size_note_time is None else window_size_note_time
        self.pitch_input_shape = 20
        self.timing_input_shape = 258

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


def relevant_notes(sequence, offset, duration):
    notes_w, notes_target = sequence.get_notes(offset, offset + duration)
    minus = notes_w.start_first

    notes_w = notes_w.clone()
    print(offset, minus)

    notes_target = notes_target.clone()
    notes_target.shift(-offset)
    notes_w.shift(-offset + minus)
    return notes_target, notes_w


# noinspection PyShadowingNames
def pre_train(path, sf_path, params):
    """ Prepare data, training and test"""
    DEBUG = True
    dm = util_dataset.DataManager(path, sets=['training', 'test'], types=['midi'])
#    onset_detector = OnsetDetector(params)
#    onset_detector.plot_model(os.path.join(path,'model_meta','onset'+'.png'))
#    duration_detector = DurationDetector(params)
#    duration_detector.plot_model(os.path.join(path,'model_meta','duration'+'.png'))
    if DEBUG:
        print('Loading Pitch Classifier')
    pitch_classifier = PitchClassifier(params)
    pitch_classifier.plot_model(os.path.join(path,'model_meta','pitch'+'.png'))
    if DEBUG:
        print('Loading Instrument Classifier')
    instrument_classifier = InstrumentClassifier(params)
    instrument_classifier.plot_model(os.path.join(path,'model_meta','instrument'+'.png'))
    frametime = params.H / params.sr
    halfwindow_frames = int(params.timing_input_shape/2)
    halfwindow_time = int(params.window_size_note_time/2)

    pb = PB.ProgressBar(300000)
    dm.set_set('training')
    for fn in dm:
        mid = note_sequence(fn[0])

        sheet = note_sequence()
        
        if DEBUG:
            DEBUG = 1
        if DEBUG:
            audio_w_last = None
        offset = 0

        if DEBUG:
            print('Generating wav for midi')
        mid_wf = audio_complete(mid.render(sf_path), params.N - 2)
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
##                   audio_complete(notes_target.render(sf_path),params.N-2).save(fn_base + '_target.flac')# -- tested, this works as intended
#                
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
            
#            onset_s = onset_detector.detect(audio_w, onset_gold)
#            duration_s = duration_detector.detect(audio_w, duration_gold)

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

            pitch_s = pitch_classifier.classify(audio_sw, pitch_gold)
            instrument_sw = instrument_classifier.classify(audio_sw, instrument_gold)
            pb.check_progress()
            print('')
            pitch_s = int(pitch_s)
            instrument_sw = int(instrument_sw)

            # subtract correct note for training:
            note_guessed = note_sequence()
            note_guessed.add_note(instrument_gold, instrument_gold, pitch_gold,
                                  0, duration_gold, velocity=100,
                                  is_drum=False)

            ac_note_guessed = audio_complete(note_guessed.render(sf_path), params.N - 2)
            if DEBUG:
#                print(onset_gold)
            #               ac_note_guessed.save(fn_base+'_guess.flac')
                note_last = note_gold
            audio_w.subtract(ac_note_guessed, offset=onset_gold)

            onset_s = onset_gold
            duration_s = duration_gold
#            instrument_sw = instrument_gold
#            pitch_s = pitch_gold
            sheet.add_note(instrument_sw, instrument_sw, pitch_s, onset_s + offset, onset_s + offset + duration_s)

        fn_result = os.path.join(path, 'results', os.path.split(fn[0])[-1])
        sheet.save(fn_result)
        audio_complete(sheet.render(sf_path), params.N - 2).save(fn_result + '.flac')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AMT-SAGA entry point.')
    parser.add_argument('-soundfont_path',nargs='?',
                        default=os.path.join('..','soundfonts','GM_soundfonts.sf2'),
                        help = 'The path to the soundfont file')
    parser.add_argument('-data_path',
                        default=os.path.join('.','data'),
                        help = 'The directory containing the files for training,\
                        testing etc.')
    args = vars(parser.parse_args())
    # Hyperparams
    path_data = args['data_path']
    path_sf = args['soundfont_path']
    p = Hyperparams(window_size_note_time=2)
    p.path = path_data

    pre_train(path_data, path_sf, p)

    training_session = True
