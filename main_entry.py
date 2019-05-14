# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 17:04:08 2019

@author: Hesiris
"""

import numpy as np
import os

from onset_detector import onset_detector as OnsetDetector
from pitch_classifier import pitch_classifier as PitchClassifier
from instrument_classifier import instrument_classifier as InstrumentClassifier
import util_dataset
from util_audio import note_sequence
from util_audio import audio_complete


class Hyperparams:
    def __init__(self, N=4096, sr=44100, H=None, window_size_note_time=None, window_th=None):
        self.N = N
        self.sr = sr
        self.H = np.int(N / 4) if H is None else H
        self.window_size_note_time = 3 if window_size_note_time is None else window_size_note_time  # in FS
        self.window_th = 0.2 if window_th is None else window_th
        self.pitch_input_shape = 20
        self.timing_input_shape = 215


def relevant_notes(sequence, audio, offset, duration, duration_in_frames):
    notes_w, notes_target = sequence.get_notes(offset, offset + duration)
    minus = notes_w.start_first

    notes_w = notes_w.clone()
    audio = audio.section(offset, None, duration_in_frames)
    print(offset, minus)

    notes_target = notes_target.clone()
    notes_target.shift(-offset)
    notes_w.shift(-offset + minus)
    return audio, notes_target, notes_w


# noinspection PyShadowingNames
def pre_train(path, sf_path, params):
    """ Prepare data, training and test"""
    dm = util_dataset.DataManager(path, sets=['training', 'test'], types=['midi'])
    onset_detector = OnsetDetector(params)
    pitch_classifier = PitchClassifier(params)
    instrument_classifier = InstrumentClassifier(params)
    frametime = params.H / params.sr

    dm.set_set('training')
    for fn in dm:
        mid = note_sequence(fn[0])

        sheet = note_sequence()

        DEBUG = 0
        if DEBUG:
            audio_w_last = None
        offset = 0

        mid_wf = audio_complete(mid.render(sf_path), params.N - 2)
        # TODO - Remove drums - hopefully it can learn to ignore it though
        # TODO -  Add random instrument shifts
        audio_w, notes_target, notes_w = relevant_notes(mid, mid_wf, offset, params.window_size_note_time,
                                                        params.timing_input_shape)
        while offset < mid.duration:

            if DEBUG:
                fn_base = os.path.join(path, 'debug', os.path.split(fn[0])[-1][:-4] + str(DEBUG))
                print(fn_base)
                #               if audio_w is not audio_w_last:
                #               audio_w.save(fn_base+'.flac')
                #               audio_complete(notes_target.render(sf_path),params.N-2).save(fn_base + '_target.flac')# -- tested, this works as intended

                #               audio_w_last = audio_w
                #               notes_target.save(fn_base+'_target.mid')
                #               notes_w.save(fn_base+'_inclusive.mid')
                DEBUG += 1

            note_gold = notes_target.pop(lowest=True, threshold=frametime)
            # training
            if note_gold is not None:
                print(offset, note_gold.start_time)
                onset_gold = note_gold.start_time
                duration_gold = note_gold.end_time - note_gold.start_time
                pitch_gold = note_gold.pitch
                instrument_gold = note_gold.program
            else:
                onset_gold = params.window_size_note_time + offset
            onset_s, duration_s = onset_detector.detect(audio_w, onset_gold, duration_gold)

            # use correct value to move window
            if onset_gold + duration_gold >= offset + params.window_size_note_time:
                # TODO: Fine tune window shift -- need to subtract the end of
                # the subtracted notes from subsequent windows, otherwise,
                # they sound like they start there. Or remove overlapping notes

                offset += params.window_size_note_time - params.window_th
                audio_w, notes_target, notes_w = relevant_notes(mid, mid_wf, offset, params.window_size_note_time,
                                                                params.timing_input_shape)
                continue

            audio_sw = audio_w.resize(onset_gold, duration_gold, params.pitch_input_shape)

            pitch_s = pitch_classifier.classify(audio_sw, pitch_gold)
            instrument_sw = instrument_classifier.classify(audio_sw, instrument_gold)

            # subtract correct note for training:
            note_guessed = note_sequence()
            note_guessed.add_note(instrument_gold, instrument_gold, pitch_gold,
                                  0, duration_gold, velocity=100,
                                  is_drum=False)

            ac_note_guessed = audio_complete(note_guessed.render(sf_path), params.N - 2)
            if DEBUG:
                print(onset_gold)
            #               ac_note_guessed.save(fn_base+'_guess.flac')
            #               note_last = note_gold
            audio_w.subtract(ac_note_guessed, offset=onset_gold)

            onset_s = onset_gold
            instrument_sw = instrument_gold
            pitch_s = pitch_gold
            duration_s = duration_gold
            sheet.add_note(instrument_sw, instrument_sw, pitch_s, onset_s + offset, onset_s + offset + duration_s)

        fn_result = os.path.join(path, 'results', os.path.split(fn[0])[-1])
        sheet.save(fn_result)
        audio_complete(sheet.render(sf_path), params.N - 2).save(fn_result + '.flac')


if __name__ == '__main__':
    # Hyperparams
    path = './data/'
    sf_path = '/home/hesiris/Documents/Thesis/GM_soundfonts.sf2'
    p = Hyperparams(window_size_note_time=5, window_th=1)
    p.path = path

    pre_train(path, sf_path, p)

    training_session = True
