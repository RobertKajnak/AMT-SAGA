#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 11:13:26 2019

@author: hesiris
"""


import os
from multiprocessing import Process, Value, Queue, Lock, Pool
import logging

#from onsetdetector import OnsetDetector as OnsetDetector
#from durationdetector import DurationDetector as DurationDetector
from pitch_classifier import pitch_classifier as PitchClassifier
from instrumentclassifier import InstrumentClassifier as InstrumentClassifier
import util_dataset
from util_audio import note_sequence
from util_audio import audio_complete
from util_train_test import relevant_notes,note_sample,PATH_MODEL_META,PATH_NOTES

import ProgressBar as PB

try:
    logging.DETAILED
except:
    logging.DETAILED = 15
    logging.addLevelName(logging.DETAILED, "DETAILED")
    def detailed(self, message, *args, **kws):
        if self.isEnabledFor(logging.DETAILED):
            # Yes, logger takes its '*args' as 'args'.
            self._log(logging.DETAILED, message, args, **kws) 
    logging.Logger.detailed = detailed

def train_sequential(params):
    """ DEPRECATED
    Prepare data, training and test"""
    logger = logging.getLogger('AMT-SAGA.train_seq')

    dm = util_dataset.DataManager(params.path_data, sets=['training', 'test'], types=['midi'])
    
#    onset_detector = OnsetDetector(params)
#    onset_detector.plot_model(os.path.join(path_output,PATH_MODEL_META,'onset'+'.png'))
#    duration_detector = DurationDetector(params)
#    duration_detector.plot_model(os.path.join(path_output,PATH_MODEL_META,'duration'+'.png'))

    logger.info('Loading Pitch Classifier')
    pitch_classifier = PitchClassifier(params)
    pitch_classifier.plot_model(os.path.join(params.path_output,
                                             PATH_MODEL_META,'pitch'+'.png'))

    logger.info('Loading Instrument Classifier')
    instrument_classifier = InstrumentClassifier(params)
    instrument_classifier.plot_model(os.path.join(params.path_output
                                                  ,PATH_MODEL_META,
                                                  'instrument'+'.png'))
    frametime = params.H / params.sr
    halfwindow_frames = int(params.timing_input_shape/2)
    halfwindow_time = int(params.window_size_note_time/2)

    pb = PB.ProgressBar(300000)
    print('')
    dm.set_set('training')
    for fn in dm:
        mid = note_sequence(fn[0])
        sheet = note_sequence()
        
        note_i = 0
        offset = 0

        logger.info('Generating wav for midi {}'.format(fn))
        mid_wf = audio_complete(mid.render(params.sf_path), params.N - 2)
        # TODO - Remove drums - hopefully it can learn to ignore it though
        # TODO -  Add random instrument shifts
            
        audio_w = mid_wf.section(offset, None, params.timing_input_shape)
        notes_target, notes_w = relevant_notes(mid, offset, 
                                               params.window_size_note_time)
        while offset < mid.duration:
            note_gold = notes_target.pop(lowest=True, threshold=frametime)
            # training
            if note_gold is not None:
                logger.debug('Offset/Note start/end time = {:.3f} / {:.3f} / {:.3f}'.
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
            pitch_s = int(pitch_s)
            instrument_sw = int(instrument_sw)

            # subtract correct note for training:
            note_guessed = note_sequence()
            note_guessed.add_note(instrument_gold, instrument_gold, pitch_gold,
                                  0, duration_gold, velocity=100,
                                  is_drum=False)
            
            ac_note_guessed = audio_complete(note_guessed.render(params.sf_path), params.N - 2)

            if params.note_save_freq:
                note_i += 1
                if note_i % params.note_save_freq == 0:
                    fn_base = os.path.join(params.path_output, PATH_NOTES, 
                                           os.path.split(fn[0])[-1][:-4] + 
                                           str(note_i))
                    logger.detailed('Saving sample {} to {}'.format(note_i,fn_base))
                    
                    audio_w.save(fn_base+'_full_window.flac')
#                    audio_sw.save(fn_base+'_short_window.flac')
                    note_guessed.save(fn_base + '_guessed.mid')
                    ac_note_guessed.save(fn_base+'_guessed.flac')
                    
            audio_w.subtract(ac_note_guessed, offset=onset_gold)
    
            if (params.note_save_freq!=0) and \
                (note_i % params.note_save_freq == 0):
                audio_w.save(fn_base + '_after_subtr.flac')

            onset_s = onset_gold
            duration_s = duration_gold
#            instrument_sw = instrument_gold
#            pitch_s = pitch_gold
            sheet.add_note(instrument_sw, instrument_sw, pitch_s, onset_s + offset, onset_s + offset + duration_s)

        fn_result = os.path.join(params.path_output, 'results',
                                 os.path.split(fn[0])[-1])
        sheet.save(fn_result)
#        audio_complete(sheet.render(sf_path), params.N - 2).save(fn_result + '.flac')

def thread_classification(model_name,params,q_samples, training_finished, 
                          training_lock = None):
    i_b = 0
    training_lock and training_lock.acquire()
    if model_name == 'pitch':
        logger = logging.getLogger('AMT-SAGA.pitch_class')
        logger.info('Loading Pitch Classifier')
        model = PitchClassifier(params)
        model.plot_model(os.path.join(params.path_output,
                                      PATH_MODEL_META,'pitch'+'.png'))
    if model_name == 'instrument':
        logger = logging.getLogger('AMT-SAGA.instrument_class')
        logger.info('Loading Instrument Classifier')
        model = InstrumentClassifier(params)
        model.plot_model(os.path.join(params.path_output,
                                      PATH_MODEL_META,'instrument'+'.png'))
    training_lock and training_lock.release()
#    onset_detector = OnsetDetector(params)
#    onset_detector.plot_model(os.path.join(path_output,PATH_MODEL_META,'onset'+'.png'))
#    duration_detector = DurationDetector(params)
#    duration_detector.plot_model(os.path.join(path_output,PATH_MODEL_META,'duration'+'.png'))

    while training_finished.value == 0:
        sample = q_samples.get()
        training_lock and training_lock.acquire()
        
        logger.detailed('Starting {} Classification for Batch {}'.format(model_name,i_b))
        i_b+=1
#        model.classify(sample[0],sample[1])
        training_lock and training_lock.release()
    

def thread_training(samples_q, params,training_finished, 
                    allow_parallel_training=False):

    b_i=0
    logger = logging.getLogger('AMT-SAGA.training_master')
    
    #Technically pipes may be better, but the ease of use outweighs the 
    #performance penalty, especially compared to audio generation and training
    q_pitch = Queue(1)
    q_inst = Queue(1)
    if not allow_parallel_training:
        training_lock = Lock()
    else:
        training_lock = None
    
    proc_pitch = Process(target=thread_classification, 
                            args=('pitch',params, q_pitch,
                                  training_finished,
                                  training_lock))
    proc_inst = Process(target=thread_classification, 
                        args=('instrument',params, q_inst,
                              training_finished,
                              training_lock))
    proc_pitch.start()
    proc_inst.start()
    
    while training_finished.value == 0:
        pitch_x,pitch_y = [],[]
        instrument_x,instrument_y = [],[]
        for i in range(params.batch_size):
            sample = samples_q.get()
            pitch_x.append(sample.audio)
            pitch_y.append(sample.pitch)
            instrument_x.append(sample.audio)
            instrument_y.append(sample.instrument)

        logger.debug('Sending Batch {}'.format(b_i))
        b_i += 1
        
        q_pitch.put((pitch_x,pitch_y))
        q_inst.put((instrument_x,instrument_y))
    
    proc_pitch.join()
    proc_inst.join()
    
def init_sample_aquisition(samples_q,note_i):
    global samples_q_g
    samples_q_g = samples_q
    global note_i_g
    note_i_g = note_i

def thread_sample_aquisition(filename,params):
    fn = filename
    mid = note_sequence(fn[0])
    logger = logging.getLogger('AMT-SAGA.sample_gen')

    frametime = params.H / params.sr
    halfwindow_frames = int(params.timing_input_shape/2)
    halfwindow_time = int(params.window_size_note_time/2)

    offset = 0
    
    logger.info('Generating wav for midi {}'.format(fn))
    try:
        mid_wf = audio_complete(mid.render(), params.N - 2)
        # TODO - Remove drums - hopefully it can learn to ignore it though
        # TODO -  Add random instrument shifts
                
        audio_w = mid_wf.section(offset, None, params.timing_input_shape)
        notes_target, notes_w = relevant_notes(mid, offset, 
                                               params.window_size_note_time)
    except Exception as ex:
        logger.detailed('Could not process file {} with error {}. Skipping...'.
              format(fn,ex))
        return
        
    while offset < mid.duration:
        note_gold = notes_target.pop(lowest=True, threshold=frametime)
        # training
        if note_gold is not None:
            logger.debug('Offset/Note start/end time = {:.3f} / {:.3f} / {:.3f}'.
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
   
        samples_q_g.put(sample)
        
        # subtract correct note for training:
        note_guessed = note_sequence()
        note_guessed.add_note(instrument_gold, instrument_gold, pitch_gold,
                              0, duration_gold, velocity=100,
                              is_drum=False)
        
        ac_note_guessed = audio_complete(note_guessed.render(), params.N - 2)
        
        if params.note_save_freq:
            with note_i_g.get_lock():
                note_i_g.value += 1
                if note_i_g.value % params.note_save_freq == 0:
                    fn_base = os.path.join(params.path_output, PATH_NOTES, 
                                           os.path.split(fn[0])[-1][:-4] +
                                           '_' + str(note_i_g.value))
                    logger.detailed('Saving sample {} to {}'.
                                    format(note_i_g.value,fn_base))
                    
                    audio_w.save(fn_base+'_full_window.flac')
    #                audio_sw.save(fn_base+'_short_window.flac')
                    note_guessed.save(fn_base + '_guessed.mid')
                    ac_note_guessed.save(fn_base+'_guessed.flac')
                
                    audio_w.subtract(ac_note_guessed, offset=onset_gold)
                    audio_w.save(fn_base + '_after_subtr.flac')
        else:#Otherwise the lock will cause race conditions
            audio_w.subtract(ac_note_guessed, offset=onset_gold)
            
# noinspection PyShadowingNames
def train_parallel(params):
    """ Prepare data, training and test"""
    logger = logging.getLogger('AMT-SAGA.master')
    dm = util_dataset.DataManager(params.path_data, sets=['training', 'test'], types=['midi'])
        
    samples_q = Queue(params.batch_size)

    #Thread that does a training cycle each time q is full
    training_finished = Value('b',0)
    proc_training = Process(target=thread_training, args=(samples_q,params,
                                                          training_finished,
                                                          params.parallel_train))
    proc_training.start()

    #Pre-loading sounfont here means that threads don't use the memory separately
    #And other tests have shown that it is thread-safe
    note_sequence(sf2_path = params.sf_path) 
    dm.set_set('training') 
    note_index = Value('L',1)
    if params.synth_worker_count == 1:
        init_sample_aquisition(samples_q,note_index)
        for fn in dm:
            thread_sample_aquisition(fn, params)
    else:
        pool = Pool(processes = params.synth_worker_count,
                    initializer=init_sample_aquisition,
                    initargs=(samples_q,note_index))  
        
        results = [pool.apply_async(thread_sample_aquisition, 
                                   args=(fn, params)
                                   ) for fn in dm]
        for result in results:
            result.get()
    training_finished.value = 1 
    logger.info('Training finished!')
    proc_training.terminate()
    
    proc_training.join()