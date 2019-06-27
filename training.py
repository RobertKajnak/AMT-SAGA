#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 11:13:26 2019

@author: hesiris
"""


import os
from multiprocessing import Process, Value, Queue, Lock, Pool
from queue import Empty as EmptyException
from queue import Full as FullException
import logging
from functools import wraps

try:
    from pynput.keyboard import KeyCode, Key, Listener
    X_AVAILABLE = True
except:
    X_AVAILABLE = False

#from onsetdetector import OnsetDetector as OnsetDetector
#from durationdetector import DurationDetector as DurationDetector
from pitch_classifier import pitch_classifier as PitchClassifier
from instrumentclassifier import InstrumentClassifier as InstrumentClassifier
from util_dataset import DataManager, get_latest_file
from util_audio import note_sequence
from util_audio import audio_complete
from util_train_test import relevant_notes,note_sample, PATH_MODEL_META,PATH_NOTES

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

def logged_thread(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        logger = logging.getLogger('AMT-SAGA.thread_logger')
        try:
            return func(*args,**kwargs)
        except Exception:
            logger.exception('Unexpected Exception occured in function {}'.
                             format(func.__name__))
            return None
        
    return wrapper

@logged_thread
def train_sequential(params):
    """ DEPRECATED
    Prepare data, training and test"""
    logger = logging.getLogger('AMT-SAGA.train_seq')

    dm = DataManager(params.path_data, sets=['training', 'test'], types=['midi'])
    
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
        mid_wf = audio_complete(mid.render(params.sf_path), params.N)
        mid_wf.mag # Evaluate mag to prime it for the NN. Efficiency trick
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
                #Otherwise the F would be calculated for both
                audio_w.slice(halfwindow_frames, 2*halfwindow_frames)
                audio_w.concat(audio_w_new)
                
                notes_target, notes_w = relevant_notes(mid, offset, 
                                                       params.window_size_note_time)
                continue

            audio_sw = audio_w.resize(onset_gold, duration_gold, 
                                      params.pitch_input_shape,
                                      attribs=['mag','ph'])
            C_sw = audio_w.slice_C(onset_gold, duration_gold, 
                                      params.pitch_input_shape)

            pitch_s = pitch_classifier.classify(C_sw, pitch_gold)
            instrument_sw = instrument_classifier.classify(audio_sw.mag, instrument_gold)

            pb.check_progress()
            pitch_s = int(pitch_s)
            instrument_sw = int(instrument_sw)

            # subtract correct note for training:
            note_guessed = note_sequence()
            note_guessed.add_note(instrument_gold, instrument_gold, pitch_gold,
                                  0, duration_gold, velocity=100,
                                  is_drum=False)
            
            ac_note_guessed = audio_complete(note_guessed.render(params.sf_path), params.N)

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
#        audio_complete(sheet.render(sf_path), params.N).save(fn_result + '.flac')

@logged_thread
def thread_classification(model_name,params,q_samples, training_finished, 
                          training_lock = None):
    i_b = 0
    training_lock and training_lock.acquire()
    
    if model_name == 'pitch':
        logger = logging.getLogger('AMT-SAGA.pitch_class')
        logger.info('Loading Pitch Classifier')
        prefix  = 'checkpoint_pitch'
        model = PitchClassifier(params, checkpoint_prefix= prefix)
    if model_name == 'instrument':
        logger = logging.getLogger('AMT-SAGA.instrument_class')
        logger.info('Loading Instrument Classifier')
        prefix = 'checkpoint_intrument'
        model = InstrumentClassifier(params, checkpoint_prefix= prefix)
        
    if model_name == 'instrument_focused':
        logger = logging.getLogger('AMT-SAGA.instrument_focused_class')
        logger.info('Loading Focused Instrument Classifier')
        prefix = 'checkpoint_intrument_focused'
        model = InstrumentClassifier(params, checkpoint_prefix= prefix)
    
    if params.autoload:
        fn_check = get_latest_file(params.checkpoint_dir, prefix)
        check_ind = int(fn_check[fn_check.rfind('_')+1:fn_check.rfind('.')])
        fn_metr = get_latest_file(params.checkpoint_dir,'metrics_logs')
        metr_ind = fn_metr[:fn_metr.rfind('_training')]
        metr_ind = int(metr_ind[metr_ind.rfind('_')+1:])
        if fn_metr is not None and fn_check is not None and \
            check_ind==metr_ind:
            model.load_weights(fn_check)
            model.load_metrics(params.checkpoint_dir, index=metr_ind,
                              training=True, test = False, use_csv=False,
                              load_metric_names = False)
            model.current_batch = check_ind
            i_b = check_ind
            logger.info('{} continuing from batch {}'.
                        format(model_name,check_ind))
        else:
            raise ValueError('Filenames missing of mismatched: {} {}'.
                             format(fn_check,fn_metr))
            
    model.plot_model(os.path.join(params.path_output,
                                  PATH_MODEL_META, model_name+'.png'))
    
    training_lock and training_lock.release()
#    onset_detector = OnsetDetector(params)
#    onset_detector.plot_model(os.path.join(path_output,PATH_MODEL_META,'onset'+'.png'))
#    duration_detector = DurationDetector(params)
#    duration_detector.plot_model(os.path.join(path_output,PATH_MODEL_META,'duration'+'.png'))

    while training_finished.value == 0:
        try:
            sample = q_samples.get(timeout=1)
        except EmptyException:
            continue
        except BrokenPipeError:
            logger.debug('Broken Pipe Detected. Assuming training was '
                         'terminated.')
            break
        training_lock and training_lock.acquire()
        
        logger.detailed('Starting {} Classification for Batch {}'.format(model_name,i_b))
        i_b+=1
        model.classify(sample[0],sample[1]) #!DEBUG
        training_lock and training_lock.release()
        
    if training_finished.value == 1:
        logger.info('Training finished, saving {}'.format(model_name))
    else:
        logger.info('Training terminated, saving {}'.format(model_name))
#    model.load_metrics(directory='/home/hesiris/Documents/Thesis/AMT-SAGA/output/checkpoints/',prefix='metrics_logs_', index = 3,
#                     training=True, test = False, use_csv=False, 
#                     load_metric_names=True) #DEBUG
#    model.current_batch=len(model.metrics_train)-1#DEBUG
    model.save_checkpoint()
    
@logged_thread
def init_sample_aquisition(samples_q,note_i,training_finished):
    global samples_q_g
    samples_q_g = samples_q
    global note_i_g
    note_i_g = note_i
    global training_finished_g
    training_finished_g = training_finished
    
@logged_thread
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
        mid_wf = audio_complete(mid.render(), params.N)
        mid_wf.mag # Evaluate mag to prime it for the NN. Efficiency trick.
        #Also prevents problems if the first section is complete silence
        
        # TODO -  Add random instrument shifts
                
        audio_w = mid_wf.section(offset, None, params.timing_input_shape)
        notes_target, notes_w = relevant_notes(mid, offset, 
                                               params.window_size_note_time)
    except Exception as ex:
        logger.detailed('Could not process file {} with error {}. Skipping...'.
              format(fn,ex))
        return
        
    while offset < mid.duration and training_finished_g.value==0:
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

        try:
            if onset_gold >= halfwindow_time:                
                offset += halfwindow_time
                
                audio_w_new = mid_wf.section(offset+halfwindow_time,
                                             None, halfwindow_frames)
                #Otherwise the F would be calculated for both
                audio_w.slice(halfwindow_frames, 2*halfwindow_frames)
                audio_w.concat(audio_w_new)
                
                notes_target, notes_w = relevant_notes(mid, offset, 
                                                       params.window_size_note_time)
                continue
    
            audio_sw = audio_w.resize(onset_gold, duration_gold, 
                                      params.pitch_input_shape,
                                      attribs=['mag','ph'])
            C_sw = audio_w.slice_C(onset_gold, duration_gold, 
                                      params.pitch_input_shape)
        except:
            logger.info('Faulty note in midi {}. Skipping file'.format(fn))
            return

        sample = note_sample(fn, audio_sw.mag, C_sw , pitch_gold, instrument_gold,
                             onset_gold, duration_gold)
                
        while training_finished_g.value==0:
            try:
                samples_q_g.put(sample,timeout=1)
            except FullException:
                if training_finished_g.value==0:
                    continue
                else:
                    return
            except BrokenPipeError:
                logger.debug('Broken Pipe Detected. Assuming training was '
                             'terminated.')
                return
            else:
                break
        
        # subtract correct note for training:
        note_guessed = note_sequence()
        note_guessed.add_note(instrument_gold, instrument_gold, pitch_gold,
                              0, duration_gold, velocity=100,
                              is_drum=False)
        
        ac_note_guessed = audio_complete(note_guessed.render(), params.N)
        
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
            
@logged_thread
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
        i=0
        while training_finished.value == 0 and i<params.batch_size :
            try:
                sample = samples_q.get(timeout=1)
            except EmptyException:
                if training_finished.value==0:
                    continue
            except BrokenPipeError:
                logger.debug('Broken Pipe Detected. Assuming training was '
                             'terminated.')
                break
            except Exception:
                logger.exception('Unexpected error while reading from pipe, '
                                 'attempting to continue')
                continue
            try:
                pitch_x.append(sample.audio_sw_C)
                pitch_y.append(sample.pitch)
                instrument_x.append(sample.audio_sf_F)
                instrument_y.append(sample.instrument)
                i+=1
            except Exception:
                logger.exception('Unexpected error while sending samples.'
                                 'Attempting to continue')
                continue
            
        if training_finished.value == 0:
            try:
                logger.debug('Sending Batch {}'.format(b_i))
                b_i += 1
                q_pitch.put((pitch_x,pitch_y))
                q_inst.put((instrument_x,instrument_y))
            except BrokenPipeError:
                logger.debug('Broken Pipe Detected. Assuming training was '
                             'terminated.')
                break
        
    proc_pitch.join()
    proc_inst.join()

# noinspection PyShadowingNames
def train_parallel(params):
    """ Prepare data, training and test"""
    logger = logging.getLogger('AMT-SAGA.master')
    dm = DataManager(params.path_data, sets=['training', 'test'], types=['midi'])
        
    samples_q = Queue(params.batch_size)

    #Thread that does a training cycle each time q is full
    training_finished = Value('b',0)
    proc_training = Process(target=thread_training, args=(samples_q,params,
                                                          training_finished,
                                                          params.parallel_train))
    proc_training.start()
    
    if X_AVAILABLE:
        held_down = set()
        key_q = KeyCode.from_char('q')
        key_Q = KeyCode.from_char('Q')
        def on_press(key):
            held_down.add(key)
            if (key==key_q or key==key_Q) \
                    and Key.ctrl in held_down and Key.alt in held_down:
                training_finished.value=2
            
        def on_release(key):
            try:
                held_down.remove(key)
            except KeyError:
                pass
        listener = Listener(
                on_press=on_press,
                on_release=on_release)
        listener.start()
    else:
        logger.info('X not available. Stopping hotkey not available.')
        
    #Pre-loading sounfont here means that threads don't use the memory separately
    #And other tests have shown that it is thread-safe
    note_sequence(sf2_path = params.sf_path) 
    dm.set_set('training') 
    note_index = Value('L',1)
    if params.synth_worker_count == 1:
        init_sample_aquisition(samples_q,note_index,training_finished)
        for fn in dm:
            thread_sample_aquisition(fn, params)
    else:
        pool = Pool(processes = params.synth_worker_count,
                    initializer=init_sample_aquisition,
                    initargs=(samples_q,note_index,training_finished))  
        
        results = [pool.apply_async(thread_sample_aquisition, 
                                   args=(fn, params)
                                   ) for fn in dm]
        for result in results:
            result.get()
        pool.close()
        pool.join()
    training_finished.value = 1 
    proc_training.join()
    if X_AVAILABLE:
        listener.stop()
    
    logger.info('Training finished!')
#    proc_training.terminate()
    