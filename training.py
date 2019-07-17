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

def gen_model(model_name,params):
    if model_name == 'pitch':
        logger = logging.getLogger('AMT-SAGA.pitch_class')
        logger.info('Loading Pitch Classifier')
        prefix  = 'checkpoint_pitch'
        model = PitchClassifier(params, checkpoint_prefix= prefix)
    if model_name == 'instrument':
        logger = logging.getLogger('AMT-SAGA.instrument_class')
        logger.info('Loading Instrument Classifier')
        model = InstrumentClassifier(params, 
                                     variant=InstrumentClassifier.INSTRUMENT)
    if model_name == 'instrument_focused':
        logger = logging.getLogger('AMT-SAGA.instrument_focused_class')
        logger.info('Loading Focused Instrument Classifier')
        model = InstrumentClassifier(
                params, variant= InstrumentClassifier.INSTRUMENT_FOCUESD)
    if model_name == 'instrument_dual':
        logger = logging.getLogger('AMT-SAGA.instrument_dual_class')
        logger.info('Loading Focused Instrument Classifier')
        model = InstrumentClassifier(
                params, variant= InstrumentClassifier.INSTRUMENT_DUAL)
        
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
            batch_index = check_ind
            logger.info('{} continuing from batch {}'.
                        format(model_name,check_ind))
        else:
            raise ValueError('Filenames missing of mismatched: {} {}'.
                             format(fn_check,fn_metr))
    else:
        batch_index = 0
            
    model.plot_model(os.path.join(params.path_output,
                                  PATH_MODEL_META, model_name+'.png'))
    
#    onset_detector = OnsetDetector(params)
#    onset_detector.plot_model(os.path.join(path_output,PATH_MODEL_META,'onset'+'.png'))
#    duration_detector = DurationDetector(params)
#    duration_detector.plot_model(os.path.join(path_output,PATH_MODEL_META,'duration'+'.png'))
    return model, logger, batch_index

@logged_thread
def thread_classification(model_name,params,q_samples, training_finished, 
                          training_lock = None):
    training_lock and training_lock.acquire()    
    model, logger, i_b = gen_model(model_name, params)
    training_lock and training_lock.release()

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
        try:
            model.classify(sample[0],sample[1]) #!DEBUG
        except:#TODO: Try
            logger.exception('An Error occured while processing the sample. Skipping')
#        print(sample[0][0].shape,sample[1][0])#DEBUG
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
    logger = logging.getLogger('AMT-SAGA.sample_gen')
    fn = filename
    try:
        mid = note_sequence(fn[0])
    except:
        logger.info('File {} Corrupt. Skipping'.format(fn))
        return

    frametime = params.H / params.sr
    print(frametime)
    halfwindow_frames = int(params.timing_frames/2)
    halfwindow_time = int(params.window_size_note_time/2)

    offset = 0
    
    logger.info('Generating wav for midi {}'.format(fn))
    try:
        mid_wf = audio_complete(mid.render(), params.N)
        mid_wf.mag # Evaluate mag to prime it for the NN. Efficiency trick.
        #Also prevents problems if the first section is complete silence
        
        audio_w = mid_wf.section(offset, None, params.timing_frames)
        notes_target, _ = relevant_notes(mid, offset, 
                                               params.window_size_note_time)
    except Exception as ex:
        logger.detailed('Could not process file {} with error {}. Skipping...'.
              format(fn,ex))
        return
    #TODO only process an n notes long sectionf rom the song
    while offset < mid.duration and training_finished_g.value==0:
        note_gold = notes_target.pop(lowest=True, threshold=frametime)
        if note_gold is not None: 
            if note_gold.program>=params.instrument_classes:
                logger.debug('Program out of range: {}'.format(note_gold.program))
                continue
            elif note_gold.pitch<params.pitch_low or note_gold.pitch>params.pitch_high:
                logger.debug('Pitch out of range: {}'.note_gold.pitch)
                continue
            elif note_gold.is_drum:
                continue#TODO: test this
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
                
                notes_target, _ = relevant_notes(mid, offset, 
                                                       params.window_size_note_time)
                #TODO:check if this should be halfwindow to make the window not include the first note after the offset
                continue
    
#            audio_sw = audio_w.resize(onset_gold, duration_gold, 
#                                      params.pitch_frames,
#                                      attribs=['mag','ph'])
            C_sw_pitch = audio_w.slice_C(onset_gold, duration_gold, 
                                      params.pitch_frames,
                                      bins_per_note=1)
            C_sw_inst = audio_w.slice_C(onset_gold, duration_gold,
                                        params.instrument_frames,
                                        bins_per_note=params.instrument_bins_per_tone)
        except:
            logger.info('Faulty note in midi {}. Skipping file'.format(fn))
            return

        sample = note_sample(fn, None, C_sw_pitch, C_sw_inst, pitch_gold, instrument_gold,
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
    if not allow_parallel_training:
        training_lock = Lock()
    else:
        training_lock = None
    
    model_names = ['pitch','instrument']#, 'instrument_focused', 'instrument_dual']
    qs = [Queue(1) for _ in model_names]
    model_processes = [Process(target=thread_classification, 
                            args=(model_name,params, q,
                                  training_finished,
                                  training_lock))
                        for model_name,q in zip(model_names,qs)]
    for proc in model_processes:
        proc.start()
    
    while training_finished.value == 0:
        sample_x,sample_y = [],[]
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
                sample_x.append((sample.audio_sw_C_pitch,sample.audio_sw_C_inst))
                sample_y.append((sample.pitch,sample.instrument))
                i+=1
            except Exception:
                logger.exception('Unexpected error while sending samples.'
                                 'Attempting to continue')
                continue
            
        if training_finished.value == 0:
            try:
                logger.debug('Sending Batch {}'.format(b_i))
                b_i += 1
                sample_x = list(map(list, zip(*sample_x)))
                sample_y = list(map(list, zip(*sample_y)))
                for x,y,q in zip(sample_x,sample_y,qs):
                    q.put((x,y))
            except BrokenPipeError:
                logger.debug('Broken Pipe Detected. Assuming training was '
                             'terminated.')
                break
      
    for proc in model_processes:
        proc.join()
        
def attach_keyboard_abort(training_finished):
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
    return listener
    
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
        listener = attach_keyboard_abort(training_finished)
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
    