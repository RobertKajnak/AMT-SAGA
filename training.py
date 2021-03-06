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
import warnings
from librosa import midi_to_note
import numpy as np
try:
    from pynput.keyboard import KeyCode, Key, Listener
    X_AVAILABLE = True
except:
    X_AVAILABLE = False

from timing_classifier import timming_classifier as TimingClassifier
from pitch_classifier import pitch_classifier as PitchClassifier
from instrumentclassifier import InstrumentClassifier as InstrumentClassifier
from velocity_classifier import VelocityClassifier as VelocityClassifier
from util_dataset import DataManager, get_latest_file
from util_audio import note_sequence
from util_audio import audio_complete
from util_train_test import relevant_notes, note_sample, validate_note, \
    PATH_MODEL_META,PATH_NOTES,sumrize

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
    logger = logging.getLogger('AMT-SAGA.' + model_name + '_class')
    checkpoint_prefix = 'checkpoint_' + model_name
    
    if model_name == 'timing_start':
        logger.info('Loading Timing Start Detector ({})'.format(model_name))
        model = TimingClassifier(params, checkpoint_prefix= checkpoint_prefix,
                                 metrics_prefix = 'metrics_timing_start')
    if model_name == 'timing_end':
        logger.info('Loading Timing End Detector ({})'.format(model_name))
        model = TimingClassifier(params, checkpoint_prefix= checkpoint_prefix,
                                 metrics_prefix = 'metrics_timing_end')
        
    if model_name == 'pitch':
        logger.info('Loading Pitch Classifier ({})'.format(model_name))
        model = PitchClassifier(params, checkpoint_prefix= checkpoint_prefix)
        
    if model_name == 'instrument':
        logger.info('Loading Instrument Classifier ({})'.format(model_name))
        model = InstrumentClassifier(params, 
                                     variant=InstrumentClassifier.INSTRUMENT)
    if model_name == 'instrument_focused' or model_name == 'instrument_focused_lin' \
        or model_name == 'instrument_focused_lin_log10':
        logger.info('Loading Focused Instrument Classifier ({})'.format(model_name))
        model = InstrumentClassifier(
                params, variant= InstrumentClassifier.INSTRUMENT_FOCUSED, 
                prefix = model_name)
    if model_name == 'instrument_focused_const' or model_name == 'instrument_focused_const_lin' \
        or model_name == 'instrument_focused_const_lin_log10':
        logger.info('Loading Focused Instrument Classifier with fixed window '
                    'position  ({})'.format(model_name))
        model = InstrumentClassifier(
                params, variant= InstrumentClassifier.INSTRUMENT_FOCUSED_CONST, 
                prefix = model_name)
    if model_name == 'instrument_dual' or model_name == 'instrument_dual_lin' \
        or model_name == 'instrument_dual_lin_phase':
        logger.info('Loading Focused Instrument Classifier  ({})'.format(model_name))
        model = InstrumentClassifier(
                params, variant= InstrumentClassifier.INSTRUMENT_DUAL, 
                prefix = model_name)
        
    if model_name == 'velocity':
        logger.info('Loading Veloicty Classifier')
        model = VelocityClassifier(params,checkpoint_prefix= checkpoint_prefix)
        
    if params.autoload:
        fn_check = get_latest_file(params.checkpoint_dir, 
                                   [model.checkpoint_prefix,'.h5'])
        check_ind = int(fn_check[fn_check.rfind('_')+1:fn_check.rfind('.')])
        if fn_check is None:
            raise ValueError('No checkppint fond in the checkpoint directory')
        fn_metr = get_latest_file(params.checkpoint_dir, 
                                  [model.metrics_prefix,'training','.npz'])
        if fn_metr is None:
            logger.error('No metric found in checkpoint directory. '
                         'Continuing without metrics')
            model.load_weights(fn_check)
            batch_index = 0
        else:            
            metr_ind = fn_metr[:fn_metr.rfind('_training')]
            metr_ind = int(metr_ind[metr_ind.rfind('_')+1:])
            
        if fn_metr is not None and fn_check is not None and \
            check_ind==metr_ind:
            try:
                model.load_metrics(params.checkpoint_dir, index=metr_ind,
                                  training=True, test = True, use_csv=False,
                                  load_metric_names = False)
            except:
                logger.info('No test data found. Loading only training data')
                model.load_metrics(params.checkpoint_dir, index=metr_ind,
                                  training=True, test = False, use_csv=False,
                                  load_metric_names = False)
            model.current_batch = check_ind
            batch_index = check_ind
            logger.info('{} continuing from batch {}'.
                        format(model_name,check_ind))
        else:
            logger.error('Filenames missing of mismatched: {} {}. Continuing '
                         'without metrics'.format(fn_check,fn_metr))
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
def thread_classification(model_name,params,q_samples, training_state, 
                          training_lock = None):
    training_lock and training_lock.acquire()    
    model, logger, i_b = gen_model(model_name, params)
    training_lock and training_lock.release()

    while training_state.value > 0:
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
            if training_state.value == 1:
                model.classify(sample[0],sample[1]) #!DEBUG
            elif training_state.value == 2:
                model.classify(sample[0],sample[1],test_phase=True)
            elif training_state.value == 3:
                model.classify(sample[0],None)
        except:
            logger.exception('An Error occured while processing the sample. Skipping')
#        print(sample[0][0].shape,sample[1][0])#DEBUG
        training_lock and training_lock.release()
        
    if training_state.value == 0:
        logger.info('Training finished, saving {}'.format(model_name))
    else:
        logger.info('Training terminated, saving {}'.format(model_name))
#    model.load_metrics(directory='/home/hesiris/Documents/Thesis/AMT-SAGA/output/checkpoints/',prefix='metrics_logs_', index = 3,
#                     training=True, test = False, use_csv=False, 
#                     load_metric_names=True) #DEBUG
#    model.current_batch=len(model.metrics_train)-1#DEBUG
    try:
        logger.info('Saving last checkpoint')
        model.save_checkpoint()
    except:
        logger.exception('Error occured during saving the checkpoint')
        
    try:
        model.report(filename_test = os.path.join(params.checkpoint_dir,
                                                  'test' + model.metrics_prefix
                                                  + '.txt'))
    except Exception:
        logger.exception('Failed to generate report.')
    try:
        if 'pitch' in model_name or 'timing' in model_name:
            m2p = [0,1,2,3,4,5,6]
        else:
            m2p = [0,1,2]
        model.plot(metrics_to_plot= m2p,moving_average_window=1,
             filename_training = os.path.join(params.checkpoint_dir,
                                              'training_' + model.metrics_prefix
                                              + '.png'),
             filename_test = os.path.join(params.checkpoint_dir,
                                          'test_' + model.metrics_prefix + 
                                          '.png'))
             
        model.plot(metrics_to_plot= m2p,moving_average_window=50,
             filename_training = os.path.join(params.checkpoint_dir,
                                              'training_' + model.metrics_prefix
                                              + '_ws50.png'),
             filename_test = os.path.join(params.checkpoint_dir,
                                          'test_' + model.metrics_prefix + 
                                          '_ws50.png'))
    except Exception:
        logger.exception('Failed to plot model:.')
    logger.info('Checkpoints saved, concluding model thread')
            
@logged_thread
def init_sample_aquisition(samples_q,note_i,training_state):
    global samples_q_g
    samples_q_g = samples_q
    global note_i_g
    note_i_g = note_i
    global training_state_g
    training_state_g = training_state
    
@logged_thread
def thread_sample_aquisition(filename,params):
#    warnings.filterwarnings("error")
    logger = logging.getLogger('AMT-SAGA.sample_gen')
    fn = filename
    try:
        mid = note_sequence(fn[0])
    except:
        logger.info('File {} Corrupt. Skipping'.format(fn))
        return

    frametime = params.H / params.sr
    halfwindow_frames = int(params.timing_frames/2)
    halfwindow_time = int(params.window_size_note_time/2)
    
    offset = 0
    
    total_note_estimation = len(mid.sequence.notes)
    if params.use_precise_note_count:
        s=0
        for note in mid.sequence.notes:
           if not validate_note(note,params):
               s+=1        
        logger.info('Generating wav for midi {} Notes to be processed: {}'
                .format(fn,total_note_estimation))
    else:
        logger.info('Generating wav for midi {} Estimated number of notes: {}'
                .format(fn,total_note_estimation))
    try:
        mid_wf = audio_complete(mid.render(), params.N)
        if mid_wf.spectral_flatness()>0.3:
            logger.info('Could not generate sound for {}. Skipping'.format(fn))
            return
        mid_wf.mag # Evaluate mag to prime it for the NN. Efficiency trick.
        #Also prevents problems if the first section is complete silence
        ref_C_1 = np.max(mid_wf.slice_C(0, mid_wf._frames_to_seconds(mid_wf.shape[1]),
                                        mid_wf.shape[1],
                                        params.pitch_frames,
                                        bins_per_tone=1))
        ref_C_inst = np.max(mid_wf.slice_C(0, mid_wf._frames_to_seconds(mid_wf.shape[1]),
                                           mid_wf.shape[1],
                                           params.pitch_frames,
                                           bins_per_tone=params.instrument_bins_per_tone))
        ref_C_foc = np.max(mid_wf.slice_C(0, mid_wf._frames_to_seconds(mid_wf.shape[1]),
                                          mid_wf.shape[1],
                                          params.pitch_frames,
                                          bins_per_tone=params.instrument_bins_per_tone*4))
        
        audio_w = mid_wf.section(offset, None, params.timing_frames)
        notes_target, _ = relevant_notes(mid, offset, 
                                               params.window_size_note_time)
        
        #these constant needed the audio_complete object to calculate
        fft_bin_min_const = mid_wf.midi_tone_to_FFT(60) #C4, kinda in the middle
        fft_bin_max_const = fft_bin_min_const + params.instrument_bands
    except Exception as ex:
        logger.detailed('Could not process file {} with error {}. Skipping...'.
              format(fn,ex))
        return
    
    while offset < mid.duration and training_state_g.value>0:
        note_gold = notes_target.pop(lowest=True, threshold=frametime)
        
        if note_gold is not None:
            note_problem = validate_note(note_gold,params)
            if note_problem:
                logger.debug(note_problem)
                continue
            
            logger.debug('Offset/Note start/end time = {:.3f} / {:.3f} / {:.3f}'.
                  format(offset, note_gold.start_time,note_gold.end_time))
        
            onset_gold = note_gold.start_time
            duration_gold = note_gold.end_time - note_gold.start_time
            pitch_gold = note_gold.pitch
            instrument_gold = note_gold.program
            velocity_gold = note_gold.velocity
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
                
                continue
#            C_timing = audio_w.slice_C(0,params.window_size_note_time,
#                                       params.timing_frames,
#                                       bins_per_note=1)
                
            C_timing = audio_complete.compress_bands(audio_w.mag,
                                        bands = params.timing_bands)
            C_timing = audio_complete._resize(C_timing,params.timing_frames)\
                                                /mid_wf.ref_mag
            audio_sw = audio_w.resize(onset_gold, duration_gold, 
                                      params.pitch_frames,
                                      attribs=['mag','ph'])
            C_sw_pitch = audio_w.slice_C(onset_gold, duration_gold, 
                                      params.pitch_frames,
                                      bins_per_tone=params.pitch_bins_per_tone)/ref_C_1
            C_sw_inst = audio_w.slice_C(onset_gold, duration_gold,
                                        params.instrument_frames,
                                        bins_per_tone=params.instrument_bins_per_tone)\
                                        /ref_C_inst
            F_sw_inst_foc_const = audio_sw.section_power('mag',
                                                  fft_bin_min_const,fft_bin_max_const)
            
            F_sw_inst_foc_const_log10 = np.log10(F_sw_inst_foc_const*1000 + 1)
            F_sw_inst_foc_const_log10 /= np.max(F_sw_inst_foc_const_log10)
            F_sw_inst_foc_const/=mid_wf.ref_mag
                                                         
            fft_bin_min = audio_w.midi_tone_to_FFT(pitch_gold)
            fft_bin_max = fft_bin_min + params.instrument_bands
            F_sw_inst_foc = audio_sw.section_power('mag',
                                                  fft_bin_min,fft_bin_max)
                                                  
            F_sw_inst_foc_log10 = np.log10(F_sw_inst_foc*1000+1)
            F_sw_inst_foc_log10 /= np.max(F_sw_inst_foc_log10)
            F_sw_inst_foc/=mid_wf.ref_mag
            ph = audio_sw.section_power('ph', fft_bin_min,fft_bin_max)
            ph = (np.angle(ph)+3.15)/6.3
            
            C_sw_inst_foc = audio_w.slice_C(onset_gold, duration_gold,
                                        params.instrument_frames,
                                        bins_per_tone=params.instrument_bins_per_tone*4,
                                        highest_note = None,
                                        nbins = params.instrument_bands,
                                        lowest_note= midi_to_note(pitch_gold))\
                                        /ref_C_foc
            
            C_sw_inst_foc_const = audio_w.slice_C(onset_gold, duration_gold,
                                        params.instrument_frames,
                                        bins_per_tone=params.instrument_bins_per_tone*4,
                                        highest_note = None, 
                                        nbins = params.instrument_bands,
                                        lowest_note= midi_to_note(60)
                                        )\
                                        /ref_C_foc
                                        
            C_velocity = audio_w.slice_C(onset_gold, duration_gold,
                                        params.instrument_frames,
                                        bins_per_tone=2,
                                        highest_note = None,
                                        nbins = params.bins_velocity,
                                        lowest_note= midi_to_note(pitch_gold-10))\
                                        /ref_C_foc
        except:
            logger.exception('Faulty note in midi {}. Skipping file'.format(fn))
            return

        sample = note_sample(fn,
                             C_timing,
                             C_sw_pitch, C_sw_inst, 
                             F_sw_inst_foc,F_sw_inst_foc_log10,
                             F_sw_inst_foc_const, F_sw_inst_foc_const_log10, 
                             C_sw_inst_foc, C_sw_inst_foc_const,
                             C_velocity,
                             ph,
                             pitch_gold, instrument_gold,
                             onset_gold, onset_gold + duration_gold,
                             velocity_gold)

        while training_state_g.value>0:
            try:
                samples_q_g.put(sample,timeout=1)
            except FullException:
                if training_state_g.value>0:
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
                              0, duration_gold, velocity=velocity_gold,
                              is_drum=False)
        
        ac_note_guessed = audio_complete(note_guessed.render(),params.N)
        
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
                    audio_sw.save(fn_base+'_short_window.flac')
                    note_guessed.save(fn_base + '_guessed.mid')
                    ac_note_guessed.save(fn_base+'_guessed.flac')
                    #This way the subtractoin is done twice, however the 
                    #subtraction on the other threads will not be locked,
                    #the lock expires after checking the note_i_g.value
                    audio_subd = audio_w.clone()
                    audio_subd.subtract(ac_note_guessed, offset=onset_gold)
                    audio_subd.save(fn_base + '_after_subtr.flac')
                    
        audio_w.subtract(ac_note_guessed, offset=onset_gold)
    logger.info('Finished processing: {}'.format(fn))
    
@logged_thread
def thread_training(samples_q, params,training_state, 
                    allow_parallel_training=False):

    b_i=0
    logger = logging.getLogger('AMT-SAGA.training_master')
    
    #Technically pipes may be better, but the ease of use outweighs the 
    #performance penalty, especially compared to audio generation and training
    if not allow_parallel_training:
        training_lock = Lock()
    else:
        training_lock = None
    
    all_model_names = ['timing_start', 'timing_end',
                    'pitch',
                    'instrument',
                    'instrument_focused_lin', 
                    'instrument_focused_lin_log10', 
                    'instrument_focused_const_lin', 
                    'instrument_focused_const_lin_log10',
                    'instrument_dual_lin',
                    'instrument_dual_lin_phase',
                    'instrument_focused', 'instrument_focused_const',
                    'instrument_dual',
                    'velocity']
    
    #Selects the models to use from the possiblities
    model_names = [all_model_names[i] for i in params.models_to_train]
    
    qs = [Queue(1) for _ in model_names]
    model_processes = [Process(target=thread_classification, 
                            args=(model_name,params, q,
                                  training_state,
                                  training_lock))
                        for model_name,q in zip(model_names,qs)]
    for proc in model_processes:
        proc.start()
    
    while training_state.value > 0:
        sample_x,sample_y = [],[]
        i=0
        while training_state.value > 0 and i<params.batch_size :
            try:
                sample = samples_q.get(timeout=1)
            except EmptyException:
                if training_state.value>0:
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
                all_x = (sample.C_timing,
                         sample.C_timing,
                         sample.C_sw_pitch,
                         sample.C_sw_inst,
                         sample.F_sw_inst_foc,
                         sample.F_sw_inst_foc_log10,
                         sample.F_sw_inst_foc_const,
                         sample.F_sw_inst_foc_const_log10,
                         [sample.C_sw_inst,sample.F_sw_inst_foc],
                         [sample.F_sw_inst_foc,sample.ph],
                         sample.C_sw_inst_foc,
                         sample.C_sw_inst_foc_const,
                         [sample.C_sw_inst,sample.C_sw_inst_foc],
                         sample.C_velocity)
                all_y = (sample.time_start,
                         sample.time_end,
                         sample.pitch,
                         sample.instrument,
                         sample.instrument,
                         sample.instrument,
                         sample.instrument,
                         sample.instrument,
                         sample.instrument,
                         sample.instrument,
                         sample.instrument,
                         sample.instrument,
                         sample.instrument,
                         sample.velocity)
                current_x = [all_x[mi] for mi in params.models_to_train]
                current_y = [all_y[mi] for mi in params.models_to_train]
                
                sample_x.append(current_x)
                sample_y.append(current_y)
                i+=1
            except Exception:
                logger.exception('Unexpected error while sending samples.'
                                 'Attempting to continue')
                continue
            
        if training_state.value > 0:
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
      
    logger.info('Training Thread finished, waiting for models to finish')
    for proc in model_processes:
        proc.join()
    logger.info('Models terminated')
        
def attach_keyboard_abort(training_state):
    held_down = set()
    key_q = KeyCode.from_char('q')
    key_Q = KeyCode.from_char('Q')
    def on_press(key):
        held_down.add(key)
        if (key==key_q or key==key_Q) \
                and Key.ctrl in held_down and Key.alt in held_down:
            training_state.value=-2
        
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
    training_state = Value('b',1)
    proc_training = Process(target=thread_training, args=(samples_q,params,
                                                          training_state,
                                                          params.parallel_train))
    proc_training.start()
    
    if X_AVAILABLE:
        listener = attach_keyboard_abort(training_state)
    else:
        logger.info('X not available. Stopping hotkey not available.')
        
    #Pre-loading sounfont here means that threads don't use the memory separately
    #And other tests have shown that it is thread-safe
    note_sequence(sf2_path = params.sf_path) 
    note_index = Value('L',1)
    if params.synth_worker_count == 1:
        init_sample_aquisition(samples_q,note_index,training_state)
        dm.set_set('training')
        for fn in dm:
            thread_sample_aquisition(fn, params)
        #Empty any saples left over from the training
        while not samples_q.empty():
            samples_q.get()
        dm.set_set('test')
        for fn in dm:
            thread_sample_aquisition(fn, params)
    else:
        pool = Pool(processes = params.synth_worker_count,
                    initializer=init_sample_aquisition,
                    initargs=(samples_q,note_index,training_state))  
        
        dm.set_set('training')
        results = [pool.apply_async(thread_sample_aquisition, 
                                   args=(fn, params)
                                   ) for fn in dm]
        
        #Join the threads without closing the pools
        for result in results:
            result.get()
        #Empty any saples left over from the training
        while not samples_q.empty():
            samples_q.get()
        #Switch to test set
        logger.info('Switching to test phase.')
        dm.set_set('test')
        training_state.value = 2
        results += [pool.apply_async(thread_sample_aquisition, 
                                   args=(fn, params)
                                   ) for fn in dm]
        for result in results:
            result.get()
        logger.info('Training phase finished. Wrapping up')
        pool.close()
        pool.join()
        
    training_state.value = 0
    proc_training.join()
    if X_AVAILABLE:
        listener.stop()
    
    logger.info('Training finished!')
#    proc_training.terminate()
    