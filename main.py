# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 17:04:08 2019

@author: Hesiris

Dataset: https://colinraffel.com/projects/lmd/
"""

import argparse
import os
import logging

from util_train_test import Hyperparams,PATH_MODEL_META,PATH_NOTES,\
    PATH_CHECKPOINTS

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AMT-SAGA entry point. '
                                     'Press ctr+alt+q to stop the training '
                                     'process and create a save point '
                                     '(even when process is not in focus) ',
                                     formatter_class=\
                                     argparse.ArgumentDefaultsHelpFormatter)
    #Path and crucial files
    parser.add_argument('-soundfont_path',
                        nargs='?',
                        default=os.path.join('..',
                                             'soundfonts','GM_soundfonts.sf2'),
                        help = 'The path to the soundfont file')
    parser.add_argument('-data_path',
                        default=os.path.join('.','data'),
                        help = 'The directory containing the files for '
                        'training, testing etc.')
    parser.add_argument('-output_path',
                        default=os.path.join('.','output'),
                        help = 'Path to create log files, model structure ' 
                        'graphs, metrics etc.')
    
    #Model parameters
    parser.add_argument('-bins_per_semitone',
                        default=4,
                        type=int,
                        help = 'Amount of bins used per semitone when using '
                        'the CQ tranform. The overall size resulting will be '
                        'used for linear elements as well. The default size '
                        'is the minimum based on recent papers.')
    
    #Model parallelization
    parser.add_argument('-batch_size',
                        default=8,
                        type=int,
                        help = 'The batch size used in the training process.'
                        'Using a higher batch number on multi-core processors '
                        'may be useful.')
    parser.add_argument('-synth_workers',
                        default = 2,
                        type = int,
                        help = 'Number of threads that will be used for '
                        'midi->wav generation. This is most likely not the'
                        'bottleneck, a number of 2 is optimal for avoiding '
                        'the longer synthesis at the end of the song taking up'
                        ' too much time. In case there are audio glitches try '
                        'a worker count of 1.')
    parser.add_argument('-partrain',
                        action = 'store_true',
                        help = 'Specifying to allow traing models in parallel.'
                        ' This can cause instability or slower performance '
                        'than the sequential approach')
    
    #Checkpoint & debug
    parser.add_argument('-auto_continue',
                        action = 'store_true',
                        help = 'Attempt to continue from the last checkpoint '
                        'and metrics. The latest files will be attempted to '
                        'be loaded from the checkpoint directory '
                        '(output/checkpoints)')
    parser.add_argument('-checkpoint_freq',
                        default = 200,
                        type = int,
                        help = 'The freqency at which the model weights will '
                        'be saved. Measured in batches')
    parser.add_argument('-note_save_freq',
                        default = 500,
                        type = int,
                        help = 'The frequency at which the currently generated'
                        ' and subtracted notes will be displayed. This can be '
                        'useful to see if no synthetizer error has occured')
    #Logging/Messages
    parser.add_argument('-log_file_name',
                        default = 'log.log',
                        type = str,
                        help = 'Filename used for logging. A suffix will be '
                        'inserted before the last dot or if no extension is '
                        'provided, at the end of the name')
    parser.add_argument('-logging_level_console',
                        default = 2,
                        type = int,
                        help = 'The amount of messages that should be printed '
                        'to the console or out stream. Levels: 3 - exceptions '
                        'only; 2 - infrequent, e.g. ininitalization and '
                        'checkpointing, loading songs. 1 - detailed. '
                        'sending batches etc. 0 - debug. Everything, '
                        'including every note generation')
    parser.add_argument('-disable_file_logging',
                        action = 'store_true',
                        help = 'Nothing will be logged to files. '
                        'Model metrics are handled by the checkpoint_freq '
                        'parameter and are independent of this.')
    
    args = vars(parser.parse_args())
    # Command line args and hyperparms
    path_data = args['data_path']
    path_output = args['output_path']
    path_sf = args['soundfont_path']
    bins_per_tone = args['bins_per_semitone']
    synth_workers = args['synth_workers']
    par_train = args['partrain']
    batch_size = args['batch_size']
    autoload = args['auto_continue']
    check_freq = args['checkpoint_freq']
    note_save_freq = args['note_save_freq']
    log_fn = args['log_file_name']
    verbosity = args['logging_level_console']
    is_log_files = not args['disable_file_logging']
    
    #Configure Logging
    #Add custom level
    logging.DETAILED = 15
    logging.addLevelName(logging.DETAILED, 'DETAILED')
    def detailed(self, message, *args, **kws):
        if self.isEnabledFor(logging.DETAILED):
            # Yes, logger takes its '*args' as 'args'.
            self._log(logging.DETAILED, message, args, **kws) 
    logging.getLoggerClass().detailed = detailed
    
    
    logger = logging.getLogger('AMT-SAGA')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if verbosity==3:
        verbosity = logging.ERROR
    elif verbosity==2:
        verbosity = logging.INFO
    elif verbosity==1:
        verbosity = logging.DETAILED
    elif verbosity==0:
        verbosity = logging.DEBUG
    
    ch = logging.StreamHandler()
    ch.setLevel(verbosity)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    #Create debug and output folder
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    dirs = [PATH_MODEL_META,PATH_NOTES, PATH_CHECKPOINTS]
    for d in dirs:
        if not os.path.exists(os.path.join(path_output,d)):
            os.makedirs(os.path.join(path_output,d))
        
    #Attach files unless forbidden
    if is_log_files:    
        dot_pos = log_fn.rfind('.')
        if dot_pos == -1:
            fn_debug = log_fn + '_debug'
            fn_info = log_fn + '_info'
            fn_detailed = log_fn + '_detailed'
        else:
            fn_debug = log_fn[:dot_pos] + '_debug' + log_fn[dot_pos:]
            fn_info = log_fn[:dot_pos] + '_info' + log_fn[dot_pos:]
            fn_detailed = log_fn[:dot_pos] + '_detailed' + log_fn[dot_pos:]
        
        f_debug = logging.FileHandler(os.path.join(path_output,fn_debug))
        f_debug.setLevel(logging.DEBUG)
        f_debug.setFormatter(formatter)
        logger.addHandler(f_debug)
        
        f_info = logging.FileHandler(os.path.join(path_output,fn_info))
        f_info.setLevel(logging.INFO)
        f_info.setFormatter(formatter)
        logger.addHandler(f_info)
        
        f_detailed = logging.FileHandler(os.path.join(path_output,fn_detailed))
        f_detailed.setLevel(logging.DETAILED)
        f_detailed.setFormatter(formatter)
        logger.addHandler(f_detailed)
    
    #using the import here allows using the --help without loading the libraries
    try:            
        p = Hyperparams(path_data, path_sf, path_output = path_output,
                        
                        bins_per_tone = bins_per_tone,
                        
                        batch_size = batch_size,
                        parallel_train=par_train,
                        synth_worker_count=synth_workers,
                        
                        window_size_note_time=6,
                        
                        checkpoint_dir = os.path.join(path_output,PATH_CHECKPOINTS),
                        checkpoint_frequency = check_freq,
                        note_save_freq = note_save_freq,
                        autoload = autoload)
        args_str = '\n' + '\n'.join('{} = {}'.format(k, v) for k, v in args.items())
        logger.info('All hyperparemeters set, starting training: {}'.format(args_str))

        from training import train_parallel
        train_parallel(p)
        logger.info('Main Thread Terminated Gracefully')
    except Exception:
        logger.exception('Exception occured in thread handler')
        

