# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 17:04:08 2019

@author: Hesiris

Dataset: https://colinraffel.com/projects/lmd/
"""

import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AMT-SAGA entry point.')
    parser.add_argument('-soundfont_path',nargs='?',
                        default=os.path.join('..',
                                             'soundfonts','GM_soundfonts.sf2'),
                        help = 'The path to the soundfont file')
    parser.add_argument('-data_path',
                        default=os.path.join('.','data'),
                        help = 'The directory containing the files for '
                        'training, testing etc.')
    parser.add_argument('-sequential',
                        action = 'store_true',
                        help = 'Specifying this will force the algorithm to '
                        'not use any multithreading or multiprocessing '
                        'methods. Batch size will be set to 1.')
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
                        ' This can cause instability or slowerperformance than'
                        ' the sequential approach')
    parser.add_argument('-silent',
                        action='store_true',
                        help = 'Do not display progress messages')
    args = vars(parser.parse_args())
    # Command line args and hyperparms
    path_data = args['data_path']
    path_sf = args['soundfont_path']
    synth_workers = args['synth_workers']
    par_train = args['partrain']
    batch_size = args['batch_size']
    silent = args['silent']
    sequential = args['sequential']
    #mp.set_start_method('spawn')
    
    #using the import here allows using the --help without loading the libraries
    from util_train_test import Hyperparams
    if sequential:
        from training import train_sequential
        p = Hyperparams(path_data, path_sf, 
                    window_size_note_time=6,
                    batch_size = 1,
                    parallel_train= False,
                    synth_worker_count= 1)

        train_sequential(p, not silent)
    else:
        from training import train_parallel
        p = Hyperparams(path_data, path_sf, batch_size = batch_size,
                        parallel_train=par_train,
                        synth_worker_count=synth_workers,
                        window_size_note_time=6)
    
        train_parallel(p, not silent)
        

