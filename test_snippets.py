# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 12:05:48 2019

@author: Hesiris
"""


#from magenta.models.onsets_frames_transcription import data

from util_audio import note_sequence as nsequence
from util_audio import midi_from_file
import util_audio
from util_audio import audio_complete
from tensorflow.keras.utils import Sequence

from essentia.standard import (NSGConstantQ, ConstantQ,
    NSGIConstantQ)

import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display
import soundfile
import scipy

#import fluidsynth

import tensorflow.keras as keras
from RDCNN import res_net
import ProgressBar as PB
import os
from util_dataset import DataManager
#soundfile.write('stereo_file.flac', data, samplerate, format='flac', subtype='PCM_24')

import pandas
import logging

#%% Predef functions

def add_note(sequence, instrument, program, pitch, start, end, velocity=127,  is_drum=False):
    note = sequence.notes.add()
    note.instrument = instrument
    note.program = program
    note.start_time = start
    note.end_time = end
    note.pitch = pitch
    note.velocity = velocity
    note.is_drum = is_drum

#%% Load File
path = 'C:/Code/Python/[Thesis]/AMT SAGA/data/'
filename = 'Nyan_Cat_piano.mp3'
#filename = 'Supersonic.mp3'
#filename = 'Fuyu no Epilogue.flac'
#filename = 'Utauyo!! MIRACLE.mp3'
#filename = 'Runaway.mp3'

wav,sr = librosa.load(path+filename,sr=None,mono=False,duration=30,offset=0)
y = wav[0]

#filename = librosa.util.example_audio_file()
#y, sr = librosa.load(filename)


#%% Chroma test
chroma_orig = librosa.feature.chroma_cqt(y=y, sr=sr)

# For display purposes, let's zoom in on a 15-second chunk from the middle of the song
idx = tuple([slice(None), slice(*list(librosa.time_to_frames([0, 15])))])

# And for comparison, we'll show the CQT matrix as well.
C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=12*3, n_bins=7*12*3))

chroma_os = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=12*3)
y_harm = librosa.effects.harmonic(y=y, margin=8)
chroma_os_harm = librosa.feature.chroma_cqt(y=y_harm, sr=sr, bins_per_octave=12*3)

chroma_filter = np.minimum(chroma_os_harm,
                       librosa.decompose.nn_filter(chroma_os_harm,
                                                   aggregate=np.median,
                                                   metric='cosine'))
chroma_smooth = scipy.ndimage.median_filter(chroma_filter, size=(1, 9))


plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max)[idx],
                         y_axis='cqt_note', bins_per_octave=12*3)
plt.colorbar()
plt.ylabel('CQT')
plt.subplot(3, 1, 2)
librosa.display.specshow(chroma_orig[idx], y_axis='chroma')
plt.ylabel('Original')
plt.colorbar()
plt.subplot(3, 1, 3)
librosa.display.specshow(chroma_smooth[idx], y_axis='chroma', x_axis='time')
plt.ylabel('Processed')
plt.colorbar()
plt.tight_layout()
plt.show()


#%% Read MIDI test
'''midi.notes -- list of notes
    midi.notes[i].instrument --- instrument id in song, 0,1,2...
    midi.notes[i].program --- instrument id: 0--piano, electric bass(pick)--34 
    midi.notes[i].is_drum --- ~
    midi_io.pretty_midi.program_to_instrument_name
''' 
#midi_filename = 'Nyan_cat_web_transcription.mid'
midi_filename = 'Runaway.mid'
with open(path+midi_filename,'rb') as midi_file:
    nyan_midi_raw = midi_file.read()
#nyan_midi_raw = midi_io.midi_file_to_note_sequence(path+midi_filename)

nyan_midi = midi_from_file(nyan_midi_raw)

#%% Synthetize MIDI test



#%% Write MIDI test

#sequence = music_pb2.NoteSequence()
sequence = nsequence()
  
sequence.add_note(0,0,62,1,2)
sequence.add_note(0,0,66,1,2)
sequence.add_note(0,0,69,1,2)

sequence.add_note(1,64,60,2.1,3)
sequence.add_note(2,52,64,2.1,3)
sequence.add_note(2,55,67,2.1,3)

output_file_name = 'midi_out_test.mid'
#midi_io.note_sequence_to_midi_file(sequence, path+output_file_name)
sequence.save(path+output_file_name)

#%% Test simple transcription approach
    # For display purposes, let's zoom in on a 15-second chunk from the middle of the song
idx = tuple([slice(None), slice(*list(librosa.time_to_frames([0, 30])))])

# And for comparison, we'll show the CQT matrix as well.
os_factor = 3

C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=12*os_factor, n_bins=7*12*os_factor))

y_harm = librosa.effects.harmonic(y=y, margin=8)
C_os_harm = librosa.cqt(y=y_harm, sr=sr, bins_per_octave=12*3)
C_db = librosa.amplitude_to_db(C, ref=np.max)

C_filter = np.minimum(C_db,
                   librosa.decompose.nn_filter(C_db,
                                               aggregate=np.median,
                                               metric='cosine'))
C_smooth = scipy.ndimage.median_filter(C_filter, size=(1, 9))

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
librosa.display.specshow(C_db[idx],
                         y_axis='cqt_note', bins_per_octave=12*os_factor, x_axis='time')
plt.colorbar()
plt.ylabel('CQT')
plt.subplot(3, 1, 2)
librosa.display.specshow(C_filter[idx], bins_per_octave=12*os_factor,x_axis='time', y_axis='cqt_note')
plt.ylabel('C filter')
plt.colorbar()
plt.subplot(3, 1, 3)
librosa.display.specshow(C_smooth[idx], y_axis='cqt_note', x_axis='time')
plt.ylabel('C Smooth')
plt.colorbar()
plt.tight_layout()
plt.show()

#%% Specrtogram subtraction test
#add perceptual weighting?

def band_stop(D,filter_min,filter_max,filter_db,sr=44100,n_fft=2048):
    filter_min = int(filter_min* n_fft/sr)
    filter_max = int(filter_max* n_fft/sr)
    for i in range(D.shape[1]):
        for j in range(filter_min,filter_max):
            D[j,i] -= filter_db
    return D

def subract_pen(base,to_subtract,reflect=False,offset=0,threshold = -60):
    for i in range(to_subtract.shape[1]):
        for j in range(to_subtract.shape[0]):
            base[j,offset+i] -= to_subtract[j,i] *\
                    -1 if reflect and base[j,offset+i]<threshold else 1
    return base

buckets = 4096
F = librosa.stft(y,center=False,n_fft=buckets)
mag,ph = librosa.core.magphase(F)
ref_mag = np.max(mag)

D = librosa.amplitude_to_db(mag,ref=ref_mag)
plt.figure(figsize=(12, 24))
#plt.subplot(5, 1, 1)
#librosa.display.specshow(D,y_axis='linear',sr=sr)
#plt.ylabel('Linear DB')
#plt.colorbar(format='%+2.0f dB')

plt.subplot(3,1,1)
librosa.display.specshow(D, y_axis='log',sr=sr)
plt.ylabel('Log DB')
plt.colorbar(format='%+2.0f dB')

#plt.subplot(5,1,3)
#mag_mel = librosa.feature.melspectrogram(S=mag**2,n_mels = 512)
#librosa.display.specshow(librosa.power_to_db(mag_mel,ref=np.max), y_axis='mel',sr=sr) 
#plt.ylabel('Mel')
#plt.colorbar(format='%+2.0f dB')

offset = 16000
rem_wav_offset = y - np.array([0]*offset+list(y[:-offset]))
F2 = librosa.stft(rem_wav_offset,center=False,n_fft=buckets)
mag2,ph2 = librosa.core.magphase(F2)
mag2 = mag-mag2
D = librosa.amplitude_to_db(mag2,ref=ref_mag)


#band_stop(D,12000,18000,50,n_fft=buckets)
plt.subplot(3,1,2)
librosa.display.specshow(D, y_axis='log',sr=sr)
plt.ylabel('Log, removed filter')
plt.colorbar(format='%+2.0f dB')

plt.subplot(3,1,3)
plt.ylabel('Waveform transformed back')
y_back = librosa.istft(librosa.db_to_amplitude(D,ref=ref_mag)*ph)
librosa.display.waveplot(y_back,sr=sr)

soundfile.write(path+'filtered.flac', y_back, sr, format='flac', subtype='PCM_24')


#%% Phase-based Harmonic/Percussive Separation by 
# Estefan´ıa Cano, Mark Plumbley, Christian Dittmar,


#Constants:
fs = sr
N = 4096
H = np.int(N/4)

#Step 1: calculate S and phi
F = librosa.stft(y,center=False,n_fft=N)
S,ph = librosa.core.magphase(F)
ref_mag = np.max(S)

#Step 2 Dk_low, Dk_high for each freq. bin
dk_low = np.zeros(int(N/2)+1,dtype=np.double)
dk_high = np.zeros(int(N/2)+1,dtype=np.double)

f_inc = fs/N
f_d = f_inc/2
 
f_k = f_d
dfk = 2*np.pi * H  /fs
for i in range(int(N/2)+1):
    dk_low[i] = dfk * (f_k-f_d)
    dk_high[i] = dfk * (f_k+f_d) 
    f_k += f_inc
    
#Step 3&4: Peaks: Top 5 values, at least 0.5 Barks apart

#3.1: Remove all Values that are below a threshold
th = librosa.db_to_amplitude(-25)*ref_mag
Q = np.where(S>th,S,0)
M = np.zeros(S.shape,dtype=np.int8)


#3.2: remove reigster outside saxophone range -- note relevant -> skipped
#3.3: The peaks within 0.5 Bark of the highest amplitude are replaced with the highest peak
bark = lambda f: 6 * np.arcsinh(f/600.0)
arcbark = lambda B: np.sinh(B/6.0) * 600.0
for i in range(Q.shape[1]):
    for j in range(5):
        if np.max(Q[:,i]) == 0:
            break;
        lmaxind = np.argmax(Q[:,i])
        fmax = lmaxind*sr/N
        B = bark(fmax)
        llimf,hlimf = arcbark(B-0.5),arcbark(B+0.5)
        llimi,hlimi = int(llimf*N/sr),int(hlimf*N/sr)
        M[llimi:hlimi,i] = 1
        Q[llimi:hlimi,i] = 0
     
#graphical check
#plt.figure(figsize=(12, 16))
#plt.subplot(4, 1, 1)
#librosa.display.specshow(librosa.amplitude_to_db(S,ref=ref_mag), y_axis='log',sr=sr)
#plt.ylabel('Original spectrogram')
#plt.subplot(4, 1, 2)
#librosa.display.specshow(librosa.amplitude_to_db(np.where(S>th,S,0),ref=ref_mag), y_axis='log',sr=sr)
#plt.ylabel('S after threshold filtered')
#plt.subplot(4, 1, 3)
#librosa.display.specshow(librosa.amplitude_to_db(Q,ref=ref_mag), y_axis='log',sr=sr)
#plt.ylabel('S after removing peaks')
#plt.colorbar(format='%+2.0f dB')
#plt.subplot(4, 1, 4)
#librosa.display.specshow((M-1)/20, y_axis='log',sr=sr)
#plt.ylabel('Peaks, Bark scaled')

#Step 5 Masked phase spectrogram
ph_rad = np.angle(F)
ph_masked = ph_rad * M


#Step 6
ph_unwrap = np.unwrap(ph_masked)
derivate = lambda a: [0 if i==0 else (a[i]-a[i-1]) for i in range(a.shape[0])]

Ph= np.apply_along_axis(derivate,1,ph_unwrap) /2.0/np.pi #/fs
#Step 7
binary_spectral_mask = lambda a: [(l<i) and (i<h) for i,l,h in zip(a,dk_low,dk_high)]
H_bsm = np.apply_along_axis(binary_spectral_mask,0,Ph)
P_bsm = 1-H_bsm

#Step8 Spectral leakage

#Step9
S_harm = S*H_bsm
S_perc = S*P_bsm

#graphical check
plt.figure(figsize=(12, 16))
plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(S,ref=ref_mag), y_axis='log',sr=sr)
plt.ylabel('Original spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(S_harm,ref=ref_mag), y_axis='log',sr=sr)
plt.ylabel('Harmonic Components')
plt.colorbar(format='%+2.0f dB')
plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(S_perc,ref=ref_mag), y_axis='log',sr=sr)
plt.ylabel('Percussive Components')
plt.colorbar(format='%+2.0f dB')

y_harm =  librosa.istft(S_harm*ph)
y_perc =  librosa.istft(S_perc*ph)
soundfile.write(path+'harmonic.flac', y_harm, sr, format='flac', subtype='PCM_24')
soundfile.write(path+'percussive.flac', y_perc, sr, format='flac', subtype='PCM_24')

#%% Test transcription

def get_pitch(frame):
    framesize = frame.shape[0]
    
    for j in range(framesize):
        if (frame[j]>threshold):
            cand = []
            for k in range(j,framesize):
                if frame[k]>threshold:
                    cand.append(frame[k])
                else:
                    break;
            if len(cand)>2:
                return np.int(np.round(j/3+ np.argmax(cand)))
    return -1
sequence = nsequence()


pitch_prev = -1;
start_time = 0;
end_time = 0;
threshold = -40;
is_silence = True
for i,frame in enumerate(C_smooth[idx].transpose()):    
    

    pitch= get_pitch(frame)  
    is_silence =  pitch==-1
    #pitch = np.int(np.round(np.argmin(frame)/os_factor))

    if pitch!=pitch_prev and not is_silence:            
        if pitch_prev!=-1:
            end_time = i/43
            print('{} at {}'.format(pitch,end_time))
            sequence.add_note(instrument=0,program=85,pitch=pitch_prev+48,start = start_time,end=end_time)
            start_time = end_time



    #is_silence = np.max(frame)<threshold
            
    pitch_prev = pitch
 
output_file_name = 'midi_out_test.mid'
sequence.save(path+output_file_name)

#%% Test MIDi generation-subtraction:
# Generate a sustained G major chord for a whole note with a C, one octave lower for a half in the middle
from magenta.music import midi_io
sr = 44100
sf_path = '/home/hesiris/Documents/Thesis/soundfonts/GM_soundfonts.sf2'

ns = nsequence()
ns.add_note(0,0,67,start=0,end=2)
ns.add_note(0,0,71,start=0,end=2)
ns.add_note(0,0,74,start=0,end=2)

ns.add_note(0,0,48,start=0.5,end=1.5)

mid = midi_io.note_sequence_to_pretty_midi(ns.sequence)
wf = mid.fluidsynth(fs=sr, sf2_path=sf_path)#[:sr*2]

#Generate Fourier and subtract
buckets = 4096
F = librosa.stft(wf,n_fft=buckets)
mag,ph = librosa.core.magphase(F)
ref_mag = np.max(mag)

D = librosa.amplitude_to_db(mag,ref=ref_mag)


guess = nsequence()
guess.add_note(0,0,48,0,1)
mid_guess = midi_io.note_sequence_to_pretty_midi(guess.sequence)
wf_guess = mid_guess.fluidsynth(fs=sr, sf2_path=sf_path)#[:sr]

F_guess = librosa.stft(wf_guess,center=False,n_fft=buckets)
mag_guess,_ = librosa.core.magphase(F_guess)
ref_mag_guess = np.max(mag_guess)
overkill_factor = 1
mag_guess = mag_guess * ref_mag / ref_mag_guess * overkill_factor
D_guess = librosa.amplitude_to_db(mag_guess,ref=ref_mag)

total_s = wf.shape[0]/sr
offset = np.int(np.floor( mag.shape[1]/total_s * 0.5) )
mag_sub = mag - np.concatenate(
                 (np.zeros((mag.shape[0],offset)) ,
                 mag_guess,
                 np.zeros((mag.shape[0],mag.shape[1]-offset-mag_guess.shape[1]))),
                 axis = 1)

mag_sub = np.maximum(mag_sub,0,mag_sub)
D_sub = librosa.amplitude_to_db(mag_sub,ref=ref_mag)
wf_sub = librosa.istft(mag_sub*ph)

plt.figure(figsize=(12, 15))

plt.subplot(3,1,2)
librosa.display.specshow(D, y_axis='log',sr=sr)
plt.ylabel('Log DB')
plt.colorbar(format='%+2.0f dB')

plt.subplot(3,1,1)
librosa.display.specshow(D_guess, y_axis='log',sr=sr)
plt.ylabel('Log DB')
plt.colorbar(format='%+2.0f dB')

plt.subplot(3,1,3)
librosa.display.specshow(D_sub, y_axis='log',sr=sr)
plt.ylabel('Log DB')
plt.colorbar(format='%+2.0f dB')

#Save both midi and wav
path = './data/'

output_file_name = 'midi_out_test.mid'
ns.save(path+output_file_name)

soundfile.write(path + 'wave_test.flac', wf, sr, format='flac', subtype='PCM_24')
soundfile.write(path + 'wave_test_guess.flac', wf_guess, sr, format='flac', subtype='PCM_24')
soundfile.write(path + 'wave_test_sub.flac', wf_sub, sr, format='flac', subtype='PCM_24')

#%% Same thing, short version with the audio_util
sf_path = '/home/hesiris/Documents/Thesis/soundfonts/GM_soundfonts.sf2'
buckets = 4096

#Generate notes
ns = nsequence(sf2_path = sf_path)
ns.add_note(0,0,'G4',start=0,end=2)
ns.add_note(0,0,'b',start=0,end=2)
ns.add_note(0,0,'D5',start=0,end=2)

ns.add_note(0,0,'C3',start=0.5,end=1.5)

#Generate wave and spectral representations
wf = ns.render()
ac = util_audio.audio_complete(wf,buckets)

#same for the C only
guess = nsequence()
guess.add_note(0,0,48,0,1)
ac_guess = util_audio.audio_complete(guess.render(),buckets)

#Subtract. Create a copy to plot later
ac_sub = ac.clone()
ac_sub.subtract(ac_guess,offset=0.5,attack_compensation = 0)

#plot
util_audio.plot_specs([ac_guess,ac,ac_sub])

#Save both midi and wav
path = './data/'
ac_guess.save(path + 'wave_test_guess.flac')
ac.save(path + 'wave_test.flac')
ac_sub.save(path + 'wave_test_sub.flac',)

#%% File save test
#    y_foreground = librosa.istft(D_harmonic)
#    y_background = librosa.istft(D_percussive)
#
#    soundfile.write('C:/Code/Python/[Thesis]/Magenta/data/audio/librosa_voice.flac', y_foreground, sr, format='flac', subtype='PCM_24')
#    soundfile.write(path + 'wave_test.flac', y_background, sr, format='flac', subtype='PCM_24')
#    librosa.output.write_wav(path + 'sample.wav', sf.samples[0].raw_sample_data,sf.samples[0].sample_rate)


#%% RNN test
res_dir = './results_mnist'
convolutional_layer_count = 16
pool_layer_frequency = 6
feature_expand_frequency = 6#pool_layer_frequency
residual_layer_frequencies = [2]

#for convolutional_layer_count,pool_layer_frequency,feature_expand_frequency, \
#    residual_layer_frequency in zip([16,32,32],[6,12,32],[6,12,32],[2,None,2]):
    
    #TO test: -residuals -dualchannel -two_fc_at_the_end

fashion_mnist = keras.datasets.fashion_mnist
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print('Loaded {} images with shape: {}'.format(train_images.shape[0],train_images.shape[1:]))
 
train_images = train_images / (255.0/2)-1
test_images = test_images / (255.0/2)-1

train_images = train_images.reshape(train_images.shape[0],28,28,1)
test_images = test_images.reshape(test_images.shape[0],28,28,1)

def sim_shuff(a,b):
    c = np.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)]
    np.random.shuffle(c)
    a2 = c[:, :a.size//len(a)].reshape(a.shape)
    b2 = c[:, a.size//len(a):].reshape(b.shape)
    return a2,b2
train_images,train_labels = sim_shuff(train_images,train_labels)
test_images,test_labels = sim_shuff(test_images,test_labels)
#train_labels = keras.utils.to_categorical(train_labels,10)
#test_labels = keras.utils.to_categorical(test_labels,10)
#%% Using train_on_batch
suffix = '_' + str(convolutional_layer_count) + '_' + str(pool_layer_frequency) + \
         '_' + str(feature_expand_frequency) + '_' + str(residual_layer_frequencies) + \
                '_checkpoint_verif'
print('Suffix used: ' + suffix)

rn = res_net(input_shapes=[(28,28,1)],kernel_sizes=[(3,3)],pool_sizes=[(2,2)],
             output_classes=10,
             convolutional_layer_count=convolutional_layer_count,
             pool_layer_frequency=pool_layer_frequency,
             feature_expand_frequency = feature_expand_frequency,
             residual_layer_frequencies=residual_layer_frequencies,
             checkpoint_dir= './data/checkpoints/metrics_test', checkpoint_frequency = 2000,
             checkpoint_prefix = 'checkpoint_' + suffix)
rn.plot_model(os.path.join(rn.checkpoint_dir,'model_' + suffix + '.png'))


test_set_size = len(train_labels)//4
pb = PB.ProgressBar(test_set_size,sound='beep')#(train_images.shape[0])
for image,label in zip(train_images[:test_set_size,:],train_labels[:test_set_size]):
    expanded_image = np.expand_dims(image,axis=0)
    expanded_label = np.expand_dims(label,axis=0)
    rn.train([expanded_image],expanded_label)
    pb.check_progress()
    
test_set_size = len(test_labels)//2
pb = PB.ProgressBar(test_set_size)
for image,label in zip(test_images[:test_set_size,:],test_labels[:test_set_size]):
    expanded_image = np.expand_dims(image,axis=0)
    expanded_label = np.expand_dims(label,axis=0)
    rn.test(expanded_image,expanded_label)
    pb.check_progress()

rn.report(training=True, filename_training=os.path.join(rn.checkpoint_dir,'training' + suffix + '.csv'),
          test = True,   filename_test = os.path.join(res_dir,'test' + suffix + '.csv'),
                          class_names = class_names)

rn.plot(metrics_to_plot=[1],moving_average_window=100,
             filename_training = os.path.join(rn.checkpoint_dir,'training' + suffix + '.png'), 
             filename_test = os.path.join(res_dir,'test' + suffix + '.png'))
#%% Verify correct checkpoint load
checkpoint_idx = 2000
fn_weights = ('./data/checkpoints/200/'
            'checkpoint__16_6_6_[2]_checkpoint_verif_{}.h5').format(checkpoint_idx)
            
rn = res_net(input_shapes=[(28,28,1)],kernel_sizes=[(3,3)],pool_sizes=[(2,2)],
             output_classes=10,
             convolutional_layer_count=convolutional_layer_count,
             pool_layer_frequency=pool_layer_frequency,
             feature_expand_frequency = feature_expand_frequency,
             residual_layer_frequencies=residual_layer_frequencies,
             checkpoint_dir= './data/checkpoints/200', checkpoint_frequency = 200,
             checkpoint_prefix = 'checkpoint_' + suffix,
             weights_load_checkpoint_filename=fn_weights)

test_set_size = 2000
pb = PB.ProgressBar(test_set_size)
for image,label in zip(test_images[:test_set_size,:],test_labels[:test_set_size]):
    expanded_image = np.expand_dims(image,axis=0)
    expanded_label = np.expand_dims(label,axis=0)
    rn.test(expanded_image,expanded_label)
    pb.check_progress()
    
rn.report(test = True, filename_test= fn_weights + '.csv', 
          class_names = class_names)

#%% Verify logging
suffix = '_' + str(convolutional_layer_count) + '_' + str(pool_layer_frequency) + \
         '_' + str(feature_expand_frequency) + '_' + str(residual_layer_frequencies) + \
                '_checkpoint_verif'
                
chkp_dir = './data/checkpoints/200'
print('Suffix used: ' + suffix)

rn = res_net(input_shapes=[(28,28,1)],kernel_sizes=[(3,3)],pool_sizes=[(2,2)],
             output_classes=10,
             convolutional_layer_count=convolutional_layer_count,
             pool_layer_frequency=pool_layer_frequency,
             feature_expand_frequency = feature_expand_frequency,
             residual_layer_frequencies=residual_layer_frequencies,
             checkpoint_dir= chkp_dir, checkpoint_frequency = 250,
             checkpoint_prefix = 'checkpoint_' + suffix)
rn.plot_model(os.path.join(res_dir,'model_' + suffix + '.png'))


test_set_size = 500
pb = PB.ProgressBar(test_set_size,sound='beep')#(train_images.shape[0])
for image,label in zip(train_images[:test_set_size,:],train_labels[:test_set_size]):
    expanded_image = np.expand_dims(image,axis=0)
    expanded_label = np.expand_dims(label,axis=0)
    rn.train([expanded_image],expanded_label)
    pb.check_progress()
    
    
rn2 = res_net(input_shapes=[(28,28,1)],kernel_sizes=[(3,3)],pool_sizes=[(2,2)],
             output_classes=10,
             convolutional_layer_count=convolutional_layer_count,
             pool_layer_frequency=pool_layer_frequency,
             feature_expand_frequency = feature_expand_frequency,
             residual_layer_frequencies=residual_layer_frequencies,
             checkpoint_dir= chkp_dir, checkpoint_frequency = 200,
             checkpoint_prefix = 'checkpoint_' + suffix)

def check(rn1,rn2):
    if rn1.metrics_names!=rn2.metrics_names:
        return 'names mismatch!'
    if rn1.metrics_train!=rn2.metrics_train:
        return 'training mismatch!'
    if rn1.metrics_test!=rn2.metrics_test:
        return 'testing mismatch!'
    return "OK"

rn2.load_metrics(chkp_dir,index=500,test=False,use_csv=False)
print(check(rn,rn2))

test_set_size = 200
pb = PB.ProgressBar(test_set_size,sound='beep')#(train_images.shape[0])
for image,label in zip(train_images[:test_set_size,:],train_labels[:test_set_size]):
    expanded_image = np.expand_dims(image,axis=0)
    expanded_label = np.expand_dims(label,axis=0)
    rn.test([expanded_image],expanded_label)
    pb.check_progress()
    

men, tra,tes, csv = [[True, False]]*4
for me in men:
    for tr in tra:
        for te in tes:
            for cs in csv:
                rn.save_metrics(directory=chkp_dir, index=None,
                                 training=tr, test = te, use_csv=cs)
                if tr:
                    rn2.metrics_train = []
                if te:
                    rn2.metrics_test = []
                if me:        
                    rn2.metics_names = None
                    
                rn2.load_metrics(directory=chkp_dir, index = None,
                                     training=tr, test = te, use_csv=cs, 
                                     load_metric_names=me)
                print(check(rn,rn2),':',me,tr,te,cs)

#%% Using higher batch size
batch_size = 16

suffix =  '_' + str(convolutional_layer_count) + '_' + str(pool_layer_frequency) + \
         '_' + str(feature_expand_frequency) + '_' + str(residual_layer_frequencies) + \
         '_bs' + str(batch_size)
         
rn = res_net(input_shapes=[(28,28,1)],kernel_sizes=[(3,3)],pool_sizes=[(2,2)],
             output_classes=10,
             batch_size=batch_size,
             convolutional_layer_count=convolutional_layer_count,
             pool_layer_frequency=pool_layer_frequency,
             feature_expand_frequency = feature_expand_frequency,
             residual_layer_frequencies=residual_layer_frequencies,
             checkpoint_dir= './data/checkpoints/200/', checkpoint_frequency = 200,
             checkpoint_prefix = 'checkpoint_' + suffix,
             metrics=['sparse_categorical_accuracy'])
oname = str(rn.model.optimizer)
oname = oname[oname.find('optimizers')+11:oname.find('object')-1]
suffix += '_' + oname

rn.plot_model(os.path.join(res_dir,'model_' + suffix + '.png'))

test_set_size = 10000
pb = PB.ProgressBar(test_set_size//batch_size,sound='beep')#(train_images.shape[0])

cb_x=np.zeros([0]+list(train_images.shape[1:]))
cb_y=np.zeros([0]+list(train_labels.shape[1:]))
for image,label in zip(train_images[:test_set_size,:],train_labels[:test_set_size]):
    expanded_image = np.expand_dims(image,axis=0)
    expanded_label = np.expand_dims(label,axis=0)
    if cb_x.shape[0]<batch_size:
        cb_x=np.concatenate((cb_x,expanded_image))
        cb_y=np.concatenate((cb_y,expanded_label))
    else:
        rn.train(cb_x,cb_y)
        pb.check_progress()
        cb_x=expanded_image
        cb_y=expanded_label
    
test_set_size = 5000
pb = PB.ProgressBar(test_set_size//batch_size)
cb_x=np.zeros([0]+list(train_images.shape[1:]))
cb_y=np.zeros([0]+list(train_labels.shape[1:]))
for image,label in zip(test_images[:test_set_size,:],test_labels[:test_set_size]):
    expanded_image = np.expand_dims(image,axis=0)
    expanded_label = np.expand_dims(label,axis=0)
    if cb_x.shape[0]<batch_size:
        cb_x=np.concatenate((cb_x,expanded_image))
        cb_y=np.concatenate((cb_y,expanded_label))
    else:
        rn.test(cb_x,cb_y,use_predict=True)
        pb.check_progress()
        cb_x=expanded_image
        cb_y=expanded_label

rn.report(training=False, filename_training=os.path.join(res_dir,'training' + suffix + '.csv'),
          test = True,   filename_test = os.path.join(res_dir,'test' + suffix + '.csv'),
                          class_names = class_names)

rn.plot(metrics_to_plot=[0,1,2,3],moving_average_window=20,
             filename_training = os.path.join(res_dir,'training' + suffix + '.png'), 
             filename_test = os.path.join(res_dir,'test' + suffix + '.png'))

#%%USing fit_generator --- NOPE
import threading 
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()
        
        
def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

test_set_size = 1000

def gen_mnist(train_images,train_labels,test_set_size):
    lock = threading.Lock()
    pb = PB.ProgressBar(test_set_size,sound='beep')#(train_images.shape[0])
    for image,label in zip(train_images[:test_set_size,:],train_labels[:test_set_size]):
        expanded_image = np.expand_dims(image,axis=0)
        expanded_label = np.expand_dims(label,axis=0)
        with lock:
            yield ([expanded_image],expanded_label)
        with lock:
            pb.check_progress()
    
rn.fit_generator(gen_mnist(train_images,train_labels,test_set_size),
                 steps_per_epoch=test_set_size,
                 workers=8,use_multiprocessing=True)
#for x,y in gen_mnist(train_images,train_labels,test_set_size):
#    rn.train(x,y)

test_images = test_images[:1000]
pb = PB.ProgressBar(test_images.shape[0])
for image,label in zip(test_images,test_labels):
    expanded_image = np.expand_dims(image,axis=0)
    expanded_label = np.expand_dims(label,axis=0)
    rn.test([expanded_image],expanded_label)
    pb.check_progress()

rn.report(training=True, filename_training=os.path.join(res_dir,'training' + suffix + '.csv'),
          test = True,   filename_test = os.path.join(res_dir,'test' + suffix + '.csv'),
                          class_names = class_names)

rn.plot(metrics_to_plot=[1],moving_average_window=100,
             filename_training = os.path.join(res_dir,'training' + suffix + '.png'), 
             filename_test = os.path.join(res_dir,'test' + suffix + '.png'))


#%% RNN contiuous -- Create Model

path_cont = './data/housing'
fn_cont = 'housing.data'

convolutional_layer_count = 0
feature_expand_frequency = 0
residual_layer_frequencies = 0

suffix = 'housing_' + str(convolutional_layer_count) + \
         '_' + str(feature_expand_frequency) + '_' + str(residual_layer_frequencies) + \
                '_mutiple_residuals'


dataframe = pandas.read_csv(os.path.join(path_cont,fn_cont), delim_whitespace=True, header=None)
dataset = dataframe.values
np.random.shuffle(dataset)
training_length = int(dataset.shape[0]*0.8)
training_dataset = dataset[:training_length, :]
test_dataset = dataset[training_length:, :]
out_norm = np.max(dataframe.values[:,-1])


rn = res_net(input_shapes=[(13,1,1)],kernel_sizes=[(1,1)],pool_sizes=[(0,0)],
             output_classes=1,
             convolutional_layer_count=convolutional_layer_count,
             pool_layer_frequency=0,
             feature_expand_frequency = feature_expand_frequency,
             residual_layer_frequencies=residual_layer_frequencies,
             checkpoint_dir= './data/checkpoints', checkpoint_frequency = 200,
             checkpoint_prefix = 'checkpoint_' + suffix,
             metrics=['mean_squared_error'])

rn.plot_model(os.path.join(path_cont,'model_' + suffix + '.png'))

epochs = 1
pb = PB.ProgressBar(training_length*epochs)
for i in range(epochs):
    np.random.shuffle(training_dataset)
    for x in training_dataset:
        expanded_x = x[np.newaxis,:-1,np.newaxis,np.newaxis]
        expanded_y = x[np.newaxis,-1]
        rn.train(expanded_x, expanded_y)
        pb.check_progress()

pb = PB.ProgressBar(dataset.shape[0]-training_length)
for x in test_dataset:
    expanded_x = x[np.newaxis,:-1,np.newaxis,np.newaxis]
    expanded_y = x[np.newaxis,-1]
    rn.test([expanded_x], expanded_y)
    pb.check_progress()
  
    
#rn.plot(metrics_to_plot=[0,1,2,3,4,5],moving_average_window=0,
#             #filename_training = os.path.join(path_cont,'training' + suffix + '.png'),
#             filename_test = os.path.join(path_cont,'test' + suffix + '.png'))


mse = np.mean((rn.get_metric_test('y_true_scaled')-
                      rn.get_metric_test('y_pred_scaled'))**2)
mse_internal = np.mean(rn.get_metric_test('mse_scaled'))

print('MSE = {}; MSE scaled = {}'.format(mse,mse_internal))
rn.report(filename_test = os.path.join(path_cont,'test' + suffix + '.txt'))

#%% Housing data with a simple Keras model, used directly

import numpy
import pandas
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1,activation='sigmoid',kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


path_cont = './data/housing'
fn_cont = 'housing.data'
dataframe = pandas.read_csv(os.path.join(path_cont,fn_cont), delim_whitespace=True, header=None)
dataset = dataframe.values

output_max = np.max(dataset[:,13])
output_min = np.min(dataset[:,13])
dataset[:,13] = (dataset[:,13] - output_min)/( output_max-output_min)
training_length = int(dataset.shape[0]*0.8)
training_set = dataset[:training_length, :]
test_length = dataset.shape[0] - training_length
test_set = dataset[training_length:, :]


model = baseline_model()

epochs = 1
pb = PB.ProgressBar(training_length*epochs)
for i in range(epochs):
    np.random.shuffle(training_set)
    for sample in training_set:
        x = sample[np.newaxis,0:13]
        y = sample[np.newaxis,13]
        model.train_on_batch(x,y,sample_weight=None, class_weight=None)
        pb.check_progress()
    
y_true = np.zeros(test_length)
y_pred = np.zeros(test_length)
for i,sample in enumerate(test_set):
    x = sample[np.newaxis,0:13]
    y = sample[13]
    y_pred[i] = model.predict_on_batch(x)*(output_max-output_min)+output_min
    y_true[i] = y*(output_max-output_min)+output_min

mse = np.mean((y_pred-y_true)**2)
    
print('MSE = {}'.format(mse))
    

#%%
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


X = dataset[:,0:13]
Y = dataset[:,13]

#valuate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)



print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


#%% Generator test. Housing data with a simple Keras model, used directly

import numpy
import pandas
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1,activation='sigmoid',kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

def generate_input():
    epochs = 1
    pb = PB.ProgressBar(training_length*epochs)
    for i in range(epochs):
        np.random.shuffle(training_set)
        for sample in training_set:
            x = sample[np.newaxis,0:13]
            y = sample[np.newaxis,13]
            yield (x,y)
#            model.train_on_batch(x,y,sample_weight=None, class_weight=None)
            pb.check_progress()             


path_cont = './data/housing'
fn_cont = 'housing.data'
dataframe = pandas.read_csv(os.path.join(path_cont,fn_cont), delim_whitespace=True, header=None)
dataset = dataframe.values

output_max = np.max(dataset[:,13])
output_min = np.min(dataset[:,13])
dataset[:,13] = (dataset[:,13] - output_min)/( output_max-output_min)
training_length = int(dataset.shape[0]*0.8)
training_set = dataset[:training_length, :]
test_length = dataset.shape[0] - training_length
test_set = dataset[training_length:, :]


model = baseline_model()

model.fit_generator(generate_input(),
                    steps_per_epoch=training_length, epochs=10)

    
y_true = np.zeros(test_length)
y_pred = np.zeros(test_length)
for i,sample in enumerate(test_set):
    x = sample[np.newaxis,0:13]
    y = sample[13]
    y_pred[i] = model.predict_on_batch(x)*(output_max-output_min)+output_min
    y_true[i] = y*(output_max-output_min)+output_min

mse = np.mean((y_pred-y_true)**2)
    
print('MSE = {}'.format(mse))
    
#%% Mel-Spectrogram test




# Create the audio file
sf_path = '/home/hesiris/Documents/Thesis/soundfonts/GM_soundfonts.sf2'
buckets = 4096

#Generate notes
ns = nsequence(sf2_path = sf_path)
ns.add_note(0,0,'G4',start=0,end=2)
ns.add_note(0,0,'b',start=0,end=2)
ns.add_note(0,0,'D5',start=0,end=2)
ns.add_note(0,0,'C3',start=0.5,end=1.5)

#Generate wave and spectral representations
wf = ns.render()
ac = util_audio.audio_complete(wf,buckets)



#%% Using Librosa 
sr=44100
bins_per_note = 1
ac_short = ac.resize(attribs=['F','mag','ph'],start=0.5,duration=1,target_frame_count=40)
C = ac.slice_C(start=0.5,duration=1,target_frame_count=40,
               bins_per_note = bins_per_note,filter_scale=2)
ac.plot_spec()
ac_short.plot_spec()

plt.figure(figsize=(10, 5))
librosa.display.specshow(librosa.amplitude_to_db(np.abs(C), ref=np.max),
                          sr=ac_short.sr, hop_length=ac_short.hl,
                          x_axis='time', y_axis='cqt_note')
plt.colorbar(format='%+2.0f dB')
plt.title('Constant-Q power spectrum')
plt.tight_layout()

#plt.figure()
#nbins = (librosa.note_to_midi('C8') - librosa.note_to_midi('A1'))*bins_per_note
#Q = librosa.cqt(ac.wf,sr=sr,fmin=librosa.note_to_hz('A1'),n_bins=nbins,
#                bins_per_octave=12*bins_per_note,
#            filter_scale=2, sparsity=0.01)
#print(Q.shape)
#librosa.display.specshow(librosa.amplitude_to_db(np.abs(Q[:,43:130]), ref=np.max),
#                          sr=sr, x_axis='time', y_axis='cqt_note')
#plt.colorbar(format='%+2.0f dB')
#plt.title('Constant-Q power spectrum')
#plt.tight_layout()

#%% Testing compression / Essentia


sr = 44100
N=4096
sf_path = '/home/hesiris/Documents/Thesis/soundfonts/GM_soundfonts.sf2'
data_path = './data'
output_path = './output/frequency_compression'
midi_filename = 'Nyan_cat_web_transcription.mid'
#midi_filename = 'Runaway.mid'

mid = nsequence(os.path.join(data_path,midi_filename))
ac = audio_complete(mid.render(sf2_path = sf_path), N)

#ac.save(os.path.join(output_path,'full.flac'))

#%%
bins_per_note = 1
filter_scale=2
highest_note = 'C8'
lowest_note='A0'
nbins = (librosa.note_to_midi(highest_note) - 
                 librosa.note_to_midi(lowest_note))*bins_per_note
fmin=librosa.note_to_hz(lowest_note)
fmax=librosa.note_to_hz(highest_note)
bins_per_octave=12*bins_per_note
filter_scale=2

params = {
          # Backward transform needs to know the signal size.
          'inputSize': len(ac.wf),
          'minFrequency': fmin,
          'maxFrequency': fmax,
          'binsPerOctave': bins_per_octave,
          # Minimum number of FFT bins per CQ channel.
          'minimumWindow': ac.hl
         }


# Forward and backward transforms
constantq, dcchannel, nfchannel = NSGConstantQ(**params)(ac.wf.astype(np.single))
#constantq2 = ConstantQ()(ac.F[0])
y = NSGIConstantQ(**params)(constantq, dcchannel, nfchannel)

ac2 = audio_complete(y,N)
ac2.save(os.path.join(output_path,'back.flac'))

def compress(F,ratio=2):
#    F2 = np.zeros((F.shape[0],F.shape[1]//ratio),dtype=F.dtype)
    for j in range(F.shape[1]):
        for i in range(0,F.shape[0]-ratio,ratio):
#            F[i,j] = np.sum([F[i+k,j] for k in range(ratio)])
            for k in range(1,ratio):
                F[i+k,j] = F[i,j]
#            F[i,j+1] = F[i,j]
    return F
constantq_comp = compress(constantq,ratio=2)
y_comp = NSGIConstantQ(**params)(constantq_comp, dcchannel, nfchannel)

ac3 = audio_complete(y_comp,N)
ac3.save(os.path.join(output_path,'back_skip_2.flac'))

#%% Short Window Extraction Test

#midi_filename = 'Nyan_cat_web_transcription.mid'
midi_filename = 'Runaway.mid'

mid = nsequence(os.path.join(data_path,midi_filename))
ac = audio_complete(mid.render(sf2_path = sf_path), N)
ac.mag

#for j in [6,8,10,15]:
#    for i in range(112):
i=0
j=20
mid = nsequence()
mid.add_note(0,i,67,0,3)
ac= audio_complete(mid.render(),n_fft=4096)
ac_sw = ac.resize(mid.sequence.notes[0].start_time, mid.sequence.notes[0].end_time, 
                          j, attribs=['mag','ph'])
ac_sw.save(os.path.join(output_path,'sw_'+str(j)+'_'+str(i)+'.flac'))


#%% Instrument shift test

output_path = './output/instrument_shift'
#midi_filename = 'Nyan_cat_web_transcription.mid'
midi_filename = 'Runaway.mid'

mid = nsequence(os.path.join(data_path,midi_filename))

for note in mid.sequence.notes:
    if note.instrument == 0:
        note.program = 0
        note.instrument = 2
#        note.end_time+=1
#        note.velocity = 120
#    if note.instrument == 1:
#        note.program = 29
#mid.add_note(19,0,mid.sequence.notes[0].pitch,
#             mid.sequence.notes[0].start_time,mid.sequence.notes[0].end_time,
#             mid.sequence.notes[0].velocity)

#mid.change_program_for_instrument(2,0)
#mid.add_note(2,0,67,0,2)
#mid.change_program_for_instrument(0,15)
#mid.change_program(0,12)
    
ac = audio_complete(mid.render(sf2_path = sf_path), N)

ac.save(os.path.join(output_path,'shifted.flac'))


#%% Test Instrument distribution over data set
path_data = os.path.join('data','lakh_midi')

dm = DataManager(path_data, sets=['training'], types=['midi'])
dm.set_set('training') 

data_dist = np.zeros(112)
valid_filenames=[]
train_size = len(dm.data['training']['midi'])
pb = PB.ProgressBar(train_size)

i=0
for fn in dm:
    pb.check_progress()
    if i>train_size:
        break
    else:
        i+=1
        
    try:
        mid = nsequence(fn[0])
    except:
        continue
#    j=0
    dist_song = np.zeros(112)
    for note in mid.sequence.notes:
#    note = mid.pop()
#    while note is not None:
        if not note.is_drum and note.program<112:
            dist_song[note.program] += 1
#            j+=1
#            if j>500:
#                break;
#        note = mid.pop()
        
    if i<train_size/300 or \
            np.var(data_dist)>=np.var(data_dist+dist_song):
        data_dist+=dist_song
        valid_filenames.append(fn[0])
        
dist_total =np.zeros(112)
for fn in valid_filenames:
    try:
        mid = nsequence(fn)
    except:
        continue
    for note in mid.sequence.notes:
        if not note.is_drum and note.program<112:
            dist_total[note.program] += 1
mean_total = np.mean(dist_total)

pb = PB.ProgressBar(len(valid_filenames))
data_dist = np.zeros(112)
even_more_valid_fn = []
for fn in valid_filenames:
    pb.check_progress()
    
    try:
        mid = nsequence(fn)
    except:
        continue
    dist_song = np.zeros(112)
    for note in mid.sequence.notes:
        if not note.is_drum and note.program<112:
            dist_song[note.program] += 1
    
    if (dist_song[0])<np.sum(dist_song)*0.5: #and\
           #not (np.sum(dist_song[40:43])>np.sum(dist_song)*0.5 and data_dist[40]>mean_total):
        even_more_valid_fn.append(fn)
        data_dist+=dist_song
        
    if (np.sum(dist_song[40:43])>np.sum(dist_song)*0.5 and data_dist[40]>mean_total):
        print(data_dist[40],mean_total)
        print(fn)
        
fig = plt.figure(figsize = (7,3),dpi=300)
plt.bar(range(112),data_dist,color='black')
plt.ylabel('Occurances')
plt.xlabel('Program')
plt.tight_layout()
#plt.show()
fig.savefig('output/statistics/data_dristribution_placeholder.png')
with open('output/statistics/valid_files_from_0_paceholder.txt', 'w') as f:
    for fn in even_more_valid_fn:
        f.write("%s\n" % fn)
        
#valid_filenames = []
#with open('output/statistics/valid_files_from_0.txt', 'r') as f:
#    for fn in f:
#        valid_filenames.append(fn[:-1])
#%%
from shutil import copy2
for fn in even_more_valid_fn:
    copy2(fn,'data/lakh_filtered/training/midi/')


