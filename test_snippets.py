# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 12:05:48 2019

@author: Hesiris
"""


#from magenta.models.onsets_frames_transcription import data

from util_audio import note_sequence as nsequence
from util_audio import midi_from_file

import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display
import soundfile
import scipy
#soundfile.write('stereo_file.flac', data, samplerate, format='flac', subtype='PCM_24')

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
#filename = 'Nyan_Cat_piano.mp3'
#filename = 'Supersonic.mp3'
#filename = 'Fuyu no Epilogue.flac'
filename = 'Utauyo!! MIRACLE.mp3'
#filename = 'Runaway.mp3'

wav,sr = librosa.load(path+filename,sr=None,mono=False,duration=30,offset=20)
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

buckets = 4096
F = librosa.stft(y,center=False,n_fft=buckets)
mag,ph = librosa.core.magphase(F)
ref_mag = np.max(mag)

D = librosa.amplitude_to_db(mag,ref=ref_mag)
plt.figure(figsize=(12, 24))
plt.subplot(5, 1, 1)
librosa.display.specshow(D,y_axis='linear',sr=sr)
plt.ylabel('Linear DB')
plt.colorbar(format='%+2.0f dB')

plt.subplot(5,1,2)
librosa.display.specshow(D, y_axis='log',sr=sr)
plt.ylabel('Log DB')
plt.colorbar(format='%+2.0f dB')

plt.subplot(5,1,3)
mag_mel = librosa.feature.melspectrogram(S=mag**2,n_mels = 512)
librosa.display.specshow(librosa.power_to_db(mag_mel,ref=np.max), y_axis='mel',sr=sr) 
plt.ylabel('Mel')
plt.colorbar(format='%+2.0f dB')

#band_stop(D,12000,18000,50,n_fft=buckets)
plt.subplot(5,1,4)
librosa.display.specshow(D, y_axis='log',sr=sr)
plt.ylabel('Log, removed filter')
plt.colorbar(format='%+2.0f dB')

plt.subplot(5,1,5)
plt.ylabel('Waveform transformed back')
y_back = librosa.istft(librosa.db_to_amplitude(D,ref=ref_mag)*ph)
librosa.display.waveplot(y_back,sr=sr)

soundfile.write(path+'filtered.flac', y_back, sr, format='flac', subtype='PCM_24')


#%% Phase-based Harmonic/Percussive Separation by 
# Estefan´ıa Cano, Mark Plumbley, Christian Dittmar,

#Step 1: calculate S and phi
F = librosa.stft(y,center=False,n_fft=buckets)
S,ph = librosa.core.magphase(F)
ref_mag = np.max(S)

#Other calculations and constants:
fs = sr
N = buckets
H = np.int(N/4)

#Step 2 Dk_low, Dk_high for each freq. bin
dk_low = np.zeros(int(N/2)+1,dtype=np.double)
dk_high = np.zeros(int(N/2)+1,dtype=np.double)

f_inc = fs/N
f_d = f_inc/2
f_k = f_d
 
for i in range(int(N/2)+1):
    dfk = 2*np.pi * H  /fs
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
ph_rad = np.angle(ph)
ph_masked = ph_rad * M


#Step 6
ph_unwrap = np.unwrap(ph_masked)
derivate = lambda a: [0 if i==0 else (a[i]-a[i-1])/2 for i in range(a.shape[0])]
Ph= np.apply_along_axis(derivate,1,ph_unwrap)/2/np.pi

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
#soundfile.write(path+'harmonic.flac', S_harm, sr, format='flac', subtype='PCM_24')
#soundfile.write(path+'percussive.flac', S_perc, sr, format='flac', subtype='PCM_24')

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


#%% File save test
#    y_foreground = librosa.istft(D_harmonic)
#    y_background = librosa.istft(D_percussive)
#    
#    soundfile.write('C:/Code/Python/[Thesis]/Magenta/data/audio/librosa_voice.flac', y_foreground, sr, format='flac', subtype='PCM_24')
#    soundfile.write('C:/Code/Python/[Thesis]/Magenta/data/audio/librosa_background.flac', y_background, sr, format='flac', subtype='PCM_24')
#    librosa.output.write_wav(path + 'sample.wav', sf.samples[0].raw_sample_data,sf.samples[0].sample_rate)
    