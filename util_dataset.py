import os


class DataManager():
    def __init__(self, path, sort=False):
        ''' Convert the files into data unseable in tensorflow'''
        
        self.audio = []
        self.midi =[]
        self.audio.append(sorted([name for name in os.listdir(os.path.join(path,'train/audio'))], key=lambda v: v.upper()))
        self.midi.append(sorted([name for name in os.listdir(os.path.join(path,'train/midi'))], key=lambda v: v.upper()))
        self.audio.append(sorted([name for name in os.listdir(os.path.join(path,'test/audio'))], key=lambda v: v.upper()))
        self.midi.append(sorted([name for name in os.listdir(os.path.join(path,'test/midi'))], key=lambda v: v.upper()))
        self.audio.append(sorted([name for name in os.listdir(os.path.join(path,'validation/audio'))], key=lambda v: v.upper()))
        self.type = 0;
        self.index = 0;
        
        if sort:
            for au in self.audio:
                au.sort()
            for mid in self.midi:
                mid.sort()
        
    def switch_type(self, type):
        ''' type:
                0: training
                1: test
                2: validations
            also resets counter  (index)
        '''
        self.type=type
        self.index = 0
        
    def next_pair(self):
        
        if (self.index<len(self.audio[self.type])):
            audio_candidate = self.audio[self.type][self.index]
            midi_candidate = self.midi[self.type][self.index]
            if audio_candidate[:audio_candidate.rfind('.')].upper() == midi_candidate[:midi_candidate.rfind('.')].upper():
                self.index += 1
                return audio_candidate,midi_candidate
            else:
                raise Exception('Pair title mismatch')
        else:
            return (None,None)
        