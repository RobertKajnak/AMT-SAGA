import os


class DataManager():
    def __init__(self, path, 
                 types=('midi','audio'), sets=('training','validation','test'),
                 sort=True):
        """ Class for fetching filenames from the directories.
            The directories should be named  as follows:
                train, test and validation. train and test should contain 
                a folder called midi. If next_pair function is used, they should 
                also a folder called audio.
        """
        self.sets = sets
        self.types = types
        
        self.data={}
        
        if sort:
            l_sort = lambda v: v.upper()
        else:
            l_sort = True
            
        get_filenames = lambda dn1,nd2: sorted([name for name in 
                                os.listdir(os.path.join(path,os.path.join(dn1,nd2)))], 
                                key=l_sort)
        
        for s in sets:
            self.data[s] = {}
            for t in types:
                self.data[s][t] = get_filenames(s,t)
                
        self.set_c = sets[0]
        self.index = 0
        
        
    def switch_type(self, new_set):
        ''' new_type should be one from the ones specified in the constructor
            To check, use .types.keys()
            Also.resets counter  (index)
        '''
        self.set_c = new_set
        self.index = 0
    
    def __iter__(self):
        return self

    def __next__(self):

        """Returns the next Midi file. If there are none left, None is returned"""
        
        cset = self.data[self.set_c]
        if self.index<len(cset[self.types[0]]):
            candidates = []
            for k in cset.keys():
                candidates.append(cset[k][self.index])
            
            cprev = candidates[0]
            for cand in candidates:
                if cand[:cand.rfind('.')].upper() != cprev[:cand.rfind('.')].upper():
                    raise Exception('Pair title mismatch')
                cprev = cand
            self.index +=1
            return candidates
        else:
            self.index = 0
            raise StopIteration
    
