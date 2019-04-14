import os


class DataManager():
    def __init__(self, path, 
                 types=['midi','audio'], sets=['training','validation','test'],
                 sort=True):
        """ Class for fetching filenames from the directories.
            args:
                path: parent directory
            types and sets: inside the path directory should be as follows:
                {sets[0]}/{types[0]}, {sets[0]/types[1]}, etc.
            sort: Sorts the contents of the files by filename. If this is set
                to false, undue pair title mismatch exception may be raised
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
        
        
    def set_set(self, new_set):
        ''' new_type should be one from the ones specified in the constructor
            To check, use .types.keys()
            Also.resets counter  (index)
        '''
        if new_set not in self.sets:
            raise ValueError('Specified set not in sets from construction')
        self.set_c = new_set
        self.index = 0
    
    def __iter__(self):
        return self

    def __next__(self):        
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
    
