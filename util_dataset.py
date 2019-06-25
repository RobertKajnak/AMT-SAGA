import os

def get_latest_file(directory, substring =''):
    """Returns the full path to the file that was modified, from the 
    directory, that also contains the substring specified.
    params:
        directory: the directory to be searched
        substring: the files must contain this substring. Specify emptystring
            to consider all files
    returns:
        full path to the file. If no matches are found, None is returned"""
    
    all_files = os.listdir(directory)
    all_files = list(filter(lambda s: substring in s, all_files))
    all_files = [os.path.join(str(directory),fn) for fn in all_files]
    all_files.sort(key=os.path.getmtime)
    if len(all_files)>0:
        return all_files[-1]
    else:
        return None

class DataManager():
    def __init__(self, path, 
                 types=['midi','audio'], sets=['training','validation','test'],
                 sort=True, return_full_path_on_iteration = True):
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
        self.path = path
        self.is_full_path = return_full_path_on_iteration
        
        self.data={}
        
        if sort:
            l_sort = lambda v: v.upper()
        else:
            l_sort = True
            
        get_filenames = lambda dn1,nd2: sorted([name for name in 
                                os.listdir(os.path.join(path,dn1,nd2))], 
                                key=l_sort)
        #s = set; t =type
        for s in sets:
            self.data[s] = {}
            for t in types:
                self.data[s][t] = get_filenames(s,t)
                
        #set_current
        self.set_c = sets[0]
        self.index = 0
        
    def _append_path(self,s,t,fn):
        return os.path.join(self.path,s,t,fn)
        
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
        #get pointer to the dict containing the current set
        cset = self.data[self.set_c]
        
        if self.index<len(cset[self.types[0]]):
            candidates = []
            #get all the filenames for all the folder types in current set for the first name
            for k in cset.keys():
                if self.is_full_path:
                    candidates.append(os.path.join(self.path,self.set_c,k,cset[k][self.index]))
                else:
                    candidates.append(cset[k][self.index])
            
            #Check if last part of the path matches
            cprev = os.path.split(candidates[0])[-1]
            for full_cand in candidates:
                cand = os.path.split(full_cand)[-1]
                if cand[:cand.rfind('.')].upper() != cprev[:cprev.rfind('.')].upper():
                    raise Exception('Pair title mismatch')
                cprev = cand
            self.index +=1
            return candidates
        else:
            self.index = 0
            raise StopIteration
    
