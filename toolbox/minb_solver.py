import os
import threading
import time
import torch
import numpy as np
import random
import toolbox.utils as utils
from loaders.data_generator import SBM_Generator

from string import ascii_letters,digits
NAME_CHARS = ascii_letters+digits

class Thread_MinB_Solver(threading.Thread):
    def __init__(self, threadID, adj, name=''):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.adj = adj
        if name=='':
            name = ''.join(random.choice(NAME_CHARS) for _ in range(10))
        self.name = name
        self.fwname = name+'.minb'
        self.frname = name+'.minbs'
        self.flname = name+'.log'
        self.solutions = []
        self.done=False

    
    def _write_adj(self):
        with open(self.fwname,'w') as f:
            for row in self.adj:
                line = ""
                for value in row:
                    line+=f"{int(value)} "
                line = line[:-1] + "\n"
                f.write(line)

    def _read_adj(self):
        partitions = []
        with open(self.frname,'r') as f:
            data = f.readlines()
        for line in data:
            cur_data = {int(elt) for elt in line.split(' ')}
            partitions.append(cur_data)
        self.solutions = partitions
    
    def clear(self,erase_mode='all'):
        if erase_mode=='all':
            os.remove(self.frname)
            os.remove(self.fwname)
            os.remove(self.flname)
        elif erase_mode=='i':
            os.remove(self.flname)
            os.remove(self.fwname)
        elif erase_mode=='io':
            os.remove(self.fwname)
            os.remove(self.frname)
        elif erase_mode=='ol':
            os.remove(self.flname)
            os.remove(self.frname)

    def run(self):
        self._write_adj()
        os.system(f"./minb_solver.exe -v {self.fwname} >> {self.flname}")
        self._read_adj()
        self.done = True



class MinB_Solver():
    def __init__(self,adjs=None, max_threads=8,path='tmp_minb/',erase_mode ='all'):
        utils.check_dir(path)
        self.path = path
        if adjs is None:
            self.adjs = []
        self.adjs = adjs
        assert max_threads>0, "Thread number put to 0."
        self.max_threads = max_threads
        self.threads = [None for _ in range(self.max_threads)]
        self.solutions  = []
        self.erase_mode=erase_mode
    
    @classmethod
    def from_data(adjs,max_threads=8):
        return MinB_Solver(adjs,max_threads)
    
    def load_data(self,adjs):
        self.adjs = adjs
    
    @property
    def n_threads(self):
        return np.sum([thread is not None for thread in self.threads])
    
    def no_threads_left(self):
        return np.sum([thread is None for thread in self.threads])==self.max_threads
    
    def is_thread_available(self,i):
        return self.threads[i] is None
    
    def clean_threads(self):
        for i,thread in enumerate(self.threads):
            if thread is not None and thread.done:
                id = thread.threadID
                print(f"Solution {id} on thread {i} is done.")
                self.solutions[id] = thread.solutions
                thread.clear(erase_mode=self.erase_mode)
                self.threads[i] = None
    
    def reset(self,bs):
        self.solutions = [list() for _ in range(bs)]
        self.threads = [None for _ in range(self.max_threads)]
    
    def solve(self):
        exp_name = ''.join(random.choice(NAME_CHARS) for _ in range(10))

        solo = False
        adjs = self.adjs.detach().clone()
        if len(adjs.shape)==2:
            solo = True
            adjs = adjs.unsqueeze(0)
        bs,n,_ = adjs.shape
        self.reset(bs)

        counter = 0
        while counter<bs or not self.no_threads_left():
            for thread_slot in range(self.max_threads):
                if counter <bs and self.is_thread_available(thread_slot):
                    adj = adjs[counter]
                    new_thread = Thread_MinB_Solver(counter,adj,name=os.path.join(self.path,f'tmp-minb-{counter}-{exp_name}'))
                    #print(f"Putting problem {counter} on thread {thread_slot}")
                    self.threads[thread_slot] = new_thread
                    new_thread.start()
                    counter+=1
            self.clean_threads()
        


if __name__=='__main__':
    def test_minb_solver(bs,n,max_threads=4):
        gen_args = {
            'num_examples_train': 10,
            'num_examples_val': 10,
            'num_examples_test': 10,
            'n_vertices': n,
            'path_dataset': 'dataset_sbm',
            'p_inter': 10,
            'p_outer': 5,
            'alpha': 0.5
        }
        gen = SBM_Generator("train",gen_args)
        gen.load_dataset()
        adjs = torch.zeros((bs,n,n))
        for k,(cur_data,_) in enumerate(gen.data):
            adjs[k,:,:] = cur_data[:,:,1]
        mcp_solver = MinB_Solver(adjs,max_threads)
        mcp_solver.solve()
        clique_sols = mcp_solver.solutions
        return clique_sols
    
    n=100
    t0 = time.time()
    [test_minb_solver(10,n,max_threads=4) for _ in range(10)]
    print("Time taken :", time.time()-t0)
