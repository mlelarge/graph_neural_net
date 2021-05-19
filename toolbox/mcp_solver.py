import os
import threading
import time
import torch
import numpy as np
import random
import toolbox.utils as utils

from string import ascii_letters,digits
NAME_CHARS = ascii_letters+digits

class Thread_MCP_Solver(threading.Thread):
    def __init__(self, threadID, adj, name=''):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.adj = adj
        if name=='':
            name = ''.join(random.choice(NAME_CHARS) for _ in range(10))
        self.name = name
        self.fwname = name+'.mcp'
        self.frname = name+'.mcps'
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
        with open(self.fwname,'r') as f:
            data = f.readlines()
        cliques = []
        for i,line in enumerate(data):
            cur_data = {int(elt) for elt in line.split(' ')}
            cliques.append(cur_data)
        self.solutions = cliques
    
    def clear(self,erase=True):
        if erase:
            os.remove(self.frname)
            os.remove(self.fwname)
            os.remove(self.flname)


    def run(self):
        self._write_adj()
        os.system(f"./mcp_solver.exe -v {self.fwname} >> {self.flname}")
        self._read_adj()
        self.done = True



class MCP_Solver():
    def __init__(self,adjs=None, max_threads=8):
        utils.check_dir('./tmp_mcp')
        if adjs is None:
            self.adjs = []
        self.adjs = adjs
        assert max_threads>0, "Thread number put to 0."
        self.max_threads = max_threads
        self.threads = [None for _ in range(self.max_threads)]
        self.solutions  = []
    
    @classmethod
    def from_data(adjs,max_threads=8):
        return MCP_Solver(adjs,max_threads)
    
    def load_data(self,adjs):
        self.adjs = adjs
    
    @property
    def n_threads(self):
        return np.sum([thread is not None for thread in self.threads])
    
    def no_threads_left(self):
        return np.sum([thread is None for thread in self.threads])==self.max_threads
    
    def is_thread_available(self,i):
        return self.threads[i] is None
    
    def clean_threads(self,erase=True):
        for i,thread in enumerate(self.threads):
            if thread is not None and thread.done:
                id = thread.threadID
                print(f"Solution {id} on thread {i} is done.")
                self.solutions[id] = thread.solutions
                thread.clear(erase=erase)
                self.threads[i] = None
    
    def reset(self,bs):
        self.solutions = [list() for _ in range(bs)]
        self.threads = [None for _ in range(self.max_threads)]
    
    def solve(self):
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
                    new_thread = Thread_MCP_Solver(counter,adj,name=f'./tmp_mcp/tmp-mcp-{counter}')
                    #print(f"Putting problem {counter} on thread {thread_slot}")
                    self.threads[thread_slot] = new_thread
                    new_thread.start()
                    counter+=1
            self.clean_threads()
        


if __name__=='__main__':
    def test_mcp_solver(bs,n,max_threads=4):
        adjs = torch.empty((bs,n,n)).uniform_()
        adjs = (adjs.transpose(-1,-2)+adjs)/2
        adjs = (adjs<(0.5)).to(int)
        mcp_solver = MCP_Solver(adjs,max_threads)
        mcp_solver.solve()
        clique_sols = mcp_solver.solutions
        return clique_sols
    
    n=50
    t0 = time.time()
    [test_mcp_solver(10,n,max_threads=4) for _ in range(10)]
    print("Time taken :", time.time()-t0)
