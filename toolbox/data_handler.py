import pandas as pd
import os
from toolbox.utils import check_file
from collections import namedtuple,deque

class DataHandler():
    
    def __init__(self,filepath) -> None:
        self.filepath = filepath
        self.data = pd.DataFrame()
        self.read(filepath)
    
    def write(self,filepath=''):
        if filepath=='':
            filepath = self.filepath
        self.data.to_csv(filepath, index=False)
        print(f'Written to {filepath}')
    
    def read(self,filepath=''):
        if filepath=='':
            filepath = self.filepath
        else:
            self.filepath = filepath
        
        if os.path.exists(filepath):
            try:
                self.data = pd.read_csv(filepath)
            except pd.errors.EmptyDataError:
                self.data = pd.DataFrame()
        else:
            check_file(filepath)
            self.data = pd.DataFrame()

    def new_column(self,name)->None:
        if not name in self.data.columns:
            self.data.assign(col_name=name)

    def new_columns(self,names)->None:
        for name in names:
            self.new_column(name)
    
    def exists(self, column_name, value) -> bool:
        return (column_name in self.data.columns) and ( (self.data[column_name]==value).values.any() )
    
    def line_exists(self,line:dict) -> bool:
        df = self.data
        if df.empty:
            return False
        exists = True
        for key in line.keys():
            df = df.loc[df[key]==line[key]]
            if df.empty:
                exists = False
                break
        return exists
    
    def to_do(self, column_name, value):
        exists = self.exists(column_name,value) #Checks if a corresponding line exists
        # The second part checks if this (these ? It doesn't check there is only one line) line contains a missing value
        return (not exists) or self.data.loc[self.data[column_name]==value].isnull().values.any()
    
    def add_entry(self, line : dict, save=True) -> None:
        self.new_columns(line.keys())
        if not self.line_exists(line):
            self.data = self.data.append(line, ignore_index=True)
        if save:
            self.write()
    
    def add_entry_with_value(self, value_key, value, line:dict,  save=True) -> None:
        if self.to_do(value_key,value):
            if not self.exists(value_key, value):
                line[value_key] = value
                self.add_entry(line = line, save = save)
            else:
                for key in line.keys():
                    self.data.loc[self.data[value_key]==value, key] = line[key]



Task = namedtuple('Task',['column_name','value'])

class Planner(DataHandler):
    def __init__(self, filepath) -> None:
        super().__init__(filepath)
        self.tasks = deque()
    
    @property
    def n_tasks(self):
        return len(self.tasks)
    
    def add_task(self, task)->None:
        if self.to_do(*task):
            self.tasks.append(Task(*task))
    
    def add_tasks(self, tasks) -> None:
        """tasks should be a list of tuples (column_name, value)"""
        for task in tasks:
            self.add_task(*task)

    def has_tasks(self):
        return len(self.tasks)!=0

    def next_task(self):
        if len(self.tasks)==0:
            return ()
        return self.tasks.pop()