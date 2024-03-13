import numpy as np
from dataclasses import dataclass

@dataclass
class Parameter():
    value:float
    max_value:float
    min_value:float
    name:str

    def __call__(self) -> float:
        return self.value

class ParameterLin(Parameter):
    def normalize(self, x):
        return (x-self.min_value)/(self.max_value-self.min_value)

    def denormalize(self, x):
        return x*(self.max_value-self.min_value)+self.min_value

class ParameterExp(Parameter):
    def normalize(self):
        pass

    def denormalize(self):
        pass


class ParameterList():
    def __init__(self, *args):
        self.list = []
        for arg in args:
            self.append(arg)
    
    def append(self, parameter):
        self.check(parameter)
        self.list.append(parameter)
    
    def check(self, x):
        if not isinstance(x, Parameter):
            return TypeError()