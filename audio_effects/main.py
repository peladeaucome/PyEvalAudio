import numpy as np
from dataclasses import dataclass, InitVar, field

class AudioEffect():
    def __init__(self, samplerate:int=44100):
        self.samplerate=samplerate

    def process(self, x):
        return x
    
    def __call__(self, x):
        return self.process(x)

class EffectSeries(AudioEffect):
    def __init__(self, *effects_arg, samplerate=44100):
        super().__init__(samplerate)
        self.effects_list:list=[]

        for effect in effects_arg:
            self.append(effect)
    
    def process(self, x):
        out=x.copy()
        for effect in self.effects_list:
            out=effect(x)
    
    def append(self, effect):
        self.check(effect)
        self.effects_list.append(effect)
    
    def check(self, effect):
        if effect.samplerate!=self.samplerate:
            raise ValueError(f"Wrong samplerate: {effect.samplerate}. Expected: {self.samplerate}")
            