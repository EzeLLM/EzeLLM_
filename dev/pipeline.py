import toml
import os
import sys
from get_model import download_file
from ezellm import EzeLLM , EzeLLMConfig
from typing import List
from torch import nn
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

MEMORY_NAME_CONSTANT='model_path'
CONFIG_PATH='config/config.toml'

class pipeline():
    def __init__(self):
        self.model_path=self._get_model()
        self.model: EzeLLM =self._load_model()


    def _get_model(self):
        _memory = toml.load('config/memory.toml')
        if MEMORY_NAME_CONSTANT in _memory.keys():
            if _memory[MEMORY_NAME_CONSTANT].endswith('.pt') and os.path.exists(_memory[MEMORY_NAME_CONSTANT]):
                return _memory[MEMORY_NAME_CONSTANT]
        else:
            _config = toml.load(CONFIG_PATH)
            _path = download_file(_config['model_url'])
            _memory[MEMORY_NAME_CONSTANT] = _path
            toml.dump(_memory,open('config/memory.toml','w'))
            return 
    
    def _load_model(self):
        return EzeLLM.from_pretrained(self.model_path)
    
    def generate(
            self,
            input_: str= "I'm a",
            tempreature: int = 0.7,
            tempreature_interval:int = 0.1,
            topk: int = 50,
            topp: float = 0.9,
            max_l: int = 200,
            num_return_seq: int = 1

    ) -> List:
        return self.model.generate(input_,tempreature,tempreature_interval,topk,topp,max_l,num_return_seq)
    
if __name__ == '__main__':
    p=pipeline()
    while True:
        print(p.generate(input_=input("Enter the input text: "),tempreature=0.9,tempreature_interval=0.1,topk=50,topp=0.9,max_l=200,num_return_seq=1))
