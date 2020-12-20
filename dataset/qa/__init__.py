# -*- coding: utf-8 -*-


import os
import numpy as np
from .data_reader import DataReader

def setup(opt):
    
    reader = DataReader(opt)  

    res = opt.get_parameter_list()
    return reader




