# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 16:29:35 2012

@author: daverigie
"""
import numpy as np
from readdat import readdat

def test():
    readdat(35,E=np.logspace(3,11,num=100,base=10),attentype='incoh')
    return