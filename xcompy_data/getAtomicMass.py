# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 22:50:06 2012

@author: daverigie
"""

# getAtomicMass.py
# hard-coded lookup table for atomic mass

def getAtomicMass(elem):
    
    zerostr = (3-len(str(elem)))*'0' 
    f = open('/home/daverigie/PythonProjects/XCOMPY/data/MDATX3.'+ zerostr + str(elem))

    # read the entire text file into a 'fulltext'
    fulltext = f.read().split()
    
    ##################### Extract Element Info from text file ################
    return float(fulltext[1]) # atomic mass
      