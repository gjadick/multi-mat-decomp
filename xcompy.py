# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 10:32:08 2012

@author: daverigie
"""
# XCOMPY.py Module contains all of the functions to access attenuation data from NIST

# import pylab as py
import numpy as np
import re
from scipy.interpolate import interp1d as interp
import os

rootpath = os.path.dirname(__file__)
if os.path.exists(rootpath) == False:
    print('unable to find root directory, using:')
    print(rootpath)

def elematten(elem,E=np.array([-1]),attentype = 'total'):
    """

    Finds the attenuation for an element queried by atomic number
    user should input atomic number, energy in (eV), and the type of attenuation.
    The output will be in barnes/atom.

    Example: yy = elematten(Z,Energies,attentype)

    INPUTS:
    Z - integer representing atomic number
    Energies - numpy array of energies in eV
    attentype - String representing attenuation type:
             'total','coh','incoh','pe','ppelec','ppnuc'

    OUTPUTS:
    yy - numpy array of attenuation values in barnes/atom


    The attenuation values are calculated based on elemental data files I have
    copied from the old Fortran-77 NIST XCOM program. I contacted Stephen Seltzer
    in June 2012, and he confirmed that this data should still be as up to date as
    what is currently used on the NIST XCOM web interface.

    """

    zerostr = (3-len(str(elem)))*'0'
    f = open(rootpath +'/input/xcompy_data/MDATX3.'+ zerostr + str(elem))

    # read the entire text file into a 'fulltext'
    fulltext = f.read().split()

    ##################### Extract Element Info from text file ################
    Z = int(fulltext[0]) # atomic number
    AMASS = float(fulltext[1]) # atomic mass
    NUMSHELLS = int(fulltext[2]) # number of subshells (absorption edges)
    NUME = int(fulltext[3])  # number of energies in standard grid
    plc = 4

    if NUMSHELLS>0:
        EB4SHELL = fulltext[4:(4+NUMSHELLS)] #number of energies up to each shell(inclusive)
        EB4SHELL = [int(x) for x in EB4SHELL]
        plc += NUMSHELLS
        SHELLNAMES = fulltext[plc:plc+NUMSHELLS]
        plc += NUMSHELLS

    SHELLENERGIES = fulltext[plc:plc+NUMSHELLS] # in eV
    SHELLENERGIES = np.array([float(x) for x in SHELLENERGIES])
    plc += NUMSHELLS

    ENERGIES = fulltext[plc:plc + NUME] # in eV
    ENERGIES = np.array([float(x) for x in ENERGIES])
    # Add .01 eV to consecutive identical energy values to avoid interpolation problem
    for i in range(len(ENERGIES)-1):
        E1 = ENERGIES[i]
        E2 = ENERGIES[i+1]
        if E2==E1:
            ENERGIES[i+1] += 1

    plc += NUME

    COH = fulltext[plc:plc + NUME] # Coherent Scattering cross-section in barnes/atom
    COH = np.array([float(x) for x in COH])
    plc += NUME

    INCOH = fulltext[plc:plc + NUME] # Incoherent cross-section in barnes/atom
    INCOH = np.array([float(x) for x in INCOH])
    plc += NUME

    PE = fulltext[plc:plc + NUME] # Photo-electric cross-section in barnes/atom
    PE = np.array([float(x) for x in PE])
    plc += NUME

    PPNUC = fulltext[plc:plc + NUME] # Pair Production (electron) cross-section in barnes/atom
    PPNUC = np.array([float(x) for x in PPNUC])
    plc += NUME

    PPELEC = fulltext[plc:plc + NUME] # Pair Production (nuclear) cross-section in barnes/atom
    PPELEC = np.array([float(x) for x in PPELEC])
    plc += NUME

    if NUMSHELLS>0:

        NUMSHELLS2 = int(fulltext[plc]) # Number of subshells (should be same as before?)
        plc += 1

        EB5SHELL = fulltext[plc:plc+NUMSHELLS2] # number of energies from shell energy up until next shell (not inclusive)
        EB5SHELL = np.array([int(x) for x in EB5SHELL])
        plc += NUMSHELLS2

        SHELLENERGIES2 = fulltext[plc:plc+sum(EB5SHELL)] # These are in MeV
        SHELLENERGIES2 = np.array([float(x) for x in SHELLENERGIES2])
        plc += sum(EB5SHELL)

        TOTAL_NO_INCOH = fulltext[plc:plc+sum(EB5SHELL)] # barnes/atom
        TOTAL_NO_INCOH = np.array([float(x) for x in TOTAL_NO_INCOH])
        plc += sum(EB5SHELL)

    ###################### debugging #############################

    if len(E)==1 and E[0]==-1:
        E = ENERGIES

    x = ENERGIES
    y = np.zeros(len(x))
    ynew = np.zeros(len(E))

    if (attentype == 'coh') or (attentype=='incoh'):
        x = np.log(x)
        if attentype == 'coh':
            y = COH.copy()
            ytemp = COH
        elif attentype == 'incoh':
            y = INCOH.copy()
            ytemp = INCOH

        xmin = np.min(x[y>0])
        xmax = np.max(x[y>0])

        y[y>0] = np.log(y)

        f = interp(x,y,kind='linear')


        xnew = np.log(E)
        ynew[(xnew>xmin)&(xnew<xmax)] = np.exp(f(xnew[(xnew>xmin)&(xnew<xmax)]))

    elif (attentype == 'ppnuc') or (attentype == 'ppelec'):
        if attentype == 'ppnuc':
           ETHRESH = 1.022e6
           PPTEMP = PPNUC
        elif attentype == 'ppelec':
            ETHRESH = 2.044e6
            PPTEMP = PPELEC

        y[x>ETHRESH] = np.log(pow((1-ETHRESH/x[x>ETHRESH]),3)*PPTEMP[x>ETHRESH])
        y[x<ETHRESH] = 0

        f = interp(x,y,kind='linear')

        xnew = E
        ynew[xnew>ETHRESH] = np.exp(f(xnew)[xnew>ETHRESH])/pow(1-ETHRESH/xnew[xnew>ETHRESH],3)

    elif (attentype == 'pe') or (attentype == 'total'):
        if (attentype=='pe'):
            COEFF = PE.copy()
        elif (attentype == 'total'):
            COEFF = (PE + COH + INCOH + PPNUC + PPELEC).copy()

        ELIMITS = np.append(np.min(ENERGIES),SHELLENERGIES[::-1])
        ELIMITS = np.append(ELIMITS,np.max(ENERGIES))

        xnew = np.log(E)

        for i in range(len(ELIMITS)-1):
            Emin = ELIMITS[i]
            Emax = ELIMITS[i+1]
            Esub = ENERGIES[(ENERGIES>=Emin)&(ENERGIES<Emax)]
            COEFFsub = COEFF[(ENERGIES>=Emin)&(ENERGIES<Emax)]

            x = np.log(Esub)
            y = np.log(COEFFsub)

            if i==NUMSHELLS:
                ff = interp(x,y,kind='cubic',bounds_error=False)
            else:
                ff = interp(x,y,kind='linear',fill_value='extrapolate')

            ynew[(E>=Emin)&(E<Emax)] = np.exp(ff(xnew[(E>=Emin)&(E<Emax)]))

    return np.array(ynew)




# Returns the attenuation for a compound
# ENERGY IS keV!!!
# Attenuation is g/cm^2!!
def cmpatten(name,E,attentype='total'):
    """

    Finds the attenuation for a compound specified by chemical formula.

    Example: yy = elematten(formula,Energies,attentype)

    INPUTS:
    formula - String specifying chemical formula
    material name - use * as the first character then type the name. See
                    lookuptable.csv
    Energies - numpy array of energies in keV
    attentype - String representing attenuation type:
             'total','coh','incoh','pe','ppelec','ppnuc'

    OUTPUTS:
    yy - numpy array of attenuation values in g/cm^2


    Based on elematten

    """

    E = np.array(E,dtype=float)

    if name[0]=='*':
        name = getFormula(name[1::])

    Elems = re.findall('[A-Z][^A-Z]*',name)
    Quant = [0]*len(Elems)
    Mass = [0]*len(Elems)
    for i in range(len(Elems)):
        if Elems[i][-1].isdigit():
            quant_re = re.compile('\d+\.?\d*')
            elem_re = re.compile('[^0-9^\.]+')
            Quant[i] = float(quant_re.findall(Elems[i])[0])
            Elems[i]= elem_re.findall(Elems[i])[0]
        else:
            Quant[i] = 1
        Mass[i] = getAtomicMass(parseElement(Elems[i]))

    Mass = np.array(Mass)
    Quant = np.array(Quant)

    totalMass = sum(Mass*Quant)
    MassFraction = Mass*Quant/totalMass

    yy = 0*E

    for i in range(len(Elems)):
        mu_rho = elematten(parseElement(Elems[i]),E*1e3,attentype)
        yy += (MassFraction[i]*mu_rho)*(1e-24)*(6.022e23/Mass[i]) #converts to cm^2/g from barnes/atom

    return np.array(yy)

# Returns the attenuation for a mixture 'form1(frac1)form2(frac2)...'
# ENERGY IS keV!!!
# Attenuation is g/cm^2!!
def mixatten(mixture,E,attentype='total'):
    """

    Finds the attenuation for a mixture specified by chemical formulae and mass
          fractions.

    Example: yy = elematten(mixstring,Energies,attentype)

    INPUTS:
    mixstring - String specifying chemical formulae and mass fractions
                ie) 'I2(0.04)H2O(0.96)'
    Energies - numpy array of energies in keV
    attentype - String representing attenuation type:
             'total','coh','incoh','pe','ppelec','ppnuc'

    OUTPUTS:
    yy - numpy array of attenuation values in g/cm^2


    Based on elematten

    """

    # Check if input is an entry in the lookup table if it starts with an asterisk
    if mixture[0:7].lower() == 'Toshiba'.lower():
        return toshibaatten(mixture[8::],E)

    if mixture[0]=='*':
        mixture = getFormula(mixture[1::])

    ss = mixture
    cmps = []
    weights = []

    ii = 0
    sub = mixture
    lp = sub.find('(')
    rp = sub.find(')')

    yy=0*E

    while lp != -1:
        cmps = cmps + [sub[0:lp]]
        weights = weights + [float(sub[(lp+1):rp])]
        ii = rp+1
        sub = sub[ii:]
        lp = sub.find('(')
        rp = sub.find(')')

        yy = 0*E

    # If input is not a mixture, try treating it as a single compound
    if len(weights)==0:
        return cmpatten(sub,E,attentype)

    weights = np.array(weights)
    weights = weights/float(sum(weights)) #normalize the mass fractions so they add to 1
    for i in range(len(weights)):
        yy += cmpatten(cmps[i],E,attentype)*weights[i]

    return np.array(yy)


# returns the atomic mass for an element specified by its atomic number
def getAtomicMass(elem):
    """
    Finds the atomic mass of an element in g/mole.
        Example: amass = getAtomicMass(Z)

    INPUTS:
    Z - atomic number (integer)

    OUTPUTS:
    amass - double specifiying atomic mass in grams per mole
    """
    f = open(rootpath +'/input/xcompy_data/MDATX3.'+'%03d'%elem)
    return float(f.readline().split()[1])


def gridenergies(elem):
    """

    Returns the standard grid of energies in eV for an elem specified by atomic
    number

    Example: EE = gridenergies(Z)

    INPUTS:
    Z - integer representing atomic number

    OUTPUTS:
    EE - numpy array of standard energies in eV

    """

    zerostr = (3-len(str(elem)))*'0'
    f = open(rootpath + '/input/xcompy_data/MDATX3.'+ zerostr + str(elem))

    # read the entire text file into a 'fulltext'
    fulltext = f.read().split()

    ##################### Extract Element Info from text file ################
    Z = int(fulltext[0]) # atomic number
    AMASS = float(fulltext[1]) # atomic mass
    NUMSHELLS = int(fulltext[2]) # number of subshells (absorption edges)
    NUME = int(fulltext[3])  # number of energies in standard grid
    plc = 4

    if NUMSHELLS>0:
        EB4SHELL = fulltext[4:(4+NUMSHELLS)] #number of energies up to each shell(inclusive)
        EB4SHELL = [int(x) for x in EB4SHELL]
        plc += NUMSHELLS
        SHELLNAMES = fulltext[plc:plc+NUMSHELLS]
        plc += NUMSHELLS

    SHELLENERGIES = fulltext[plc:plc+NUMSHELLS] # in eV
    SHELLENERGIES = np.array([float(x) for x in SHELLENERGIES])
    plc += NUMSHELLS

    ENERGIES = fulltext[plc:plc + NUME] # in eV
    ENERGIES = np.array([float(x) for x in ENERGIES])

    return ENERGIES


def getRho(matname):

    # read file  into 2d array of strings
    f = open(rootpath +'/input/xcompy_data/lookuptable.txt')
    fulltext = f.read()
    fulltext = fulltext[0:-1]
    fulltext = fulltext.split('\n')
    fulltext = np.array([x.split('\t') for x in fulltext])

    if matname[0]=='*':
        matname = matname[1::]

    #find row number of given material
    names = fulltext[:,0].tolist()
    if (not (matname in names)):
        return -1
    row = names.index(matname)

    return float(fulltext[row,1])


def getFormula(matname):

    # read file  into 2d array of strings
    f = open(rootpath +'/input/xcompy_data/lookuptable.txt')
    fulltext = f.read()
    fulltext = fulltext[0:-1]

    if fulltext.find('\r\n')!=-1:
        fulltext = fulltext[0:-1]
        fulltext = fulltext.split('\r\n')
    else:
        fulltext = fulltext.split('\n')

    fulltext = np.array([x.split('\t') for x in fulltext])

    #find row number of given material
    names = fulltext[:,0].tolist()
    if (not (matname in names)):
        return -1
    row = names.index(matname)

    return fulltext[row,2]


# returns the atomic number when the chemical name is queried
def parseElement(symbol):
    """
    Finds the atomic number for an atom specified by element symbol

    Example: Z = parseElement(symbol)

    INPUTS:
    name - 1 or 2 character String specifying element name

    OUTPUTS:
    Z - integer representing atomic number
    """


    symbols = ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si',\
    'P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',\
    'Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh',\
    'Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd',\
    'Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re',\
    'Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th',\
    'Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm']

    return symbols.index(symbol) + 1
