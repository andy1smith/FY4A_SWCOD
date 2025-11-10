
'''This is the LBL model written by Mengying Li.
   Include functions that interact with HITRAN. Main iteration program is in the iPython notebook.'''

# import all necessary libraries
import numpy as np
from hapi import *
from LBL_funcs import *

#db_begin('LW/HITRAN data')
#db_begin('SW/HITRAN data')
db_begin('data/hitran_SW')
# # Start of function definitions
#--------------------------------------------------------------------------------
# find Hitran Molecule ID by Molecule name
def MolID_rho(molecule,P,T):
    if (molecule=='H2O'):
        mol_id=1
        M=18 # molecular weight g/mol
    elif (molecule=='CO2'):
        mol_id=2
        M=44
    elif (molecule=='O3'):
        mol_id=3
        M=48
    elif (molecule=='N2O'):
        mol_id=4
        M=44
    elif (molecule == 'CO'):
        mol_id = 5
        M = 28
    elif (molecule=='CH4'):
        mol_id=6
        M=16
    elif (molecule=='O2'):
        mol_id=7
        M=32
    elif (molecule=='N2'):
        mol_id=22
        M=28

    Ru=8.314 # in J/mol*K
    rho=M*P/Ru/T # in g/m^3
    rho/=10**6 # in g/cm^3

    return mol_id,rho
#-------------------------------------------------------------------------------
# downloading data from HITRAN using HITRAN API and store in local folder
def getDataFromHitran(molecule,nu_min,nu_max):
# LAST MODIFIED: Mengying Li 03/07/2017
# INPUTS:
# molecule: name of molecule, e.g. 'H2O', 'CO2'...
# nu_min/nu_max: starting/ending wavenumber, cm-1
# OUTPUTS: store line data in loca directory
# WRITTEN BY: Mengying Li 03/01/2017       
    mol_id,rho=MolID_rho(molecule,1.013*10**5,296)
    fetch(molecule,mol_id,1,nu_min,nu_max)
#---------------------------------------------------------------------------------------
# get absorption coeff of molecules at each layer with abundance equals to 1
def getKappa_SingleLayer(args):
# LAST MODIFIED: Mengying Li 04/20/2017
#INPUTS: args is a list (help with parallel programming)
#molecules: name of molecules (vector)
#nu: vector of wavenumbers, cm-1
#T: temperature, K (scalar)
#P: pressure, Pa (scalar)
#OUTPUT: 
#coeff: volumetric absorption coefficient of each modecules at one layer, cm-1, list of size len(molecules)
#WRITTEN BY: Mengying Li 04/20/2017
    print ('enter getKappa_SingleLayer once.')
    #get the arguments from list args
    molecules=args[0]
    nu=args[1]
    T=args[2]
    P=args[3]
   
    #initialize coeff
    coeff_V=[] # list of coefficients, each element is coeff for one gas
    for i in range(0,len(molecules)): 
        mol_id,mol_rho=MolID_rho(molecules[i],P,T)
        nu_raw,coeff=absorptionCoefficient_Voigt(Components=[(mol_id,1,1)],# List of tuples (M,I,D)
                                                  SourceTables=molecules[i],# list of source tables 
                                                  HITRAN_units=False,# unit in cm-1
                                                  OmegaGrid=nu, # wavenumber grid
                                                  OmegaWing=25, # absolute value of line wing, cm-1
                                                  OmegaWingHW=50,# line wing relative ratio to half width
                                                  #OmegaWing=250, # absolute value of line wing, cm-1
                                                  #OmegaWingHW=500,# line wing relative ratio to half width
                                                  Environment={'p':P/1.013/1e5,'T':T} # in unit cm-1
                                                 )
        #coeff= np.interp(nu, nu_raw, coeff_raw,left=0,right=0) # coeff=0 if out of range
        coeff/=mol_rho # in unit cm2/g # mass absorption coefficient
        coeff_V.append(coeff)
    print ('finish getKappa_SingleLayer once.')
    return coeff_V
#-----------------------------------------------------------------------------------------------
# get absorption coeff of molecules at each layer with abundance equals to 1 and write to file
def getKappa_AllLayers(inputs):
# LAST MODIFIED: Mengying Li 04/20/2017
#INPUTS: 
#model: profile model, e.g. 'tropical'
#N_layer: number of layers
#molecules: name of molecules (vector)
#nu: vector of wavenumbers, cm-1
#ta: temperature of each layer, K (vector)
#pa: pressure of each layer, Pa (vector)
#OUTPUT: 
#coeff_M saved to file: volumetric absorption coefficient of each modecules at each layer, cm-1, list of size N_layer
#WRITTEN BY: Mengying Li 04/20/2017
    N_layer=inputs[0]
    model=inputs[1]
    molecules=inputs[2]
    nu=inputs[3]
    pa=inputs[4]
    ta=inputs[5]

    # create input list of args to func getMixKappa
    list_args=[]
    for i in range(0,N_layer+1):
        args=[molecules,nu,ta[i],pa[i]]
        list_args.append(args)  
    # parallel computing  
    pool = Pool() # workers
    coeff_M = list(pool.map(getKappa_SingleLayer,list_args))
    pool.terminate()

    dnu=nu[1]-nu[0]
    fileName="results/coeffM_"+str(N_layer)+"layers"+"_"+model+"_"+str(dnu)+"cm-1"
    np.save(fileName,coeff_M)
