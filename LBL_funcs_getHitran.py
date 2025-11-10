"""
Functions for interfacing with HITRAN.

Author: Mengying Li

"""

import numpy as np
from multiprocessing import Pool
import hapi

__all__ = [
    "MolID_rho",
    "getDataFromHitran",
    "getKappa_SingleLayer",
    "getKappa_AllLayers",
]


# find Hitran Molecule ID by Molecule name
def MolID_rho(molecule, P, T):
    """Get HITRAN molecule ID and density.

    Get the HITRAN molecule ID based on the molecule name, and calculate the
    molecule's density.

    Parameters
    ----------
    molecule : str
        Molecule name.
    P : float
        Pressure [Pa].
    T : float
        Temperature [K].

    Returns
    -------
    mol_id : int
        HITRAN molecule ID.
    rho : float
        Density [g/cm^3].

    """

    if molecule == "H2O":
        mol_id = 1
        M = 18  # molecular weight g/mol
    elif molecule == "CO2":
        mol_id = 2
        M = 44
    elif molecule == "O3":
        mol_id = 3
        M = 48
    elif molecule == "N2O":
        mol_id = 4
        M = 44
    elif molecule == "CO":
        mol_id = 5
        M = 28
    elif molecule == "CH4":
        mol_id = 6
        M = 16
    elif molecule == "O2":
        mol_id = 7
        M = 32
    elif molecule == "N2":
        mol_id = 22
        M = 28

    Ru = 8.314  # [J/(mol * K)]
    rho = M * P / Ru / T  # [g/m^3]
    rho /= 1e6  # [g/cm^3]
    return mol_id, rho


def getDataFromHitran(molecule, nu_min, nu_max, spectral):
    """Download data from HITRAN.

    Download data from HITRAN using the HITRAN API (hapi) and then store the
    results in a local folder.

    Parameter
    ---------
    molecule : str
        Name of the molecules (e.g. "H2O", "CO2").
    nu_min, nu_max : float
        Minimum and maximum wavenumber [cm^-1].
    spectral : str
        Considered spectral, either 'LW' or 'SW'.

    Return
    ------
    None

    """
    data_dir="data/hitran_"+spectral
    hapi.db_begin(data_dir)
    mol_id, rho = MolID_rho(molecule, 1.013 * 1e5, 296)
    hapi.fetch(molecule, mol_id, 1, nu_min, nu_max)


def getKappa_SingleLayer(args):
    """Get single layer absorption coefficients.

    Get the absoprtion coefficients of the M molecules for a single layer.

    Parameters
    ----------
    args : (4,) array_like
        0) molecules : (M,) array_like
            The names of the M molecules.
        1) nu : (N_nu,) array_like
            Wavenumbers [cm^-1].
        2) T :
            Temperature [K].
        3) P :
            Pressure [Pa].

    Returns
    -------
    coeff_V : (M,) array_like
        Spectral volumetric absorption coefficients [cm^-1] of each of the M molecules,
        at one layer.

    """
    # get the arguments from list args
    molecules = args[0]
    nu = args[1]
    T = args[2]
    P = args[3]

    # initialize coeff
    coeff = np.zeros((1, len(nu)))
    coeff_V = []  # list of coefficients, each element is coeff for one gas
    for i in range(0, len(molecules)):
        mol_id, mol_rho = MolID_rho(molecules[i], P, T)
        nu_raw, coeff = hapi.absorptionCoefficient_Lorentz(
        #nu_raw, coeff = hapi.absorptionCoefficient_Voigt( # Voigt is much slower
            Components=[(mol_id, 1, 1)],  # List of tuples (M,I,D)
            SourceTables=molecules[i],  # list of source tables
            HITRAN_units=False,  # unit in cm-1
            OmegaGrid=nu,  # wavenumber grid
            OmegaWing=25,  # absolute value of line wing, cm-1
            OmegaWingHW=50,  # line wing relative ratio to half width
            Environment={"p": P / 1.013 / 1e5, "T": T},  # in unit cm-1
        )
        coeff /= mol_rho  # in unit cm2/g # mass absorption coefficient
        coeff_V.append(coeff)
    return coeff_V


def getKappa_AllLayers(inputs):
    """

    Get the absoprtion coefficients of the M molecules for N layers.

    Parameters
    ----------
    inputs :(7,) array_like
        N_layer: int
            number of atmospheric layers.
        model: str
            profile model, e.g. 'tropical'
        molecules : (M,) array_like
            The names of the M molecules.
        nu : (N_nu,) array_like
            Wavenumbers [cm^-1].
        ta : (N_layer+1,) array_like
            Averaged temperature of each layer [K].
        pa : (N_layer+1,) array_like
            Averaged pressure of each layer [K].
        spectral: str
            Considered spectral, either 'LW' or 'SW'.

    Returns
    -------
    None. Save the coefficients to local directory for reuse.

    """
    N_layer=inputs['N_layer']
    model=inputs['model']
    molecules=inputs['molecules']
    nu=inputs['nu']
    pa=inputs['pa']
    ta=inputs['ta']
    spectral=inputs['spectral']

    data_dir="data/hitran_"+spectral
    hapi.db_begin(data_dir)
    # create input list of args to func getMixKappa
    list_args=[]
    for i in range(0,N_layer+1):
        args=[molecules,nu,ta[i],pa[i]]
        list_args.append(args)  
    # parallel computing  
    pool = Pool()
    coeff_M = list(pool.map(getKappa_SingleLayer, list_args))
    pool.terminate()
    fileName= "data/computed/GOES_C02_{}_coeffM_{}layers_{}_dnu={:.2f}cm-1.npy".format(
        spectral, N_layer, model,nu[10]-nu[9])
    np.save(fileName,coeff_M)
