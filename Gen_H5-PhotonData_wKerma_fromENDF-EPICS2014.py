import os;
from pprint import pprint;
import shutil;
import subprocess;
import urllib.request;
import numpy  as np
import pandas as pd
import pkg_resources

import h5py;
import numpy as np;
import matplotlib.pyplot as plt;
import matplotlib.cm;
from matplotlib.patches import Rectangle;

import openmc;


print(" ")
print("_________________________________________")
print ("BEGIN OpenMC Processing of Photon Nuclear Data");
print("_________________________________________")
print(" ");
print ("OpenMC   version:  ",openmc.__version__);
print ("H5PY     version:  ",h5py.__version__);
print ("numpy    version:  ",np.__version__);
print ("pandas   version:  ",pd.__version__);
print(" ");
print ("Photon TLE Kerma Data       included: mt=525");
print ("Total Photon Interaction XS included: mt=501");
print ("Avg Photon Heating Number   included: mt=301");
print ("Nuclear Data is EPICS2014");
print ("ENDF format");
print ("https://www-nds.iaea.org/epics/");

print(" ");
print(" ");
print(" ");

for Z in ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm']:
# for Z in ['H']:
# Load ENDF Nuclear Data into object
       print ("Import Photon ENDF data; PhotoAtomic Data and AtomicRelaxation Data for Element:  ",Z);
       s1=f"./EPICS2014/ENDF/EPDL/{Z}.epdl.endf";
       s2=f"./EPICS2014/ENDF/EADL/{Z}.eadl.endf";
       s3=f"./EPICS2014/H5-photon-kerma/{Z}-kerma.h5";
       X = openmc.data.IncidentPhoton.from_endf(s1,s2);

# Create H5 Nuclear Data
       print ("Producing .h5  of element:  ",Z);
       X.export_to_hdf5(s3,'a','latest');
       print(" ");




print(" ");
print(" ");
print(" ");

print("_________________________________________")
print ("END OpenMC Processing of Photon Nuclear Data");
print("_________________________________________")