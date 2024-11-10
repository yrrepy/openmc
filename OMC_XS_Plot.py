import os
from pprint import pprint
import shutil
import subprocess
import urllib.request
import pkg_resources

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.patches import Rectangle
matplotlib.pyplot.show()

import openmc.data


print(" ");
print ("OpenMC version:  ",openmc.__version__);
print ("H5PY   version:  ",h5py.__version__);
print(" ");


# Load Neutron H5 data into object
h1 = openmc.data.IncidentNeutron.from_hdf5('/home/pyoung/Nuclear_Data/OpenMC/endfb80_hdf5/H1.h5')
print(h1);
print(list(h1.reactions.values())[:10]);
# print(h1.energy);

total = h1[1]
print(total);
heating = h1[301]
print(heating);

print(heating.xs['294K']([1.0]));


energies = h1.energy['294K']
heating_xs = heating.xs['294K'](energies)





# Load Photon H5 data into object
h1ENDFepics2017 = openmc.data.IncidentPhoton.from_hdf5('/home/pyoung/Nuclear_Data/OpenMC/Photon_testing/EPICS2017/H5-OMC13/H.OMC13-kerma.test.h5')



print(h1ENDFepics2017);
print(list(h1ENDFepics2017.reactions.values())[:10]);
print(list(h1ENDFepics2017.bremsstrahlung)[:10]);
print(list(h1ENDFepics2017.bremsstrahlung.values())[:10]);
print(list(h1ENDFepics2017.compton_profiles)[:10]);
print(list(h1ENDFepics2017.compton_profiles.values())[:10]);





heating17 = h1ENDFepics2017[525]
print(heating17.xs);
print(heating17.xs([1000000,2000000,3000000,4000000,5000000]));

heating17_xs = heating17.xs(energies);


# Plot EPICS2017(fromENDF)  mt 525
# energies = h1.energy['294K']
plt.loglog(energies, heating17_xs)
plt.xlabel('Energy (eV)')
plt.ylabel('Heating XS (eV.barn)')
plt.xlim(1,    100000000000) 
plt.ylim(0.1, 1000000000000) 
plt.show(block=True)




# Plot EPICS2017  mt 501
total17 = h1ENDFepics2017[501]
print(total17.xs);
print(total17.xs([1000000,2000000,3000000,4000000,5000000]));

total17_xs = total17.xs(energies);


plt.loglog(energies, total17_xs)
plt.xlabel('Energy (eV)')
plt.ylabel('Total Photon Interaction XS (Barns)')
plt.xlim(1000,    100000000000) 
plt.ylim(0.01, 100) 
plt.show(block=True)



