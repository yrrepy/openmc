import itertools
import json
import os
import re
from pathlib import Path
from math import sqrt, log
from warnings import warn

# Isotopic abundances from M. Berglund and M. E. Wieser, "Isotopic compositions
# of the elements 2009 (IUPAC Technical Report)", Pure. Appl. Chem. 83 (2),
# pp. 397--410 (2011).
# This is NIST standard 03/2023 (https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=)
# Note that the abundances are given as atomic fractions!
NATURAL_ABUNDANCE = {
    'H1': 0.999885, 'H2': 0.000115,
    'He3': 1.34e-06, 'He4': 0.99999866,
    'Li6': 0.0759, 'Li7': 0.9241,
    'Be9': 1.0,
    'B10': 0.199, 'B11': 0.801,
    'C12': 0.9893, 'C13': 0.0107,
    'N14': 0.99636, 'N15': 0.00364,
    'O16': 0.99757, 'O17': 0.00038, 'O18': 0.00205,
    'F19': 1.0,
    'Ne20': 0.9048, 'Ne21': 0.0027, 'Ne22': 0.0925,
    'Na23': 1.0,
    'Mg24': 0.7899, 'Mg25': 0.1, 'Mg26': 0.1101,
    'Al27': 1.0,
    'Si28': 0.92223, 'Si29': 0.04685, 'Si30': 0.03092,
    'P31': 1.0,
    'S32': 0.9499, 'S33': 0.0075, 'S34': 0.0425, 'S36': 0.0001,
    'Cl35': 0.7576, 'Cl37': 0.2424,
    'Ar36': 0.003336, 'Ar38': 0.000629, 'Ar40': 0.996035,
    'K39': 0.932581, 'K40': 0.000117, 'K41': 0.067302,
    'Ca40': 0.96941, 'Ca42': 0.00647, 'Ca43': 0.00135, 'Ca44': 0.02086, 'Ca46': 4e-05, 'Ca48': 0.00187,
    'Sc45': 1.0,
    'Ti46': 0.0825, 'Ti47': 0.0744, 'Ti48': 0.7372, 'Ti49': 0.0541, 'Ti50': 0.0518,
    'V50': 0.0025, 'V51': 0.9975,
    'Cr50': 0.04345, 'Cr52': 0.83789, 'Cr53': 0.09501, 'Cr54': 0.02365,
    'Mn55': 1.0,
    'Fe54': 0.05845, 'Fe56': 0.91754, 'Fe57': 0.02119, 'Fe58': 0.00282, 
    'Co59': 1.0,
    'Ni58': 0.68077, 'Ni60': 0.26223, 'Ni61': 0.011399, 'Ni62': 0.036346, 'Ni64': 0.009255,
    'Cu63': 0.6915, 'Cu65': 0.3085,
    'Zn64': 0.4917, 'Zn66': 0.2773, 'Zn67': 0.0404, 'Zn68': 0.1845, 'Zn70': 0.0061,
    'Ga69': 0.60108, 'Ga71': 0.39892, 'Ge70': 0.2057,
    'Ge72': 0.2745, 'Ge73': 0.0775, 'Ge74': 0.365, 'Ge76': 0.0773,
    'As75': 1.0,
    'Se74': 0.0089, 'Se76': 0.0937, 'Se77': 0.0763, 'Se78': 0.2377, 'Se80': 0.4961, 'Se82': 0.0873,
    'Br79': 0.5069, 'Br81': 0.4931,
    'Kr78': 0.00355, 'Kr80': 0.02286, 'Kr82': 0.11593, 'Kr83': 0.115, 'Kr84': 0.56987, 'Kr86': 0.17279,
    'Rb85': 0.7217, 'Rb87': 0.2783,
    'Sr84': 0.0056, 'Sr86': 0.0986, 'Sr87': 0.07, 'Sr88': 0.8258,
    'Y89': 1.0,
    'Zr90': 0.5145, 'Zr91': 0.1122, 'Zr92': 0.1715, 'Zr94': 0.1738, 'Zr96': 0.028,
    'Nb93': 1.0,
    'Mo92': 0.1453, 'Mo94': 0.0915, 'Mo95': 0.1584, 'Mo96': 0.1667, 'Mo97': 0.096, 'Mo98': 0.2439, 'Mo100': 0.0982,
    'Ru96': 0.0554, 'Ru98': 0.0187, 'Ru99': 0.1276, 'Ru100': 0.126, 'Ru101': 0.1706, 'Ru102': 0.3155, 'Ru104': 0.1862,
    'Rh103': 1.0,
    'Pd102': 0.0102, 'Pd104': 0.1114, 'Pd105': 0.2233, 'Pd106': 0.2733, 'Pd108': 0.2646, 'Pd110': 0.1172,
    'Ag107': 0.51839, 'Ag109': 0.48161,
    'Cd106': 0.0125, 'Cd108': 0.0089, 'Cd110': 0.1249, 'Cd111': 0.128, 'Cd112': 0.2413, 'Cd113': 0.1222, 'Cd114': 0.2873, 'Cd116': 0.0749,
    'In113': 0.0429, 'In115': 0.9571,
    'Sn112': 0.0097, 'Sn114': 0.0066, 'Sn115': 0.0034, 'Sn116': 0.1454, 'Sn117': 0.0768, 'Sn118': 0.2422, 'Sn119': 0.0859, 'Sn120': 0.3258, 'Sn122': 0.0463, 'Sn124': 0.0579,
    'Sb121': 0.5721, 'Sb123': 0.4279,
    'Te120': 0.0009, 'Te122': 0.0255, 'Te123': 0.0089, 'Te124': 0.0474, 'Te125': 0.0707, 'Te126': 0.1884, 'Te128': 0.3174, 'Te130': 0.3408,
    'I127': 1.0,
    'Xe124': 0.000952, 'Xe126': 0.00089, 'Xe128': 0.019102, 'Xe129': 0.264006, 'Xe130': 0.04071, 'Xe131': 0.212324, 'Xe132': 0.269086, 'Xe134': 0.104357, 'Xe136': 0.088573,
    'Cs133': 1.0,
    'Ba130': 0.00106, 'Ba132': 0.00101, 'Ba134': 0.02417, 'Ba135': 0.06592, 'Ba136': 0.07854, 'Ba137': 0.11232, 'Ba138': 0.71698,
    'La138': 0.0008881, 'La139': 0.9991119,
    'Ce136': 0.00185, 'Ce138': 0.00251, 'Ce140': 0.8845, 'Ce142': 0.11114,
    'Pr141': 1.0,
    'Nd142': 0.27152, 'Nd143': 0.12174, 'Nd144': 0.23798, 'Nd145': 0.08293, 'Nd146': 0.17189, 'Nd148': 0.05756, 'Nd150': 0.05638,
    'Sm144': 0.0307, 'Sm147': 0.1499, 'Sm148': 0.1124, 'Sm149': 0.1382, 'Sm150': 0.0738, 'Sm152': 0.2675, 'Sm154': 0.2275,
    'Eu151': 0.4781, 'Eu153': 0.5219,
    'Gd152': 0.002, 'Gd154': 0.0218, 'Gd155': 0.148, 'Gd156': 0.2047, 'Gd157': 0.1565, 'Gd158': 0.2484, 'Gd160': 0.2186,
    'Tb159': 1.0,
    'Dy156': 0.00056, 'Dy158': 0.00095, 'Dy160': 0.02329, 'Dy161': 0.18889, 'Dy162': 0.25475, 'Dy163': 0.24896, 'Dy164': 0.2826,
    'Ho165': 1.0,
    'Er162': 0.00139, 'Er164': 0.01601, 'Er166': 0.33503, 'Er167': 0.22869, 'Er168': 0.26978, 'Er170': 0.1491,
    'Tm169': 1.0,
    'Yb168': 0.00123, 'Yb170': 0.02982, 'Yb171': 0.1409, 'Yb172': 0.2168, 'Yb173': 0.16103, 'Yb174': 0.32026, 'Yb176': 0.12996,
    'Lu175': 0.97401, 'Lu176': 0.02599,
    'Hf174': 0.0016, 'Hf176': 0.0526, 'Hf177': 0.186, 'Hf178': 0.2728, 'Hf179': 0.1362, 'Hf180': 0.3508,
    'Ta180': 0.0001201, 'Ta181': 0.9998799,
    'W180': 0.0012, 'W182': 0.265, 'W183': 0.1431, 'W184': 0.3064, 'W186': 0.2843,
    'Re185': 0.374, 'Re187': 0.626,
    'Os184': 0.0002, 'Os186': 0.0159, 'Os187': 0.0196, 'Os188': 0.1324, 'Os189': 0.1615, 'Os190': 0.2626, 'Os192': 0.4078,
    'Ir191': 0.373, 'Ir193': 0.627, 
    'Pt190': 0.00012, 'Pt192': 0.00782, 'Pt194': 0.3286, 'Pt195': 0.3378, 'Pt196': 0.2521, 'Pt198': 0.07356,
    'Au197': 1.0,
    'Hg196': 0.0015, 'Hg198': 0.0997, 'Hg199': 0.1687, 'Hg200': 0.231, 'Hg201': 0.1318, 'Hg202': 0.2986, 'Hg204': 0.0687,
    'Tl203': 0.2952, 'Tl205': 0.7048,
    'Pb204': 0.014, 'Pb206': 0.241, 'Pb207': 0.221, 'Pb208': 0.524,
    'Bi209': 1.0,
    'Th232': 1.0,
    'Pa231': 1.0,
    'U234': 5.4e-05, 'U235': 0.007204, 'U238': 0.992742
}

# Dictionary to give element symbols from IUPAC names
# (and some common mispellings)
ELEMENT_SYMBOL = {'neutron': 'n', 'hydrogen': 'H', 'helium': 'He',
                 'lithium': 'Li', 'beryllium': 'Be', 'boron': 'B',
                 'carbon': 'C', 'nitrogen': 'N', 'oxygen': 'O', 'fluorine': 'F',
                 'neon': 'Ne', 'sodium': 'Na', 'magnesium': 'Mg',
                 'aluminium': 'Al', 'aluminum': 'Al', 'silicon': 'Si',
                 'phosphorus': 'P', 'sulfur': 'S', 'sulphur': 'S',
                 'chlorine': 'Cl', 'argon': 'Ar', 'potassium': 'K',
                 'calcium': 'Ca', 'scandium': 'Sc', 'titanium': 'Ti',
                 'vanadium': 'V', 'chromium': 'Cr', 'manganese': 'Mn',
                 'iron': 'Fe', 'cobalt': 'Co', 'nickel': 'Ni', 'copper': 'Cu',
                 'zinc': 'Zn', 'gallium': 'Ga', 'germanium': 'Ge',
                 'arsenic': 'As', 'selenium': 'Se', 'bromine': 'Br',
                 'krypton': 'Kr', 'rubidium': 'Rb', 'strontium': 'Sr',
                 'yttrium': 'Y', 'zirconium': 'Zr', 'niobium': 'Nb',
                 'molybdenum': 'Mo', 'technetium': 'Tc', 'ruthenium': 'Ru',
                 'rhodium': 'Rh', 'palladium': 'Pd', 'silver': 'Ag',
                 'cadmium': 'Cd', 'indium': 'In', 'tin': 'Sn', 'antimony': 'Sb',
                 'tellurium': 'Te', 'iodine': 'I', 'xenon': 'Xe',
                 'caesium': 'Cs', 'cesium': 'Cs', 'barium': 'Ba',
                 'lanthanum': 'La', 'cerium': 'Ce', 'praseodymium': 'Pr',
                 'neodymium': 'Nd', 'promethium': 'Pm', 'samarium': 'Sm',
                 'europium': 'Eu', 'gadolinium': 'Gd', 'terbium': 'Tb',
                 'dysprosium': 'Dy', 'holmium': 'Ho', 'erbium': 'Er',
                 'thulium': 'Tm', 'ytterbium': 'Yb', 'lutetium': 'Lu',
                 'hafnium': 'Hf', 'tantalum': 'Ta', 'tungsten': 'W',
                 'wolfram': 'W', 'rhenium': 'Re', 'osmium': 'Os',
                 'iridium': 'Ir', 'platinum': 'Pt', 'gold': 'Au',
                 'mercury': 'Hg', 'thallium': 'Tl', 'lead': 'Pb',
                 'bismuth': 'Bi', 'polonium': 'Po', 'astatine': 'At',
                 'radon': 'Rn', 'francium': 'Fr', 'radium': 'Ra',
                 'actinium': 'Ac', 'thorium': 'Th', 'protactinium': 'Pa',
                 'uranium': 'U', 'neptunium': 'Np', 'plutonium': 'Pu',
                 'americium': 'Am', 'curium': 'Cm', 'berkelium': 'Bk',
                 'californium': 'Cf', 'einsteinium': 'Es', 'fermium': 'Fm',
                 'mendelevium': 'Md', 'nobelium': 'No', 'lawrencium': 'Lr',
                 'rutherfordium': 'Rf', 'dubnium': 'Db', 'seaborgium': 'Sg',
                 'bohrium': 'Bh', 'hassium': 'Hs', 'meitnerium': 'Mt',
                 'darmstadtium': 'Ds', 'roentgenium': 'Rg', 'copernicium': 'Cn',
                 'nihonium': 'Nh', 'flerovium': 'Fl', 'moscovium': 'Mc',
                 'livermorium': 'Lv', 'tennessine': 'Ts', 'oganesson': 'Og'}

ATOMIC_SYMBOL = {0: 'n', 1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C',
                 7: 'N', 8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al',
                 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K',
                 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn',
                 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga',
                 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb',
                 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc',
                 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In',
                 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs',
                 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm',
                 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho',
                 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf', 73: 'Ta',
                 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au',
                 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At',
                 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa',
                 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk',
                 98: 'Cf', 99: 'Es', 100: 'Fm', 101: 'Md', 102: 'No',
                 103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh',
                 108: 'Hs', 109: 'Mt', 110: 'Ds', 111: 'Rg', 112: 'Cn',
                 113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts',
                 118: 'Og'}
ATOMIC_NUMBER = {value: key for key, value in ATOMIC_SYMBOL.items()}

# Values here are from the Committee on Data for Science and Technology
# (CODATA) 2018 recommendation (https://physics.nist.gov/cuu/Constants/).

# The value of the Boltzman constant in units of eV / K
K_BOLTZMANN = 8.617333262e-5

# Unit conversions
EV_PER_MEV = 1.0e6
JOULE_PER_EV = 1.602176634e-19

# Avogadro's constant
AVOGADRO = 6.02214076e23

# Neutron mass in units of amu
NEUTRON_MASS = 1.00866491595

# Used in atomic_mass function as a cache
_ATOMIC_MASS = {}

# Regex for GNDS nuclide names (used in zam function)
_GNDS_NAME_RE = re.compile(r'([A-Zn][a-z]*)(\d+)((?:_[em]\d+)?)')

# Used in half_life function as a cache
_HALF_LIFE = {}
_LOG_TWO = log(2.0)

def atomic_mass(isotope):
    """Return atomic mass of isotope in atomic mass units.

    Atomic mass data comes from the `Atomic Mass Evaluation 2020
    <https://doi.org/10.1088/1674-1137/abddaf>`_.

    Parameters
    ----------
    isotope : str
        Name of isotope, e.g., 'Pu239'

    Returns
    -------
    float
        Atomic mass of isotope in [amu]

    """
    if not _ATOMIC_MASS:

        # Load data from AME2020 file
        mass_file = os.path.join(os.path.dirname(__file__), 'mass_1.mas20.txt')
        with open(mass_file, 'r') as ame:
            # Read lines in file starting at line 37
            for line in itertools.islice(ame, 36, None):
                name = f'{line[20:22].strip()}{int(line[16:19])}'
                mass = float(line[106:109]) + 1e-6*float(
                    line[110:116] + '.' + line[117:123])
                _ATOMIC_MASS[name.lower()] = mass

        # For isotopes found in some libraries that represent all natural
        # isotopes of their element (e.g. C0), calculate the atomic mass as
        # the sum of the atomic mass times the natural abundance of the isotopes
        # that make up the element.
        for element in ['C', 'Zn', 'Pt', 'Os', 'Tl']:
            isotope_zero = element.lower() + '0'
            _ATOMIC_MASS[isotope_zero] = 0.
            for iso, abundance in isotopes(element):
                _ATOMIC_MASS[isotope_zero] += abundance * _ATOMIC_MASS[iso.lower()]

    # Get rid of metastable information
    if '_' in isotope:
        isotope = isotope[:isotope.find('_')]

    return _ATOMIC_MASS[isotope.lower()]


def atomic_weight(element):
    """Return atomic weight of an element in atomic mass units.

    Computes an average of the atomic mass of each of element's naturally
    occurring isotopes weighted by their relative abundance.

    Parameters
    ----------
    element : str
        Element symbol (e.g., 'H') or name (e.g., 'helium')

    Returns
    -------
    float
        Atomic weight of element in [amu]

    """
    weight = 0.
    for nuclide, abundance in isotopes(element):
        weight += atomic_mass(nuclide) * abundance
    if weight > 0.:
        return weight
    else:
        raise ValueError(f"No naturally-occurring isotopes for element '{element}'.")


def half_life(isotope):
    """Return half-life of isotope in seconds or None if isotope is stable

    Half-life values are from the `ENDF/B-VIII.0 decay sublibrary
    <https://www.nndc.bnl.gov/endf-b8.0/download.html>`_.

    .. versionadded:: 0.13.1

    Parameters
    ----------
    isotope : str
        Name of isotope, e.g., 'Pu239'

    Returns
    -------
    float
        Half-life of isotope in [s]

    """
    global _HALF_LIFE
    if not _HALF_LIFE:
        # Load ENDF/B-VIII.0 data from JSON file
        half_life_path = Path(__file__).with_name('half_life.json')
        _HALF_LIFE = json.loads(half_life_path.read_text())

    return _HALF_LIFE.get(isotope.lower())


def decay_constant(isotope):
    """Return decay constant of isotope in [s^-1]

    Decay constants are based on half-life values from the
    :func:`~openmc.data.half_life` function. When the isotope is stable, a decay
    constant of zero is returned.

    .. versionadded:: 0.13.1

    Parameters
    ----------
    isotope : str
        Name of isotope, e.g., 'Pu239'

    Returns
    -------
    float
        Decay constant of isotope in [s^-1]

    See also
    --------
    openmc.data.half_life

    """
    t = half_life(isotope)
    return _LOG_TWO / t if t else 0.0


def water_density(temperature, pressure=0.1013):
    """Return the density of liquid water at a given temperature and pressure.

    The density is calculated from a polynomial fit using equations and values
    from the 2012 version of the IAPWS-IF97 formulation.  Only the equations
    for region 1 are implemented here.  Region 1 is limited to liquid water
    below 100 [MPa] with a temperature above 273.15 [K], below 623.15 [K], and
    below saturation.

    Reference: International Association for the Properties of Water and Steam,
    "Revised Release on the IAPWS Industrial Formulation 1997 for the
    Thermodynamic Properties of Water and Steam", IAPWS R7-97(2012).

    Parameters
    ----------
    temperature : float
        Water temperature in units of [K]
    pressure : float
        Water pressure in units of [MPa]

    Returns
    -------
    float
        Water density in units of [g/cm^3]

    """

    # Make sure the temperature and pressure are inside the min/max region 1
    # bounds.  (Relax the 273.15 bound to 273 in case a user wants 0 deg C data
    # but they only use 3 digits for their conversion to K.)
    if pressure > 100.0:
        warn("Results are not valid for pressures above 100 MPa.")
    elif pressure < 0.0:
        raise ValueError("Pressure must be positive.")
    if temperature < 273:
        warn("Results are not valid for temperatures below 273.15 K.")
    elif temperature > 623.15:
        warn("Results are not valid for temperatures above 623.15 K.")
    elif temperature <= 0.0:
        raise ValueError('Temperature must be positive.')

    # IAPWS region 4 parameters
    n4 = [0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2,
          0.12020824702470e5, -0.32325550322333e7, 0.14915108613530e2,
          -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849,
          0.65017534844798e3]

    # Compute the saturation temperature at the given pressure.
    beta = pressure**(0.25)
    E = beta**2 + n4[2] * beta + n4[5]
    F = n4[0] * beta**2 + n4[3] * beta + n4[6]
    G = n4[1] * beta**2 + n4[4] * beta + n4[7]
    D = 2.0 * G / (-F - sqrt(F**2 - 4 * E * G))
    T_sat = 0.5 * (n4[9] + D
                   - sqrt((n4[9] + D)**2  - 4.0 * (n4[8] + n4[9] * D)))

    # Make sure we aren't above saturation.  (Relax this bound by .2 degrees
    # for deg C to K conversions.)
    if temperature > T_sat + 0.2:
        warn("Results are not valid for temperatures above saturation "
             "(above the boiling point).")

    # IAPWS region 1 parameters
    R_GAS_CONSTANT = 0.461526  # kJ / kg / K
    ref_p = 16.53  # MPa
    ref_T = 1386  # K
    n1f = [0.14632971213167, -0.84548187169114, -0.37563603672040e1,
           0.33855169168385e1, -0.95791963387872, 0.15772038513228,
           -0.16616417199501e-1, 0.81214629983568e-3, 0.28319080123804e-3,
           -0.60706301565874e-3, -0.18990068218419e-1, -0.32529748770505e-1,
           -0.21841717175414e-1, -0.52838357969930e-4, -0.47184321073267e-3,
           -0.30001780793026e-3, 0.47661393906987e-4, -0.44141845330846e-5,
           -0.72694996297594e-15, -0.31679644845054e-4, -0.28270797985312e-5,
           -0.85205128120103e-9, -0.22425281908000e-5, -0.65171222895601e-6,
           -0.14341729937924e-12, -0.40516996860117e-6, -0.12734301741641e-8,
           -0.17424871230634e-9, -0.68762131295531e-18, 0.14478307828521e-19,
           0.26335781662795e-22, -0.11947622640071e-22, 0.18228094581404e-23,
           -0.93537087292458e-25]
    I1f = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4,
           4, 4, 5, 8, 8, 21, 23, 29, 30, 31, 32]
    J1f = [-2, -1, 0, 1, 2, 3, 4, 5, -9, -7, -1, 0, 1, 3, -3, 0, 1, 3, 17, -4,
           0, 6, -5, -2, 10, -8, -11, -6, -29, -31, -38, -39, -40, -41]

    # Nondimensionalize the pressure and temperature.
    pi = pressure / ref_p
    tau = ref_T / temperature

    # Compute the derivative of gamma (dimensionless Gibbs free energy) with
    # respect to pi.
    gamma1_pi = 0.0
    for n, I, J in zip(n1f, I1f, J1f):
        gamma1_pi -= n * I * (7.1 - pi)**(I - 1) * (tau - 1.222)**J

    # Compute the leading coefficient.  This sets the units at
    #   1 [MPa] * [kg K / kJ] * [1 / K]
    # = 1e6 [N / m^2] * 1e-3 [kg K / N / m] * [1 / K]
    # = 1e3 [kg / m^3]
    # = 1 [g / cm^3]
    coeff = pressure / R_GAS_CONSTANT / temperature

    # Compute and return the density.
    return coeff / pi / gamma1_pi


def gnds_name(Z, A, m=0):
    """Return nuclide name using GNDS convention

    .. versionchanged:: 0.14.0
        Function name changed from ``gnd_name`` to ``gnds_name``

    Parameters
    ----------
    Z : int
        Atomic number
    A : int
        Mass number
    m : int, optional
        Metastable state

    Returns
    -------
    str
        Nuclide name in GNDS convention, e.g., 'Am242_m1'

    """
    if m > 0:
        return f'{ATOMIC_SYMBOL[Z]}{A}_m{m}'
    return f'{ATOMIC_SYMBOL[Z]}{A}'


def isotopes(element):
    """Return naturally occurring isotopes and their abundances

    .. versionadded:: 0.12.1

    Parameters
    ----------
    element : str
        Element symbol (e.g., 'H') or name (e.g., 'helium')

    Returns
    -------
    list
        A list of tuples of (isotope, abundance)

    Raises
    ------
    ValueError
        If the element name is not recognized

    """
    # Convert name to symbol if needed
    if len(element) > 2:
        symbol = ELEMENT_SYMBOL.get(element.lower())
        if symbol is None:
            raise ValueError(f'Element name "{element}" not recognised')
        element = symbol

    # Get the nuclides present in nature
    result = []
    for kv in NATURAL_ABUNDANCE.items():
        if re.match(r'{}\d+'.format(element), kv[0]):
            result.append(kv)

    return result


def zam(name):
    """Return tuple of (atomic number, mass number, metastable state)

    Parameters
    ----------
    name : str
        Name of nuclide using GNDS convention, e.g., 'Am242_m1'

    Returns
    -------
    3-tuple of int
        Atomic number, mass number, and metastable state

    """
    try:
        symbol, A, state = _GNDS_NAME_RE.match(name).groups()
    except AttributeError:
        raise ValueError(f"'{name}' does not appear to be a nuclide name in "
                         "GNDS format")

    if symbol not in ATOMIC_NUMBER:
        raise ValueError(f"'{symbol}' is not a recognized element symbol")

    metastable = int(state[2:]) if state else 0
    return (ATOMIC_NUMBER[symbol], int(A), metastable)
