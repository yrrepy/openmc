"""
GENDF Isomeric Branching Chain Patcher (v12)

Adds energy-dependent isomeric branching from GENDF MF=10 data to OpenMC chains.
Supports two mapping modes:
- 'elis' (default): ELIS-based mapping for accurate LFS→LISO conversion
- 'lfs_order': FISPACT-like positional mapping for validation testing

IMPORTANT: decay_file is REQUIRED for both modes (for count validation and logging).

v12 Changes:
- Added single-target isomeric yields detection and logging
- Always logs single-target cases with reason (GENDF_SINGLE_LFS, NO_DECAY_DATA, etc.)
- Added --suppress-single-target-yields flag to optionally suppress redundant yields
- Single-target section in mapping log file

v11 Changes:
- Added mapping_mode parameter ('elis' or 'lfs_order')
- Added COUNT MISMATCH REPORT section for LFS-order mode
- Log header clearly shows the mapping mode
- ELIS reference warnings for LFS-order mode (Ag116-type detection)
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from enum import Enum
from xml.dom import minidom

from openmc.deplete import Chain
from openmc.deplete.gendf import (
    GENDFLibrary, REACTION_TO_MT
)


# =============================================================================
# Single-target reason codes
# =============================================================================

class SingleTargetReason(Enum):
    """Reasons why a reaction ended up with only a single target."""
    GENDF_SINGLE_LFS = "gendf_single_lfs"
    ELIS_TOL_EXCEEDED = "elis_tol_exceeded"
    NO_DECAY_DATA = "no_decay_data"
    ZERO_ELIS_DECAY = "zero_elis_metastables"
    DUPLICATE_MAPPING = "duplicate_mapping"
    PRODUCTS_NOT_IN_CHAIN = "products_not_in_chain"
    LFS_ORDER_DROPPED = "lfs_order_dropped"
    UNKNOWN = "unknown"

# =============================================================================
# Library configurations for CLI
# =============================================================================

LIBRARY_CONFIGS = {
    'jendl50': {
        'description': 'JENDL-5.0 (Native pairing) - UKAEA-1102',
        'endf_gxs_dir': '/home/perry/NukeData/Activation/FISPACT/jendl5data/gendf-1102/',
        'decay_file': '/home/perry/NukeData/Activation/DecayData/jendl5dd/',
        'base_chain': '//home/perry/NukeData/openmc_data/src/openmc_data/depletion/JENDL50/Chain_JENDL50.xml',
        'output_dir': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/JENDL50/',
        'output_prefix': 'Chain_JENDL50-Iso',
        'log_prefix': 'JENDL50_isomer_mapping',
    },
    'cendl32': {
        'description': 'CENDL-3.2 + ENDF/B-8.0 decay - UKAEA-1102',
        'endf_gxs_dir': '/home/perry/NukeData/Activation/FISPACT/CENDL32data/gendf-1102/',
        'decay_file': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/ENDFB80/endf-b8.0-endf/decay/ENDF-B-VIII.0_decay/',
        'base_chain': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/endf-b8.0/chain_endf_b8.0.xml',
        'output_dir': '/home/perry/NukeData/Activation/OMC/Perry-made/Isomeric-Chains/',
        'output_prefix': 'chain_endfCENDL32_dkENDF80_isoCENDL32gendf.mt4.',
        'log_prefix': 'CENDL32_isomer_mapping',
    },
    'endfb80': {
        'description': 'ENDF/B-8.0 (Native pairing) - CCFE-709',
        'endf_gxs_dir': '/home/perry/NukeData/Activation/FISPACT/ENDFB80data/endfb80-n/gxs-709/',
        'decay_file': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/ENDFB80/endf-b8.0-endf/decay/ENDF-B-VIII.0_decay/',
        'base_chain': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/ENDFB80/Chain_ENDFB80.xml',
        'output_dir': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/ENDFB80/',
        'output_prefix': 'Chain_ENDFB80-Iso',
        'log_prefix': 'ENDFB80_isomer_mapping',
    },
    'endfb81': {
        'description': 'ENDF/B-8.1 (Native pairing) - UKAEA-1102',
        'endf_gxs_dir': '/home/perry/NukeData/Activation/FISPACT/ENDFB81data/endfb81-n/gxs-1102/',
        'decay_file': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/ENDFB81/decay/',
        'base_chain': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/ENDFB81/Chain_ENDFB81.xml',
        'output_dir': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/ENDFB81/',
        'output_prefix': 'Chain_ENDFB81-Iso',
        'log_prefix': 'ENDFB81_isomer_mapping',
    },
    'tendl2017': {
        'description': 'TENDL-2017 + decay2012 - CCFE-709',
        'endf_gxs_dir': '/home/perry/NukeData/Activation/FISPACT/TENDL2017data/tal2017-n/gxs-709/',
        'decay_file': '/home/perry/NukeData/Activation/DecayData/ukdd-12_decay.dat',
        'base_chain': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/TENDL2017/Chain_TENDL2017.xml',
        'output_dir': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/TENDL2017/',
        'output_prefix': 'Chain_TENDL2017-Iso',
        'log_prefix': 'TENDL2017_isomer_mapping',
    },
    'tendl2019': {
        'description': 'TENDL-2019 + decay2020 - UKAEA-1102',
        'endf_gxs_dir': '/home/perry/NukeData/Activation/FISPACT/TENDL2019data/gendf-1102/',
        'decay_file': '/home/perry/NukeData/Activation/DecayData/decay_2020/',
        'base_chain': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/TENDL2019/Chain_TENDL2019.xml',
        'output_dir': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/TENDL2019/',
        'output_prefix': 'Chain_TENDL2019-Iso',
        'log_prefix': 'TENDL2019_isomer_mapping',
    },
    'tendl2021': {
        'description': 'TENDL-2021 + decay2020 - UKAEA-1102',
        'endf_gxs_dir': '/home/perry/NukeData/Activation/FISPACT/TENDL2021data/tal2021-n/gendf-1102/',
        'decay_file': '/home/perry/NukeData/Activation/DecayData/decay_2020/',
        'base_chain': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/TENDL2021/Chain_TENDL2021.xml',
        'output_dir': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/TENDL2021/',
        'output_prefix': 'Chain_TENDL2021-Iso',
        'log_prefix': 'TENDL2021_isomer_mapping',
    },
    'jeff33': {
        'description': 'JEFF-3.3 (Native pairing) - CCFE-709',
        'endf_gxs_dir': '/home/perry/NukeData/Activation/FISPACT/JEFF33data/jeff33-n/gxs-709/',
        'decay_file': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/JEFF33/jeff-3.3-endf/decay/',
        'base_chain': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/JEFF33/Chain_JEFF33.xml',
        'output_dir': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/JEFF33/',
        'output_prefix': 'Chain_JEFF33-Iso',
        'log_prefix': 'JEFF33_isomer_mapping',
    },
    'jeff40': {
        'description':   'JEFF-4.0 (Native pairing) - UKAEA-1102',
        'endf_gxs_dir':  '/home/perry/NukeData/Activation/FISPACT/JEFF40-Processed-PREPRO-GENDF/',
        'decay_file':    '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/JEFF40/jeff-4.0-endf/decay/Radioactive_Decay_Data_JEFF-40.txt',
        'base_chain':    '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/JEFF40/Chain_JEFF40.xml',
        'output_dir':    '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/JEFF40/',
        'output_prefix': 'Chain_JEFF40-Iso',
        'log_prefix':    'JEFF40_isomer_mapping',
    },
    'eaf2010': {
        'description':   'EAF-2010 (Native pairing) - CCFE-709',
        'endf_gxs_dir':  '/home/perry/NukeData/Activation/FISPACT/EAF2010data/eaf2010-n/gxs-709/',
        'decay_file':    '/home/perry/NukeData/Activation/DecayData/JEFF311RDD_ALL.OUT',
        'base_chain':    '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/EAF2010/Chain_EAF2010.xml',
        'output_dir':    '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/EAF2010/',
        'output_prefix': 'Chain_EAF2010-Iso',
        'log_prefix':    'EAF2010_isomer_mapping',
    },
    'scale631': {
        'description':   'SCALE-6.3.1: EAF-2010 (JEFF3.1/A+) + ENDF/B-7.1 - CCFE-709',
        'endf_gxs_dir':  '/home/perry/NukeData/Activation/FISPACT/EAF2010data/eaf2010-n/gxs-709/',
        'decay_file':    '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/SCALE613/jeff-SCALE-6.1.3-endf/decay/decay/',
        'base_chain':    '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/SCALE613/Chain_SCALE613.xml',
        'output_dir':    '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/SCALE613/',
        'output_prefix': 'Chain_SCALE613-Iso',
        'log_prefix':    'SCALE613_isomer_mapping',
    },
}


def build_parser():
    """Build argument parser for CLI."""
    # Custom formatter to show library descriptions in help
    class CustomFormatter(argparse.RawDescriptionHelpFormatter):
        pass

    # Build library choices description for help
    lib_help_lines = ["Available library pairings:"]
    for key, config in LIBRARY_CONFIGS.items():
        lib_help_lines.append(f"  {key:<12} {config['description']}")

    epilog = "\n".join(lib_help_lines)

    parser = argparse.ArgumentParser(
        description='GENDF Isomeric Branching Chain Patcher v12\n\n'
                    'Adds energy-dependent isomeric branching from GENDF MF=10 data to OpenMC chains.',
        epilog=epilog,
        formatter_class=CustomFormatter
    )

    parser.add_argument(
        '-l', '--library',
        choices=list(LIBRARY_CONFIGS.keys()),
        required=True,
        metavar='LIB',
        help='Library pairing to use (see list below)'
    )

    parser.add_argument(
        '-m', '--map',
        choices=['elis', 'lfs_order'],
        default='elis',
        help="Mapping mode: 'elis' (production, default) or 'lfs_order' (FISPACT validation)"
    )

    parser.add_argument(
        '-r', '--rtol',
        type=float,
        default=0.50,
        help='Relative tolerance for ELIS matching (default: 0.50 = 50%%)'
    )

    parser.add_argument(
        '-a', '--atol',
        type=float,
        default=0.0,
        help='Absolute tolerance for ELIS matching in eV (default: 0.0)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        default=True,
        help='Enable verbose output (default: True)'
    )

    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        default=False,
        help='Disable verbose output'
    )

    parser.add_argument(
        '--prune-nn-prime-self-loops',
        action='store_true',
        default=False,
        help="Remove (n,n') reactions that have no isomeric branching (self-loops with no effect). "
             "These reactions where target=parent and no metastable production add computational "
             "overhead without affecting depletion results. Default: keep all (n,n') reactions."
    )

    parser.add_argument(
        '--suppress-single-target-yields',
        action='store_true',
        default=False,
        help="Suppress redundant single-target isomeric yields where the sole product equals "
             "the original reaction target with all 1.0 branching ratios. These add XML bloat "
             "without physics impact. Single-target cases are ALWAYS logged with reasons "
             "regardless of this flag. Default: keep all isomeric yields."
    )

    return parser


# =============================================================================
# Helper functions
# =============================================================================

def _z_to_element(z):
    """Convert atomic number to element symbol."""
    elements = [
        'n', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
        'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
        'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
        'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
        'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
        'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
        'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs',
        'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
    ]
    return elements[z] if 0 <= z < len(elements) else f'Z{z}'


def _parse_no_metastable_decay_data_error(error_str):
    """
    Parse 'no_metastable_decay_data' error string to extract details.

    Example error string:
    "WARNING: NO_METASTABLE_DECAY_DATA: Cd112((n,pa))->Rh108_m? LFS=4 ELIS=215000.0 eV. No metastable states in decay library for (Z=45, A=108). Reaction Isomeric Branching not-mapped."

    Returns dict with: parent, reaction, lfs, elis, target_z, target_a, product
    """
    result = {
        'parent': '?', 'reaction': '?', 'lfs': None, 'elis': None,
        'target_z': 0, 'target_a': 0, 'product': '?'
    }

    # Parse parent and reaction: "Cd112((n,pa))->"
    parent_rx_match = re.search(r'(\w+)\(\(([^)]+)\)\)->', error_str)
    if parent_rx_match:
        result['parent'] = parent_rx_match.group(1)
        result['reaction'] = f"({parent_rx_match.group(2)})"

    # Parse LFS: "(LFS=4,"
    lfs_match = re.search(r'LFS=(\d+)', error_str)
    if lfs_match:
        result['lfs'] = int(lfs_match.group(1))

    # Parse ELIS: "ELIS=215000.0 eV"
    elis_match = re.search(r'ELIS=([\d.]+)', error_str)
    if elis_match:
        result['elis'] = float(elis_match.group(1))

    # Parse Z, A: "(Z=45, A=108)"
    za_match = re.search(r'Z=(\d+),\s*A=(\d+)', error_str)
    if za_match:
        result['target_z'] = int(za_match.group(1))
        result['target_a'] = int(za_match.group(2))

    # Build product name from Z, A, LFS
    if result['target_z'] > 0:
        elem = _z_to_element(result['target_z'])
        mass = result['target_a']
        lfs = result['lfs']
        result['product'] = f"{elem}{mass}_m{lfs}" if lfs else f"{elem}{mass}_m?"

    return result


def _check_isomeric_state_proximity(isomer_mappings, rtol):
    """
    Check if different isomeric states of the same nuclide have ELIS values
    that are too close together (within rtol of each other).

    Parameters
    ----------
    isomer_mappings : list of dict
        Isomer mapping entries with 'product', 'elis', 'dk_elis', 'liso' fields
    rtol : float
        Relative tolerance threshold (e.g., 0.5 for 50%)

    Returns
    -------
    dict with 'gendf_overlaps', 'dk_overlaps' lists, each containing:
        {'base': str, 'state1': str, 'elis1': float, 'state2': str, 'elis2': float,
         'rel_diff': float, 'is_overlap': bool}
    """
    from itertools import combinations

    # Group by base nuclide
    by_base = defaultdict(list)
    for m in isomer_mappings:
        product = m.get('product', '')
        if '_m' in product:
            base = product.split('_m')[0]
            liso = m.get('liso')
            gendf_elis = m.get('elis')
            dk_elis = m.get('dk_elis')
            if liso is not None:
                by_base[base].append({
                    'liso': liso,
                    'lfs': m.get('lfs'),
                    'state': f"_m{liso}",
                    'gendf_elis': gendf_elis,
                    'dk_elis': dk_elis
                })

    gendf_pairs = []
    dk_pairs = []

    for base, states in by_base.items():
        # Get unique states by liso
        unique_states = {}
        for s in states:
            liso = s['liso']
            if liso not in unique_states:
                unique_states[liso] = s
            else:
                # If we have multiple entries for same liso, prefer one with values
                if s['gendf_elis'] is not None and unique_states[liso]['gendf_elis'] is None:
                    unique_states[liso] = s

        if len(unique_states) < 2:
            continue

        # Check all pairs
        for (liso1, s1), (liso2, s2) in combinations(unique_states.items(), 2):
            # GENDF-ELFS check
            e1, e2 = s1['gendf_elis'], s2['gendf_elis']
            if e1 is not None and e2 is not None and max(e1, e2) > 0:
                rel_diff = abs(e1 - e2) / max(e1, e2)
                gendf_pairs.append({
                    'base': base,
                    'state1': s1['state'], 'lfs1': s1.get('lfs'), 'elis1': e1,
                    'state2': s2['state'], 'lfs2': s2.get('lfs'), 'elis2': e2,
                    'rel_diff': rel_diff,
                    'is_overlap': rel_diff < rtol
                })

            # DK-ELIS check
            d1, d2 = s1['dk_elis'], s2['dk_elis']
            if d1 is not None and d2 is not None and max(d1, d2) > 0:
                rel_diff = abs(d1 - d2) / max(d1, d2)
                dk_pairs.append({
                    'base': base,
                    'state1': s1['state'], 'elis1': d1,
                    'state2': s2['state'], 'elis2': d2,
                    'rel_diff': rel_diff,
                    'is_overlap': rel_diff < rtol
                })

    # Sort by rel_diff (smallest first = highest risk)
    gendf_pairs.sort(key=lambda x: x['rel_diff'])
    dk_pairs.sort(key=lambda x: x['rel_diff'])

    return {'gendf_pairs': gendf_pairs, 'dk_pairs': dk_pairs}


def _write_isomeric_proximity_check(f, isomer_mappings, rtol):
    """Write isomeric state proximity check section to log file."""
    results = _check_isomeric_state_proximity(isomer_mappings, rtol)

    f.write("\n\n" + "=" * 220 + "\n")
    f.write("ISOMERIC STATE PROXIMITY CHECK\n")
    f.write("=" * 220 + "\n\n")
    f.write(f"Checking for isomeric states with ELIS values within rtol={rtol*100:.0f}% of each other.\n")
    f.write("This could indicate potential mapping ambiguity.\n\n")

    # GENDF-ELFS check (ELFS = excitation energy from GENDF Q-values: QM - QI)
    gendf_overlaps = [p for p in results['gendf_pairs'] if p['is_overlap']]
    f.write(f"GENDF-ELFS PROXIMITY:\n")
    f.write("-" * 120 + "\n")

    # Helper to format LFS value
    def fmt_lfs(lfs):
        return str(lfs) if lfs is not None else '?'

    header = f"{'Base':<12}  {'LFS1':>5}  {'GENDF-ELFS1':>14}  {'LFS2':>5}  {'GENDF-ELFS2':>14}  {'Rel-Diff':>10}  {'Status':<20}"

    if gendf_overlaps:
        f.write(f"\nPOTENTIAL OVERLAPS DETECTED: {len(gendf_overlaps)}\n\n")
        f.write(header + "\n")
        f.write("-" * 90 + "\n")
        for p in gendf_overlaps:
            status = "OVERLAP" if p['rel_diff'] < rtol * 0.5 else "WARNING: Near rtol"
            f.write(f"{p['base']:<12}  {fmt_lfs(p.get('lfs1')):>5}  {p['elis1']:>14.1f}  {fmt_lfs(p.get('lfs2')):>5}  {p['elis2']:>14.1f}  {p['rel_diff']*100:>9.1f}%  {status:<20}\n")
    else:
        f.write("\nNo overlaps detected. Closest pairs (highest risk):\n\n")
        f.write(header + "\n")
        f.write("-" * 90 + "\n")
        for p in results['gendf_pairs'][:10]:
            f.write(f"{p['base']:<12}  {fmt_lfs(p.get('lfs1')):>5}  {p['elis1']:>14.1f}  {fmt_lfs(p.get('lfs2')):>5}  {p['elis2']:>14.1f}  {p['rel_diff']*100:>9.1f}%  {'OK':<20}\n")
        if not results['gendf_pairs']:
            f.write("  (No multi-state nuclides found)\n")

    # DK-ELIS check (ELIS = excitation energy from decay library)
    dk_overlaps = [p for p in results['dk_pairs'] if p['is_overlap']]
    f.write(f"\n\nDK-ELIS PROXIMITY:\n")
    f.write("-" * 120 + "\n")

    # Helper to strip leading underscore from state name (_m1 -> m1)
    def strip_state(s):
        return s.lstrip('_') if s.startswith('_') else s

    header = f"{'Nuclide':<12}  {'LISO1':<6}  {'DK-ELIS1':>14}  {'LISO2':<6}  {'DK-ELIS2':>14}  {'Rel-Diff':>10}  {'Status':<20}"

    if dk_overlaps:
        f.write(f"\nPOTENTIAL OVERLAPS DETECTED: {len(dk_overlaps)}\n\n")
        f.write(header + "\n")
        f.write("-" * 120 + "\n")
        for p in dk_overlaps:
            status = "OVERLAP" if p['rel_diff'] < rtol * 0.5 else "WARNING: Near rtol"
            f.write(f"{p['base']:<12}  {strip_state(p['state1']):<6}  {p['elis1']:>14.1f}  {strip_state(p['state2']):<6}  {p['elis2']:>14.1f}  {p['rel_diff']*100:>9.1f}%  {status:<20}\n")
    else:
        f.write("\nNo overlaps detected. Closest pairs (highest risk):\n\n")
        f.write(header + "\n")
        f.write("-" * 120 + "\n")
        for p in results['dk_pairs'][:10]:
            f.write(f"{p['base']:<12}  {strip_state(p['state1']):<6}  {p['elis1']:>14.1f}  {strip_state(p['state2']):<6}  {p['elis2']:>14.1f}  {p['rel_diff']*100:>9.1f}%  {'OK':<20}\n")
        if not results['dk_pairs']:
            f.write("  (No multi-state nuclides found)\n")


def _write_duplicate_mapping_section(f, duplicate_mapping_errors):
    """Write duplicate mapping conflicts section to log file.

    When multiple GENDF LFS values map to the same decay library LISO
    (within tolerance), only the closest ELIS match is kept.
    This section shows all such conflicts with full context.
    """
    f.write("\n\n" + "=" * 220 + "\n")
    f.write("DUPLICATE MAPPING CONFLICTS\n")
    f.write("=" * 220 + "\n\n")

    f.write("When multiple GENDF LFS values have ELIS within tolerance of the same decay library LISO,\n")
    f.write("only the closest match is kept. Discarded LFS values are shown below with full context.\n\n")

    if not duplicate_mapping_errors:
        f.write("No duplicate mapping conflicts detected.\n")
        return

    f.write(f"Total conflicts: {len(duplicate_mapping_errors)}\n")
    f.write(f"Total LFS discarded: {sum(len(e.get('discarded', [])) for e in duplicate_mapping_errors)}\n\n")

    for err in duplicate_mapping_errors:
        nuclide = err.get('nuclide', '?')
        reaction = err.get('reaction', '?')
        mt = err.get('mt', '?')
        liso = err.get('liso', '?')
        base_nuc = err.get('base_nuclide', '?')
        kept_lfs = err.get('kept_lfs')
        kept_elis = err.get('kept_elis', 0)
        kept_diff = err.get('kept_diff', 0)
        dk_elis = err.get('dk_elis', 0)

        f.write("-" * 120 + "\n")
        f.write(f"{nuclide} {reaction} (MT={mt}) -> {base_nuc}_m{liso}\n")
        f.write("-" * 120 + "\n\n")

        # Resolution
        f.write(f"  RESOLUTION: Kept LFS={kept_lfs} (ELIS={kept_elis:.0f}eV, diff={kept_diff:.0f}eV)\n")
        f.write(f"              DK-ELIS for _m{liso}: {dk_elis:.0f}eV\n\n")

        # Discarded
        discarded = err.get('discarded', [])
        f.write(f"  DISCARDED ({len(discarded)} LFS):\n")
        for d in discarded:
            f.write(f"    LFS={d['lfs']}: ELIS={d['elis']:.0f}eV (diff={d['diff']:.0f}eV)\n")
        f.write("\n")

        # All GENDF LFS levels for this reaction
        gendf_all = err.get('gendf_all_lfs', [])
        discarded_lfs_set = {d['lfs'] for d in discarded}
        f.write(f"  ALL GENDF LFS LEVELS (this reaction):\n")
        f.write(f"    {'LFS':>5}  {'ELIS[eV]':>14}  {'Status':<20}\n")
        for g in gendf_all:
            elis_val = g.get('elis')
            elis_str = f"{elis_val:.0f}" if elis_val is not None else "N/A"
            if g['lfs'] in discarded_lfs_set:
                status = "****not-mapped****"
            elif g['lfs'] == kept_lfs:
                status = f"-> _m{liso} (kept)"
            elif g['lfs'] == 0:
                status = "-> ground"
            else:
                status = ""
            f.write(f"    {g['lfs']:>5}  {elis_str:>14}  {status:<20}\n")
        f.write("\n")

        # All decay library LISO levels for this (Z, A)
        decay_all = err.get('decay_all_liso', [])
        target_z = err.get('target_z', '?')
        target_a = err.get('target_a', '?')
        f.write(f"  ALL DECAY LIBRARY LISO LEVELS (Z={target_z}, A={target_a}):\n")
        f.write(f"    {'LISO':>5}  {'ELIS[eV]':>14}  {'T1/2[s]':>14}  {'Status':<20}\n")
        for d in decay_all:
            half_life = d.get('half_life')
            if isinstance(half_life, (int, float)):
                hl_str = f"{half_life:.3e}"
            else:
                hl_str = str(half_life) if half_life else "stable"
            if d['liso'] == liso:
                status = "<- mapped (conflict resolved)"
            else:
                status = ""
            f.write(f"    {d['liso']:>5}  {d['elis']:>14.0f}  {hl_str:>14}  {status:<20}\n")
        f.write("\n")


# =============================================================================
# Single-target reason detection
# =============================================================================

def _determine_single_target_reason(branching, valid_products, missing_products, all_products):
    """
    Determine why a reaction ended up with only a single target.

    Parameters
    ----------
    branching : IsomericBranching
        The branching data object from GENDF
    valid_products : list
        Products that passed chain validation
    missing_products : list
        Products that were filtered out (not in chain)
    all_products : list
        All original products from branching

    Returns
    -------
    tuple
        (SingleTargetReason, detail_string)
    """
    # Check if GENDF only had one LFS to begin with
    if branching.lfs_mapping and len(branching.lfs_mapping) == 1:
        return SingleTargetReason.GENDF_SINGLE_LFS, "GENDF MF=10 has only one LFS value"

    # Check ELIS mapping errors if available
    if hasattr(branching, 'elis_mapping') and branching.elis_mapping:
        errors = branching.elis_mapping.get('errors', [])
        if errors:
            error_types = {e.get('error_type') for e in errors if e.get('error_type')}

            if 'elis_tol_exceeded' in error_types:
                details = [e for e in errors if e.get('error_type') == 'elis_tol_exceeded']
                return SingleTargetReason.ELIS_TOL_EXCEEDED, f"ELIS tolerance exceeded for {len(details)} product(s)"

            if 'no_metastable_decay_data' in error_types:
                details = [e for e in errors if e.get('error_type') == 'no_metastable_decay_data']
                return SingleTargetReason.NO_DECAY_DATA, f"No decay library data for {len(details)} product(s)"

            if 'zero_elis_metastables' in error_types:
                return SingleTargetReason.ZERO_ELIS_DECAY, "Decay library has no metastable states for this product"

            if 'duplicate_mapping' in error_types:
                return SingleTargetReason.DUPLICATE_MAPPING, "Multiple LFS values mapped to same product"

            if 'lfs_order_dropped' in error_types:
                details = [e for e in errors if e.get('error_type') == 'lfs_order_dropped']
                return SingleTargetReason.LFS_ORDER_DROPPED, f"LFS-order mode dropped {len(details)} metastable(s)"

    # Check if products were filtered out (not in chain)
    if missing_products:
        return SingleTargetReason.PRODUCTS_NOT_IN_CHAIN, f"Filtered out: {', '.join(missing_products)}"

    return SingleTargetReason.UNKNOWN, "Unknown reason (single product but no error recorded)"


# =============================================================================
# XML manipulation
# =============================================================================

def add_branching_to_xml(original_xml_file, branching_data, output_xml_file,
                         chain, verbose=True, prune_nn_prime_self_loops=False,
                         suppress_single_target_yields=False):
    """Add branching data to chain XML.

    Parameters
    ----------
    original_xml_file : str
        Path to input chain XML file
    branching_data : dict
        Isomeric branching data keyed by nuclide name
    output_xml_file : str
        Path to output chain XML file
    chain : openmc.deplete.Chain
        Chain object for nuclide validation
    verbose : bool, optional
        Enable verbose output (default True)
    prune_nn_prime_self_loops : bool, optional
        If True, remove (n,n') reactions that have no isomeric branching.
        These self-loop reactions (target=parent, no metastable production)
        have no net effect on depletion and add computational overhead.
        Default is False.
    suppress_single_target_yields : bool, optional
        If True, suppress redundant single-target isomeric yields where
        the single target equals the original reaction target and all
        branching ratios are 1.0. Default is False (keep all).

    Returns
    -------
    dict
        Summary with keys: 'added', 'skipped', 'errors', 'skipped_details',
        'elis_mappings', 'renormalizations', 'nuclides_with_branching_added',
        'nn_prime_self_loops_pruned', 'single_target_cases', 'single_target_suppressed'
    """
    tree = ET.parse(original_xml_file)
    root = tree.getroot()
    root.set('version', '1.0-branching')

    summary = {
        'added': 0, 'skipped': 0, 'errors': [], 'skipped_details': [],
        'elis_mappings': [], 'renormalizations': [],
        'nuclides_with_branching_added': set(),
        'nn_prime_self_loops_pruned': [],  # Track pruned (n,n') self-loops
        'single_target_cases': [],  # Track all single-target cases
        'single_target_suppressed': 0,  # Count of suppressed single-target cases
    }

    nuclide_map = {nuc.get('name'): nuc for nuc in root.findall('nuclide')}

    for nuclide_name, nuclide_reactions in branching_data.items():
        if nuclide_name not in nuclide_map:
            summary['errors'].append(f"Nuclide {nuclide_name} not in chain")
            summary['skipped'] += len(nuclide_reactions)
            continue

        nuc_elem = nuclide_map[nuclide_name]
        reaction_map = {rx.get('type'): rx for rx in nuc_elem.findall('reaction')}

        for reaction_type, branching in nuclide_reactions.items():
            if reaction_type not in reaction_map:
                summary['skipped'] += 1
                continue

            # Convert to energy_yields format
            energy_yields = {}
            for i, energy in enumerate(branching.energies):
                energy_yields[float(energy)] = {
                    product: float(branching.branching_ratios[j, i])
                    for j, product in enumerate(branching.products)
                }

            # Validate products
            all_products = list(branching.products)
            valid_products = [p for p in all_products
                            if any(nuc.name == p for nuc in chain.nuclides)]
            missing_products = [p for p in all_products if p not in valid_products]

            if not valid_products:
                summary['skipped'] += 1
                missing_with_lfs = [{'name': mp, 'lfs': branching.lfs_mapping.get(mp) if branching.lfs_mapping else None}
                                   for mp in missing_products]
                summary['skipped_details'].append({
                    'nuclide': nuclide_name, 'reaction': reaction_type,
                    'mt': branching.mt, 'missing_with_lfs': missing_with_lfs
                })
                continue

            # Renormalize if some products missing
            if missing_products:
                new_yields = {}
                for energy, products_dict in energy_yields.items():
                    valid_sum = sum(products_dict.get(p, 0.0) for p in valid_products)
                    if valid_sum > 0:
                        new_yields[energy] = {p: products_dict[p] / valid_sum for p in valid_products}
                    else:
                        new_yields[energy] = {p: 0.0 for p in valid_products}
                energy_yields = new_yields

                summary['renormalizations'].append({
                    'nuclide': nuclide_name, 'reaction': reaction_type,
                    'mt': branching.mt, 'valid_products': valid_products,
                    'dropped_products': missing_products
                })

            # Collect ELIS mappings
            if hasattr(branching, 'elis_mapping') and branching.elis_mapping:
                for product, elis_info in branching.elis_mapping.items():
                    half_life = next((nuc.half_life for nuc in chain.nuclides if nuc.name == product), None)
                    lfs = branching.lfs_mapping.get(product) if branching.lfs_mapping else None
                    summary['elis_mappings'].append({
                        'parent': nuclide_name, 'reaction': reaction_type,
                        'mt': branching.mt, 'product': product, 'lfs': lfs,
                        'elis': elis_info.get('elis'), 'dk_elis': elis_info.get('dk_elis'),
                        'liso': elis_info.get('liso'), 'method': elis_info.get('method', 'elis'),
                        'half_life': half_life if half_life else 'stable',
                        'target_z': elis_info.get('target_z'), 'target_a': elis_info.get('target_a')
                    })

            # Detect single-target cases - ALWAYS log regardless of suppress flag
            rx_elem = reaction_map[reaction_type]
            original_target = rx_elem.get('target')

            if len(valid_products) == 1:
                single_product = valid_products[0]
                all_ratios_are_one = all(
                    abs(energy_yields[e].get(single_product, 0.0) - 1.0) < 1e-10
                    for e in energy_yields.keys()
                )

                # Determine reason
                reason, reason_detail = _determine_single_target_reason(
                    branching, valid_products, missing_products, all_products
                )

                # Check if redundant (single target == original reaction target)
                is_redundant = (single_product == original_target) and all_ratios_are_one

                # Decide action based on flag (DEFAULT: KEEP)
                if suppress_single_target_yields and is_redundant:
                    action = "suppressed"
                elif is_redundant:
                    action = "kept (redundant)"
                else:
                    action = "kept (non-redundant)"  # e.g., single metastable differs from original

                # ALWAYS log the case
                summary['single_target_cases'].append({
                    'nuclide': nuclide_name,
                    'reaction': reaction_type,
                    'mt': branching.mt,
                    'single_target': single_product,
                    'original_target': original_target,
                    'reason': reason.value,
                    'reason_detail': reason_detail,
                    'is_redundant': is_redundant,
                    'action': action,
                    'n_energies': len(energy_yields),
                    'missing_products': missing_products
                })

                # Skip XML writing if suppressed
                if action == "suppressed":
                    summary['single_target_suppressed'] += 1
                    summary['skipped'] += 1
                    continue

            # Add to XML
            existing = rx_elem.find('isomeric_yields')
            if existing is not None:
                rx_elem.remove(existing)

            yields_elem = ET.SubElement(rx_elem, 'isomeric_yields')
            yields_elem.set('type', 'energy_dependent')

            products = list(valid_products)
            energies = sorted(energy_yields.keys())

            energies_elem = ET.SubElement(yields_elem, 'energies')
            energies_elem.text = ' '.join(f'{e:.6e}' for e in energies)

            targets_elem = ET.SubElement(yields_elem, 'targets')
            targets_elem.text = ' '.join(products)

            ratios_elem = ET.SubElement(yields_elem, 'branching_ratios')
            ratios_lines = []
            for product in products:
                ratios = [energy_yields[e].get(product, 0.0) for e in energies]
                ratios_lines.append('          ' + ' '.join(f'{r:.6e}' for r in ratios))
            ratios_elem.text = '\n' + '\n'.join(ratios_lines) + '\n        '

            summary['added'] += 1
            summary['nuclides_with_branching_added'].add(nuclide_name)

    # Prune (n,n') self-loops without isomeric branching if requested
    if prune_nn_prime_self_loops:
        for nuc_elem in root.findall('nuclide'):
            nuc_name = nuc_elem.get('name')
            # Get base name (strip _m1, _m2, etc.)
            base_name = nuc_name.split('_')[0] if '_m' in nuc_name else nuc_name

            reactions_to_remove = []
            for rx_elem in nuc_elem.findall('reaction'):
                rx_type = rx_elem.get('type')
                target = rx_elem.get('target')

                # Check if this is an (n,n') self-loop without isomeric branching
                if rx_type == "(n,n')":
                    # Get target base name
                    target_base = target.split('_')[0] if target and '_m' in target else target

                    # Check if it's a self-loop (target base matches nuclide base)
                    if target_base == base_name:
                        # Check if no isomeric_yields element exists
                        isomeric_yields = rx_elem.find('isomeric_yields')
                        if isomeric_yields is None:
                            reactions_to_remove.append((rx_elem, rx_type, target))

            # Remove the identified reactions
            for rx_elem, rx_type, target in reactions_to_remove:
                nuc_elem.remove(rx_elem)
                summary['nn_prime_self_loops_pruned'].append({
                    'nuclide': nuc_name,
                    'reaction': rx_type,
                    'target': target
                })

            # Update reactions count attribute
            if reactions_to_remove:
                remaining_reactions = len(nuc_elem.findall('reaction'))
                nuc_elem.set('reactions', str(remaining_reactions))

    # Write output
    xml_str = ET.tostring(root, encoding='unicode')
    pretty = minidom.parseString(xml_str).toprettyxml(indent='  ')
    lines = [l for l in pretty.split('\n') if l.strip()]
    with open(output_xml_file, 'w') as f:
        f.write('\n'.join(lines))

    return summary


# =============================================================================
# Logging functions
# =============================================================================

def write_isomer_mapping_log(isomer_mappings, log_file, stats=None, elis_errors=None,
                             duplicate_mapping_errors=None, lfs_order_dropped=None,
                             lfs_order_orphan_dk=None, single_target_cases=None):
    """Write comprehensive isomer mapping log."""
    if elis_errors is None:
        elis_errors = []
    if duplicate_mapping_errors is None:
        duplicate_mapping_errors = []
    if lfs_order_dropped is None:
        lfs_order_dropped = []
    if lfs_order_orphan_dk is None:
        lfs_order_orphan_dk = []
    if single_target_cases is None:
        single_target_cases = []

    mapping_mode = stats.get('mapping_mode', 'elis') if stats else 'elis'

    with open(log_file, 'w') as f:
        f.write("=" * 220 + "\n")
        f.write("ISOMER MAPPING LOG\n")
        f.write("=" * 220 + "\n\n")

        # Mapping mode header
        f.write("MAPPING MODE\n")
        f.write("-" * 70 + "\n")
        if mapping_mode == 'lfs_order':
            f.write("MODE: LFS_ORDER (FISPACT-like positional mapping)\n")
            f.write("\n")
            f.write("  Maps GENDF LFS by sorted position: 1st LFS → _m1, 2nd → _m2, etc.\n")
            f.write("  WARNING: May produce incorrect results for nuclides where\n")
            f.write("           LFS order does not match LISO order (e.g., Ag116).\n")
            f.write("           Use 'elis' mode for production calculations.\n")
            f.write("\n")
            f.write("  ELIS data shown below is for REFERENCE ONLY - not used for mapping.\n")
        else:
            f.write("MODE: ELIS (decay library excitation energy matching)\n")
            f.write("\n")
            f.write("  Maps GENDF MF=10 products to OpenMC _m{n} naming based on\n")
            f.write("  excitation energy (ELIS) matching with decay library.\n")
        f.write("\n\n")

        # Source files
        if stats:
            f.write("SOURCE FILES\n")
            f.write("-" * 70 + "\n")
            f.write(f"OpenMC Chain:    {stats.get('base_chain_file', 'N/A')}\n")
            f.write(f"GENDF Library:   {stats.get('gendf_dir', 'N/A')}\n")
            f.write(f"Decay Library:   {stats.get('decay_file', 'N/A')}\n")
            f.write("-" * 70 + "\n")
            f.write(f"Output Chain:    {stats.get('output_chain_file', 'N/A')}\n")

            f.write("\n\nBASE CHAIN DETAILS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total nuclides in chain: {stats.get('chain_nuclides_total', 0)}\n")

            f.write("\n\nMATCHING SETTINGS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Mapping mode:    {mapping_mode}\n")
            f.write(f"ELIS tolerance:  rtol={stats.get('elis_rtol')} ({stats.get('elis_rtol', 0)*100:.0f}%), atol={stats.get('elis_atol')} eV\n")
            if mapping_mode == 'elis':
                f.write("Products beyond tolerance or not in decay library are skipped.\n")
                f.write("Skipped products trigger renormalization for isomeric branching to remaining isomers (constant reaction rate).\n\n")
            else:
                f.write("In LFS-order mode: tolerance used for ELIS reference warnings only.\n")
                f.write("Products dropped if GENDF LFS count > DK-Lib LISO count.\n\n")

        # Summary counts
        err_types = ('elis_tol_exceeded', 'no_metastable_decay_data', 'zero_elis_metastables')
        elis_matched_count = sum(1 for m in isomer_mappings if m.get('method') == 'elis')
        lfs_order_mapped_count = sum(1 for m in isomer_mappings if m.get('method') == 'lfs_order')
        mapped_count = elis_matched_count + lfs_order_mapped_count
        elis_exceeded_count = sum(1 for e in elis_errors if e.get('type') == 'elis_tol_exceeded')
        missing_meta_count = sum(1 for e in elis_errors if e.get('type') == 'no_metastable_decay_data')
        zero_elis_count = sum(1 for e in elis_errors if e.get('type') == 'zero_elis_metastables')
        dup_mapping_count = len(duplicate_mapping_errors)
        dup_discarded_count = sum(len(e.get('discarded', [])) for e in duplicate_mapping_errors)
        dropped_count = len(lfs_order_dropped)
        orphan_count = len(lfs_order_orphan_dk)

        # Count unique nuclides with branching
        nuclides_with_branching = set()
        for m in isomer_mappings:
            nuclides_with_branching.add(m.get('parent'))
        for e in elis_errors:
            if e.get('type') in err_types:
                nuclides_with_branching.add(e.get('parent'))
        for e in lfs_order_dropped:
            nuclides_with_branching.add(e.get('parent'))

        gendf_total = stats.get('gendf_nuclides_total', 0) if stats else 0

        # Calculate totals based on mode
        if mapping_mode == 'elis':
            total = mapped_count + elis_exceeded_count + missing_meta_count + zero_elis_count
            total_gendf_lfs = total + dup_discarded_count
        else:
            total = mapped_count + dropped_count
            total_gendf_lfs = total

        f.write(f"                      nuclides in GENDF library: {gendf_total:5d}\n")
        f.write(f"               nuclides in GENDF with branching: {len(nuclides_with_branching):5d}\n")
        f.write("-" * 52 + "\n")
        f.write(f"                          Total GENDF-LFS found: {total_gendf_lfs:5d}\n")
        if mapping_mode == 'elis':
            f.write(f"                                   ELIS matched: {mapped_count:5d}\n")
        else:
            f.write(f"                              LFS-order mapped: {mapped_count:5d}\n")
        if elis_exceeded_count:
            f.write(f"                             ELIS rtol exceeded: {elis_exceeded_count:5d}\n")
        if missing_meta_count:
            f.write(f"                           No product in DK-Lib: {missing_meta_count:5d}\n")
        if zero_elis_count:
            f.write(f"                       Zero-ELIS in DK-Lib (QA): {zero_elis_count:5d}\n")
        if dropped_count:
            f.write(f"                 LFS dropped (exceeds DK count): {dropped_count:5d}\n")
        if orphan_count:
            f.write(f"                     Orphan DK states (no LFS): {orphan_count:5d}\n")
        if dup_mapping_count:
            f.write(f"                    Duplicate mappings resolved: {dup_mapping_count:5d} ({dup_discarded_count} LFS discarded)\n")
        f.write("\n")

        # Column descriptions
        f.write("COLUMN DEFINITIONS\n")
        f.write("-" * 93 + "\n")
        f.write("MT              = ENDF reaction type number\n")
        f.write("\n")
        f.write("Reaction        = Reaction name (e.g., (n,g), (n,2n))\n")
        f.write("\n")
        f.write("GENDF-LFS       = Level number of the state of ZAP formed by the neutron interaction.\n")
        f.write("                  Indicator to specify the level number of the nuclide (ZAP) (as defined in\n")
        f.write("                  File 8) produced in the reaction (MT number).\n")
        f.write("\n")
        f.write("GENDF-Product   = Product nuclide from GENDF reaction (using _m{LFS} naming)\n")
        f.write("\n")
        f.write("DK-LISO         = Decay library isomeric state number (_m1=1, _m2=2, etc.).\n")
        f.write("                  (Only isomeric levels)\n")
        f.write("\n")
        f.write("CHAIN-Product   = Product nuclide name in chain (after ELIS mapping) (using _m{LISO} naming)\n")
        f.write("\n")
        f.write("CHAIN-t1/2      = Half-life from chain (originally from decay library)\n")
        f.write("\n")
        f.write("GENDF-ELFS[eV]  = Excitation energy of final state calculated from GENDF MF=10 (QM - QI).\n")
        f.write("                  Excitation energy of the reaction product.\n")
        f.write("\n")
        f.write("DK-ELIS[eV]     = Excitation energy from decay library MF=1 MT=451 (matched within tolerance).\n")
        f.write("                  Excitation energy of the target nucleus relative to 0.0 for the ground state.\n")
        f.write("\n")
        f.write("Δ(ELFS-ELIS)    = Discrepancy between GENDF-ELFS and DK-ELIS.\n")
        f.write("                  Shows absolute difference [eV] and relative difference [%].\n")
        f.write("\n")
        f.write("Method          = 'ELIS' (matched via excitation energy), 'not-mapped' (failed to match)\n")
        f.write("\n")
        f.write("Notes           = Additional information:\n")
        f.write("                  - 'Renorm'd (X skipped)' = Branching renormalized because sibling skipped\n")
        f.write("                  - 'ELIS rtol exceeded' = Product skipped, closest match shown\n")
        f.write("                  - 'Product not in DK-Lib' = No metastable data in decay library\n")
        f.write("\n")
        f.write("Further info:\n")
        f.write("  | LIS/LFS | Level number          | All excited states (short-lived + long-lived) |\n")
        f.write("  | LISO    | Isomeric state number | Only Isomeric (metastable/long-lived) states  |\n")
        f.write("\n")
        f.write("=" * 93 + "\n\n")

        # Build map of (parent, mt) → skipped products for renormalization notes
        skipped_by_reaction = defaultdict(list)
        for error in elis_errors:
            err_type = error.get('type')
            if err_type == 'elis_tol_exceeded':
                parent = error.get('parent', '?')
                mt = error.get('mt')
                lfs = error.get('lfs')
                liso = error.get('liso')
                base_nuc = error.get('base_nuclide', '?')
                # Build product name from LISO (closest match shown)
                skipped_name = f"{base_nuc}_m{liso}" if liso else f"{base_nuc}_m{lfs}"
                skipped_by_reaction[(parent, mt)].append({
                    'name': skipped_name,
                    'reason': 'rtol exceeded'
                })
            elif err_type == 'no_metastable_decay_data':
                # For no_metastable_decay_data, try to get info from error dict or parse
                parent = error.get('parent')
                mt = error.get('mt')
                lfs = error.get('lfs')
                base_nuc = error.get('base_nuclide', '?')
                if parent and mt:
                    skipped_name = f"{base_nuc}_m{lfs}" if lfs else f"{base_nuc}_m?"
                    skipped_by_reaction[(parent, mt)].append({
                        'name': skipped_name,
                        'reason': 'no DK-Lib data'
                    })
                else:
                    # Parse from error string
                    parsed = _parse_no_metastable_decay_data_error(error.get('error', ''))
                    parent = parsed.get('parent', '?')
                    reaction = parsed.get('reaction', '?')
                    mt = REACTION_TO_MT.get(reaction)
                    skipped_name = parsed.get('product', '?')
                    if parent and mt:
                        skipped_by_reaction[(parent, mt)].append({
                            'name': skipped_name,
                            'reason': 'no DK-Lib data'
                        })

        # Add duplicate_mapping errors to skipped_by_reaction
        for error in duplicate_mapping_errors:
            parent = error.get('nuclide')
            mt = error.get('mt')
            base_nuc = error.get('base_nuclide', '?')
            kept_lfs = error.get('kept_lfs')
            conflict_liso = error.get('liso')  # The LISO that multiple LFS mapped to
            for d in error.get('discarded', []):
                skipped_name = f"LFS={d['lfs']}"
                skipped_by_reaction[(parent, mt)].append({
                    'name': skipped_name,
                    'reason': f'duplicate mapping to _m{conflict_liso} (closer: LFS={kept_lfs})',
                    'lfs': d['lfs'],
                    'liso': conflict_liso  # The LISO they were trying to map to
                })

        # Group by parent and add renormalization notes to successful mappings
        by_parent = defaultdict(list)
        for m in isomer_mappings:
            parent = m['parent']
            mt = m.get('mt')
            # Check if any siblings were skipped for this reaction
            skipped = skipped_by_reaction.get((parent, mt), [])
            if skipped:
                # Add renormalization note - mass renorm means pro-rata redistribution
                skipped_names = [s['name'] for s in skipped]
                m = dict(m)  # Make a copy to avoid modifying original
                m['notes'] = f"Mass renorm ({', '.join(skipped_names)} omit→pro-rata)"
            by_parent[parent].append(m)

        # Add errors (including no_metastable_decay_data with parsed details)
        for error in elis_errors:
            err_type = error.get('type')
            if err_type == 'elis_tol_exceeded':
                # elis_tol_exceeded errors have 'parent' directly in dict
                by_parent[error.get('parent', '?')].append(error)
            elif err_type == 'no_metastable_decay_data':
                # Check if we have structured data or need to parse
                if error.get('parent') and error.get('mt'):
                    by_parent[error.get('parent', '?')].append(error)
                else:
                    # Parse the error string to extract details
                    parsed = _parse_no_metastable_decay_data_error(error.get('error', ''))
                    parsed['type'] = 'no_metastable_decay_data'
                    # Look up MT from reaction name using REACTION_TO_MT
                    reaction = parsed.get('reaction', '')
                    parsed['mt'] = REACTION_TO_MT.get(reaction)
                    by_parent[parsed['parent']].append(parsed)
            elif err_type == 'zero_elis_metastables':
                # Zero-ELIS metastables: structured data from gendf.py
                by_parent[error.get('parent', '?')].append(error)

        # Add duplicate_mapping discarded entries to by_parent
        for error in duplicate_mapping_errors:
            parent = error.get('nuclide')
            mt = error.get('mt')
            reaction = error.get('reaction', '?')
            base_nuc = error.get('base_nuclide', '?')
            conflict_liso = error.get('liso')
            kept_lfs = error.get('kept_lfs')
            dk_elis = error.get('dk_elis')

            for d in error.get('discarded', []):
                by_parent[parent].append({
                    'type': 'duplicate_mapping',
                    'parent': parent,
                    'mt': mt,
                    'reaction': reaction,
                    'lfs': d['lfs'],
                    'elis': d['elis'],
                    'liso': conflict_liso,
                    'dk_elis': dk_elis,
                    'product': f"{base_nuc}_m{conflict_liso}" if conflict_liso else base_nuc,
                    'base_nuclide': base_nuc,
                    'kept_lfs': kept_lfs,
                    'half_life': '-',
                    'method': 'not-mapped',
                    'notes': f'Dup map to _m{conflict_liso} (LFS={kept_lfs} closer)'
                })

        # Table header
        header = (f"{'MT':>5}  {'Reaction':<12}  {'GENDF-LFS':>9}  {'GENDF-Product':<15}  "
                  f"{'DK-LISO':>7}  {'CHAIN-Product':<15}  {'CHAIN-t½[s]':>12}  "
                  f"{'GENDF-ELFS[eV]':>14}  {'DK-ELIS[eV]':>14}  {'Δ(ELFS-ELIS)':>22}    {'Method':<12}    {'Notes':<30}")
        sep = "-" * 220

        for parent in sorted(by_parent.keys()):
            f.write(f"\n{parent}:\n{sep}\n{header}\n{sep}\n")

            for m in sorted(by_parent[parent], key=lambda x: (x.get('mt') or 0, x.get('method', ''))):
                _write_mapping_row(f, m)

        # HIGH ELIS DISCREPANCY + FAILED MAPPINGS section
        f.write("\n\n" + "=" * 220 + "\n")
        f.write("HIGH ELIS DISCREPANCY (>50%) + FAILED MAPPINGS\n")
        f.write("=" * 220 + "\n\n")

        high_disc = []
        for m in isomer_mappings:
            elis, dk_elis = m.get('elis'), m.get('dk_elis')
            if elis is not None and dk_elis is not None and dk_elis != 0:
                rel_diff = abs(elis - dk_elis) / abs(dk_elis) * 100
                if rel_diff > 50.0:
                    high_disc.append({**m, 'rel_diff': rel_diff, 'notes': ''})

        for err in elis_errors:
            if err.get('type') == 'elis_tol_exceeded' and err.get('omitted', True):
                # Build closest match product name using LISO (not LFS)
                elem = _z_to_element(err.get('target_z', 0))
                mass = err.get('target_a', '?')
                liso = err.get('liso')
                closest_product = f"{elem}{mass}_m{liso}" if liso else f"{elem}{mass}"
                high_disc.append({
                    'parent': err.get('parent'), 'mt': err.get('mt'),
                    'reaction': err.get('reaction'), 'lfs': err.get('lfs'),
                    'product': closest_product,  # Closest match product name
                    'liso': liso, 'elis': err.get('elis'),
                    'dk_elis': err.get('dk_elis'),
                    'half_life': err.get('half_life', '-'),  # From error if available
                    'method': 'not-mapped', 'rel_diff': err.get('diff_percent', 0),
                    'notes': 'ELIS rtol exceeded. Closest shown.'
                })
            elif err.get('type') == 'no_metastable_decay_data':
                # Two error sources: structured (with parent info) or unstructured (exception)
                if 'parent' in err and err.get('parent') is not None:
                    # Structured error with parent info
                    elem = _z_to_element(err.get('target_z', 0))
                    mass = err.get('target_a', '?')
                    lfs = err.get('lfs')
                    high_disc.append({
                        'parent': err.get('parent'), 'mt': err.get('mt'),
                        'reaction': err.get('reaction'), 'lfs': lfs,
                        'product': f"{elem}{mass}_m{lfs}" if lfs else f"{elem}{mass}_m?",
                        'liso': '-', 'elis': err.get('elis'), 'dk_elis': None,
                        'half_life': '-', 'method': 'not-mapped',
                        'rel_diff': float('inf'), 'notes': 'No product in DK-Lib'
                    })
                else:
                    # Unstructured error from exception - parse error string
                    parsed = _parse_no_metastable_decay_data_error(err.get('error', ''))
                    reaction = parsed['reaction']
                    high_disc.append({
                        'parent': parsed['parent'], 'mt': REACTION_TO_MT.get(reaction),
                        'reaction': reaction, 'lfs': parsed['lfs'],
                        'product': parsed['product'],
                        'liso': '-', 'elis': parsed['elis'], 'dk_elis': None,
                        'half_life': '-', 'method': 'not-mapped',
                        'rel_diff': float('inf'), 'notes': 'No product in DK-Lib',
                        'type': 'no_metastable_decay_data'
                    })
            elif err.get('type') == 'zero_elis_metastables':
                # Zero-ELIS metastables: structured data from gendf.py
                elem = _z_to_element(err.get('target_z', 0))
                mass = err.get('target_a', '?')
                lfs = err.get('lfs')
                skipped = err.get('skipped_states', [])
                skipped_str = ', '.join(f"_m{liso}" for liso, _, _ in skipped)
                high_disc.append({
                    'parent': err.get('parent'), 'mt': err.get('mt'),
                    'reaction': err.get('reaction'), 'lfs': lfs,
                    'product': f"{elem}{mass}_m{lfs}" if lfs else f"{elem}{mass}_m?",
                    'liso': '-', 'elis': err.get('elis'), 'dk_elis': 0.0,
                    'half_life': '-', 'method': 'not-mapped',
                    'rel_diff': float('inf'),
                    'notes': f'DK-Lib ELIS=0 ({skipped_str})',
                    'type': 'zero_elis_metastables',
                    'skipped_states': skipped
                })

        high_disc.sort(key=lambda x: x.get('rel_diff', 0), reverse=True)

        if not high_disc:
            f.write("No entries with >50% discrepancy or failed mappings.\n")
        else:
            mapped = sum(1 for h in high_disc if h.get('method') != 'not-mapped')
            failed = len(high_disc) - mapped
            f.write(f"Total: {len(high_disc)} (mapped: {mapped}, not-mapped: {failed})\n\n")

            f.write(f"{sep}\n")
            f.write(f"{'Parent':<12}  {header}\n")
            f.write(f"{sep}\n")

            for m in high_disc:
                f.write(f"{m.get('parent', '?'):<12}  ")
                _write_mapping_row(f, m)

        # ISOMERIC STATE PROXIMITY CHECK section
        rtol = stats.get('elis_rtol', 0.5) if stats else 0.5
        _write_isomeric_proximity_check(f, isomer_mappings, rtol)

        # DUPLICATE MAPPING CONFLICTS section
        _write_duplicate_mapping_section(f, duplicate_mapping_errors)

        # COUNT MISMATCH REPORT section (LFS-order mode only)
        if mapping_mode == 'lfs_order' and (lfs_order_dropped or lfs_order_orphan_dk):
            f.write("\n\n" + "=" * 220 + "\n")
            f.write("COUNT MISMATCH REPORT (LFS-ORDER MODE)\n")
            f.write("=" * 220 + "\n\n")

            if lfs_order_dropped:
                f.write("DROPPED LFS STATES (GENDF LFS count > DK-Lib LISO count):\n")
                f.write("-" * 100 + "\n")
                f.write(f"{'Parent':<12}  {'MT':>5}  {'Reaction':<12}  {'GENDF-LFS':>9}  "
                        f"{'Would-Be':>10}  {'GENDF-ELFS':>14}  {'Reason':<40}\n")
                f.write("-" * 100 + "\n")
                for err in lfs_order_dropped:
                    parent = err.get('parent', '?')
                    mt = err.get('mt', '?')
                    reaction = err.get('reaction', '?')
                    lfs = err.get('lfs', '?')
                    would_be = f"_m{err.get('would_be_liso', '?')}"
                    gendf_elis = err.get('gendf_elis')
                    elis_str = f"{gendf_elis:.1f}" if gendf_elis is not None else "N/A"
                    dk_count = err.get('dk_meta_count', '?')
                    reason = f"DK-Lib has only {dk_count} metastable(s)"
                    f.write(f"{parent:<12}  {mt:>5}  {reaction:<12}  {lfs:>9}  "
                            f"{would_be:>10}  {elis_str:>14}  {reason:<40}\n")
                f.write("\n")

            if lfs_order_orphan_dk:
                f.write("ORPHAN DK-Lib STATES (DK-Lib LISO count > GENDF LFS count):\n")
                f.write("-" * 100 + "\n")
                f.write(f"{'Parent':<12}  {'MT':>5}  {'Reaction':<12}  {'DK-LISO':>10}  "
                        f"{'DK-ELIS':>14}  {'Reason':<40}\n")
                f.write("-" * 100 + "\n")
                for err in lfs_order_orphan_dk:
                    parent = err.get('parent', '?')
                    mt = err.get('mt', '?')
                    reaction = err.get('reaction', '?')
                    liso = f"_m{err.get('liso', '?')}"
                    dk_elis = err.get('dk_elis')
                    elis_str = f"{dk_elis:.1f}" if dk_elis is not None else "N/A"
                    gendf_count = err.get('gendf_meta_count', '?')
                    reason = f"GENDF has only {gendf_count} metastable(s)"
                    f.write(f"{parent:<12}  {mt:>5}  {reaction:<12}  {liso:>10}  "
                            f"{elis_str:>14}  {reason:<40}\n")
                f.write("\n")

        # SINGLE-TARGET ISOMERIC YIELDS section
        if single_target_cases:
            f.write("\n\n" + "=" * 220 + "\n")
            f.write("SINGLE-TARGET ISOMERIC YIELDS\n")
            f.write("=" * 220 + "\n\n")

            suppressed = sum(1 for c in single_target_cases if 'suppressed' in c.get('action', ''))
            kept = len(single_target_cases) - suppressed

            f.write(f"Total: {len(single_target_cases)} reactions (suppressed: {suppressed}, kept: {kept})\n\n")
            f.write("These reactions have only a single target product for isomeric branching.\n")
            f.write("This can be redundant if the single target equals the original reaction target.\n\n")

            # Breakdown by reason
            reason_counts = Counter(c['reason'] for c in single_target_cases)
            f.write("Breakdown by reason:\n")
            for reason, count in reason_counts.most_common():
                pct = 100 * count / len(single_target_cases)
                f.write(f"  {reason}: {count} ({pct:.1f}%)\n")
            f.write("\n")

            f.write("-" * 140 + "\n")
            f.write(f"{'Nuclide':<12} {'MT':<6} {'Reaction':<14} {'Target':<16} "
                    f"{'Orig-Target':<16} {'Redundant':<10} {'Action':<16} {'Reason'}\n")
            f.write("-" * 140 + "\n")

            for case in single_target_cases:
                redundant = 'Yes' if case.get('is_redundant') else 'No'
                f.write(f"{case['nuclide']:<12} {case['mt']:<6} {case['reaction']:<14} "
                        f"{case['single_target']:<16} {case.get('original_target', '-'):<16} "
                        f"{redundant:<10} {case['action'].upper():<16} {case['reason']}\n")

                # Show missing products if any
                if case.get('missing_products'):
                    for mp in case['missing_products']:
                        f.write(f"             Filtered out: {mp}\n")

            f.write("\n")

        # MAPPING CONFLICTS section (expanded tabular format)
        f.write("\n\n" + "=" * 220 + "\n")
        f.write("MAPPING CONFLICTS\n")
        f.write("=" * 220 + "\n\n")

        f.write("N.B.:\n")
        f.write("ONE-TO-MANY and MANY-TO-ONE Mapping not necessarily bad\n")
        f.write("LFS is Arbitrary Per-Reaction\n\n")
        f.write("  LFS (Level number of final excited state) in GENDF MF=10 is just an index assigned\n")
        f.write("  within each reaction's data section. It's not globally consistent across the library.\n\n")
        f.write("  Example - Hf178_m2 production:\n\n")
        f.write("  | Parent | Reaction | LFS | ELFS (eV) | Physical State |\n")
        f.write("  |--------|----------|-----|-----------|----------------|\n")
        f.write("  | Hf177  | (n,g)    | 10  | 2,446,090 | Hf178_m2       |\n")
        f.write("  | Lu179  | (n,p)    | 99  | 2,446,090 | Hf178_m2       |\n")
        f.write("  | Ta181  | (n,a)    | 3   | 2,446,090 | Hf178_m2       |\n\n")
        f.write("  All three reactions produce the same physical state (Hf178_m2 at 2.446 MeV excitation),\n")
        f.write("  but they use different LFS values (10, 99, 3).\n\n")

        conflicts = _detect_mapping_conflicts(isomer_mappings)
        _write_conflicts_table(f, conflicts, isomer_mappings, header, sep)

    print(f"Isomer mapping log written to: {log_file}")


def _write_mapping_row(f, m):
    """Write a single mapping row."""
    mt = m.get('mt', '?')
    reaction = m.get('reaction', '?')
    lfs = m.get('lfs')
    product = m.get('product', '?')
    liso = m.get('liso')
    half_life = m.get('half_life', '-')
    elis = m.get('elis')
    dk_elis = m.get('dk_elis')
    method = m.get('method', 'unknown')
    notes = m.get('notes', '')

    # Handle error entries - use base_nuclide if product not set
    err_type = m.get('type')
    if err_type == 'elis_tol_exceeded':
        method = 'not-mapped'
        notes = 'ELIS rtol exceeded. Closest shown.'
        # Use base_nuclide from error dict if product is '?'
        if product == '?':
            product = m.get('base_nuclide', '?')
    elif err_type == 'no_metastable_decay_data':
        method = 'not-mapped'
        notes = 'Product not in DK-Lib'
        # Use base_nuclide from error dict if product is '?'
        if product == '?':
            product = m.get('base_nuclide', '?')
    elif err_type == 'zero_elis_metastables':
        method = 'not-mapped'
        # Format skipped states info
        skipped = m.get('skipped_states', [])
        if skipped:
            skipped_str = ', '.join(f"_m{liso}" for liso, _, _ in skipped)
            notes = f'DK-Lib ELIS=0 ({skipped_str})'
        else:
            notes = 'DK-Lib has ELIS=0 (data quality)'
        if product == '?':
            product = m.get('base_nuclide', '?')
    elif err_type == 'duplicate_mapping':
        # Duplicate mapping - this LFS was discarded because another LFS was closer
        method = 'not-mapped'
        # Notes already set when adding to by_parent

    mt_str = str(mt) if mt is not None else "?"
    lfs_str = str(lfs) if lfs is not None else "?"
    base_nuc = product.split('_')[0] if '_' in str(product) else str(product)
    gendf_product = f"{base_nuc}_m{lfs}" if lfs is not None else base_nuc
    liso_str = str(liso) if liso is not None else "-"

    # For "closest shown" entries, show actual product; otherwise OMITTED for not-mapped
    if method == 'not-mapped' and 'Closest' in notes:
        chain_product = f"{base_nuc}_m{liso}" if liso is not None else base_nuc
    elif method == 'not-mapped' and err_type == 'no_metastable_decay_data':
        chain_product = '-'  # No product exists in decay library at all
    elif method == 'not-mapped' and err_type == 'zero_elis_metastables':
        chain_product = '-'  # Product has ELIS=0 in decay library (data quality issue)
    elif method == 'not-mapped':
        chain_product = 'OMITTED'
    else:
        chain_product = product

    if isinstance(half_life, (int, float)):
        half_life_str = f"{half_life:.3e}"
    else:
        half_life_str = str(half_life)

    elis_str = f"{elis:.1f}" if elis is not None else "N/A"

    # For missing_metastable or zero_elis, show '-' or '0.0' to indicate data issues
    if err_type == 'no_metastable_decay_data':
        dk_elis_str = "-"
        disc_str = "-"
    elif err_type == 'zero_elis_metastables':
        dk_elis_str = "0.0 (invalid)"
        disc_str = "N/A"
    else:
        dk_elis_str = f"{dk_elis:.1f}" if dk_elis is not None else "N/A"
        if elis is not None and dk_elis is not None and dk_elis != 0:
            abs_diff = abs(elis - dk_elis)
            rel_diff = abs_diff / abs(dk_elis) * 100
            disc_str = f"{abs_diff:.0f}eV ({rel_diff:.2f}%)"
        else:
            disc_str = "N/A"

    # Display method name appropriately
    if method == 'elis':
        method_display = 'ELIS'
    elif method == 'lfs_order':
        method_display = 'LFS_ORDER'
    else:
        method_display = method

    f.write(f"{mt_str:>5}  {reaction:<12}  {lfs_str:>9}  {gendf_product:<15}  "
            f"{liso_str:>7}  {chain_product:<15}  {half_life_str:>12}  "
            f"{elis_str:>14}  {dk_elis_str:>14}  {disc_str:>22}    {method_display:<12}    {notes:<30}\n")


def _detect_mapping_conflicts(isomer_mappings):
    """Detect LFS->LISO mapping conflicts."""
    valid = [m for m in isomer_mappings if m.get('liso') is not None and m.get('lfs') is not None]

    lfs_to_liso = defaultdict(list)
    for m in valid:
        base = m.get('product', '').split('_')[0]
        lfs_to_liso[(base, m['lfs'])].append(m)

    one_to_many = []
    for (base, lfs), mappings in lfs_to_liso.items():
        lisos = set(m['liso'] for m in mappings)
        if len(lisos) > 1:
            one_to_many.append({'base': base, 'lfs': lfs, 'mappings': mappings})

    liso_to_lfs = defaultdict(list)
    for m in valid:
        base = m.get('product', '').split('_')[0]
        liso_to_lfs[(base, m['liso'])].append(m)

    many_to_one = []
    for (base, liso), mappings in liso_to_lfs.items():
        lfs_vals = set(m['lfs'] for m in mappings)
        if len(lfs_vals) > 1:
            many_to_one.append({'base': base, 'liso': liso, 'mappings': mappings})

    return {'one_to_many': one_to_many, 'many_to_one': many_to_one}


def _write_conflicts_table(f, conflicts, all_mappings, header, sep):
    """Write mapping conflicts in tabular format."""
    one_to_many = conflicts['one_to_many']
    many_to_one = conflicts['many_to_one']

    if not one_to_many and not many_to_one:
        f.write("No mapping conflicts detected.\n")
        return

    if one_to_many:
        f.write("ONE-TO-MANY (Same LFS → Different LISO) - CRITICAL:\n")
        f.write("Same GENDF LFS maps to different decay library LISO values.\n\n")

        for conflict in one_to_many:
            base, lfs = conflict['base'], conflict['lfs']
            mappings = conflict['mappings']
            lisos = sorted(set(m['liso'] for m in mappings))
            f.write(f"  {base} LFS={lfs} → LISO values: {lisos}\n")
            f.write(f"  {sep}\n")
            f.write(f"  {'Parent':<12}  {header}\n")
            f.write(f"  {sep}\n")
            for m in mappings:
                f.write(f"  {m.get('parent', '?'):<12}  ")
                _write_mapping_row(f, m)
            f.write("\n")

    if many_to_one:
        f.write("\nMANY-TO-ONE (Different LFS → Same LISO):\n")
        f.write("Multiple GENDF LFS values map to same decay library LISO.\n\n")

        for conflict in many_to_one:
            base, liso = conflict['base'], conflict['liso']
            mappings = conflict['mappings']
            lfs_vals = sorted(set(m['lfs'] for m in mappings))
            f.write(f"  {base} LISO={liso} ← LFS values: {lfs_vals}\n")
            f.write(f"  {sep}\n")
            f.write(f"  {'Parent':<12}  {header}\n")
            f.write(f"  {sep}\n")
            for m in mappings:
                f.write(f"  {m.get('parent', '?'):<12}  ")
                _write_mapping_row(f, m)
            f.write("\n")


def write_renormalization_log(renormalizations, log_file):
    """Write renormalization log."""
    with open(log_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RENORMALIZATION LOG\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total renormalized reactions: {len(renormalizations)}\n\n")

        for r in renormalizations:
            f.write(f"{r['nuclide']} {r['reaction']} (MT={r['mt']}):\n")
            f.write(f"  Valid: {r['valid_products']}\n")
            f.write(f"  Dropped: {r['dropped_products']}\n\n")

    print(f"Renormalization log written to: {log_file}")


# =============================================================================
# Main workflow
# =============================================================================

def main(endf_gxs_dir, base_chain_file, output_chain_file,
         decay_file,
         mapping_mode='elis',
         elis_rtol=0.50, elis_atol=0.0,
         skip_zero_elis_metastables=True,
         mt_list=None, verbose=True,
         isomer_mapping_log_file=None,
         renormalization_log_file=None,
         prune_nn_prime_self_loops=False,
         suppress_single_target_yields=False):
    """
    Main workflow: GENDF MF=10 → OpenMC chain with isomeric branching.

    Parameters
    ----------
    endf_gxs_dir : str
        Directory containing GENDF files
    base_chain_file : str
        Base OpenMC chain XML file
    output_chain_file : str
        Output chain XML file
    decay_file : str
        ENDF decay library for ELIS mapping (REQUIRED for all modes)
    mapping_mode : str
        Isomeric state mapping mode:
        - 'elis' (default): ELIS-based matching (production recommended)
        - 'lfs_order': FISPACT-like positional mapping (validation)
    elis_rtol : float
        Relative tolerance for ELIS matching (default 0.50 = 50%).
        In ELIS mode: products beyond rtol are skipped.
        In LFS-order mode: used for ELIS reference warnings only.
    elis_atol : float
        Absolute tolerance in eV (default 0.0)
    """
    # Validate required parameters
    if decay_file is None:
        raise ValueError(
            "decay_file is required for isomeric branching (both modes)."
        )
    if mapping_mode not in ('elis', 'lfs_order'):
        raise ValueError(
            f"Invalid mapping_mode '{mapping_mode}'. Must be 'elis' or 'lfs_order'."
        )

    print("=" * 60)
    print("GENDF Isomeric Branching Chain Patcher v12")
    print("=" * 60)
    print(f"\nMAPPING MODE: {mapping_mode.upper()}")
    if mapping_mode == 'lfs_order':
        print("  (FISPACT-like positional mapping for validation)")
        print("  WARNING: May produce incorrect results for nuclides where")
        print("           LFS order ≠ LISO order (e.g., Ag116)")
    else:
        print("  (ELIS-based matching - production recommended)")

    # Step 1: Load chain
    print("\nStep 1: Loading base chain...")
    chain = Chain.from_xml(base_chain_file)
    print(f"  Loaded {len(chain.nuclides)} nuclides")

    # Step 2: Detect energy structure
    # Step 2-3: Load GENDF library (energy structure auto-detected)
    print("\nStep 2: Loading GENDF library...")
    lib_kwargs = {
        'validate_energy_grid': False,
        'skip_zero_elis_metastables': skip_zero_elis_metastables,
        'decay_file': decay_file,
        'elis_rtol': elis_rtol,
        'elis_atol': elis_atol,
        'mapping_mode': mapping_mode,
    }
    print(f"  Mapping mode: {mapping_mode}")
    print(f"  ELIS tolerance: rtol={elis_rtol} ({elis_rtol*100:.0f}%), atol={elis_atol} eV")

    lib = GENDFLibrary(endf_gxs_dir, **lib_kwargs)
    print(f"  Energy structure: {lib.energy_structure}")
    print(f"  Available nuclides: {len(lib.available_nuclides())}")

    print("\nStep 4: Extracting MF=10 branching data...")
    branching_data = lib.process_library_for_branching(
        mt_list=mt_list, verbose=verbose, chain=chain
    )

    # Capture errors
    elis_errors = []
    duplicate_mapping_errors = []
    lfs_order_dropped = []
    lfs_order_orphan_dk = []
    for err in lib.processing_errors:
        err_type = err.get('type')
        if err_type in ('elis_tol_exceeded', 'no_metastable_decay_data', 'zero_elis_metastables'):
            elis_errors.append(err)
        elif err_type == 'duplicate_mapping':
            duplicate_mapping_errors.append(err)
        elif err_type == 'lfs_order_dropped':
            lfs_order_dropped.append(err)
        elif err_type == 'lfs_order_orphan_dk':
            lfs_order_orphan_dk.append(err)

    # Calculate counts for summary
    # branching_data structure: {nuclide: {reaction_name: IsomericBranching}}
    mapped_count = sum(
        sum(1 for p in branching.products if '_m' in p)
        for nuclide_reactions in branching_data.values()
        for branching in nuclide_reactions.values()
    )
    rtol_exceeded = sum(1 for e in elis_errors if e.get('type') == 'elis_tol_exceeded')
    no_dk_data = sum(1 for e in elis_errors if e.get('type') == 'no_metastable_decay_data')
    zero_elis = sum(1 for e in elis_errors if e.get('type') == 'zero_elis_metastables')
    dup_mappings = len(duplicate_mapping_errors)
    # Count discarded LFS values (each duplicate_mapping error has 'discarded' list)
    dup_discarded = sum(len(e.get('discarded', [])) for e in duplicate_mapping_errors)
    dropped_count = len(lfs_order_dropped)
    orphan_count = len(lfs_order_orphan_dk)

    if mapping_mode == 'elis':
        total_mappings = mapped_count + rtol_exceeded + no_dk_data
        total_gendf_lfs = total_mappings + dup_discarded
    else:
        total_mappings = mapped_count + dropped_count
        total_gendf_lfs = total_mappings

    print(f"\n                      nuclides in GENDF library: {len(lib.available_nuclides()):5d}")
    print(f"               nuclides in GENDF with branching: {len(branching_data):5d}")
    print("-" * 52)
    print(f"                          Total GENDF-LFS found: {total_gendf_lfs:5d}")
    if mapping_mode == 'elis':
        print(f"                                   ELIS matched: {mapped_count:5d}")
    else:
        print(f"                              LFS-order mapped: {mapped_count:5d}")
    if rtol_exceeded:
        print(f"                             ELIS rtol exceeded: {rtol_exceeded:5d}")
    if no_dk_data:
        print(f"                           No product in DK-Lib: {no_dk_data:5d}")
    if dropped_count:
        print(f"                 LFS dropped (exceeds DK count): {dropped_count:5d}")
    if orphan_count:
        print(f"                     Orphan DK states (no LFS): {orphan_count:5d}")
    if dup_mappings:
        print(f"                    Duplicate mappings resolved: {dup_mappings:5d} ({dup_discarded} LFS discarded)")

    # Step 5: Add to XML
    print("\nStep 5: Adding branching to chain XML...")
    if prune_nn_prime_self_loops:
        print("  Pruning (n,n') self-loops without isomeric branching...")
    summary = add_branching_to_xml(
        original_xml_file=base_chain_file,
        branching_data=branching_data,
        output_xml_file=output_chain_file,
        chain=chain,
        verbose=verbose,
        prune_nn_prime_self_loops=prune_nn_prime_self_loops,
        suppress_single_target_yields=suppress_single_target_yields
    )

    if summary['renormalizations']:
        print(f"\nRenormalized: {len(summary['renormalizations'])}")

    if summary['nn_prime_self_loops_pruned']:
        print(f"Pruned (n,n') self-loops: {len(summary['nn_prime_self_loops_pruned'])}")

    # Single-target summary (always printed if any exist)
    if summary['single_target_cases']:
        cases = summary['single_target_cases']
        suppressed = summary.get('single_target_suppressed', 0)
        kept = len(cases) - suppressed

        print(f"\n  Single-target isomeric yields: {len(cases)}")
        print(f"    Suppressed (redundant): {suppressed}")
        print(f"    Kept: {kept}")

        # Breakdown by reason
        reason_counts = Counter(c['reason'] for c in cases)
        print(f"\n    By reason:")
        for reason, count in reason_counts.most_common():
            print(f"      {reason}: {count} ({100*count/len(cases):.1f}%)")

    # Step 6: Write logs
    if renormalization_log_file and summary['renormalizations']:
        write_renormalization_log(summary['renormalizations'], renormalization_log_file)

    if isomer_mapping_log_file:
        stats = {
            'base_chain_file': base_chain_file,
            'gendf_dir': endf_gxs_dir,
            'decay_file': decay_file,
            'output_chain_file': output_chain_file,
            'mapping_mode': mapping_mode,
            'elis_rtol': elis_rtol,
            'elis_atol': elis_atol,
            'gendf_nuclides_total': len(lib.available_nuclides()),
            'chain_nuclides_total': len(chain.nuclides),
        }
        write_isomer_mapping_log(summary['elis_mappings'], isomer_mapping_log_file,
                                stats=stats, elis_errors=elis_errors,
                                duplicate_mapping_errors=duplicate_mapping_errors,
                                lfs_order_dropped=lfs_order_dropped,
                                lfs_order_orphan_dk=lfs_order_orphan_dk,
                                single_target_cases=summary.get('single_target_cases', []))

    return chain


# =============================================================================
# CLI execution
# =============================================================================

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    # Get library configuration
    config = LIBRARY_CONFIGS[args.library]

    # Determine verbose setting (--quiet overrides --verbose)
    verbose = not args.quiet

    # Build output filenames with mapping suffix
    suffix = '.elis_mapped' if args.map == 'elis' else '.lfs_order_mapped'
    output_chain = f"{config['output_dir']}{config['output_prefix']}{suffix}.xml"
    log_file = f"{config['output_dir']}{config['log_prefix']}{suffix}.txt"

    print("=" * 70)
    print("GENDF Isomeric Branching Chain Patcher v12")
    print("=" * 70)
    print(f"\nLibrary:      {args.library} - {config['description']}")
    print(f"Mapping mode: {args.map}")
    print(f"Tolerances:   rtol={args.rtol}, atol={args.atol}")
    if args.prune_nn_prime_self_loops:
        print("Prune (n,n') self-loops: ENABLED")
    if args.suppress_single_target_yields:
        print("Suppress single-target yields: ENABLED")
    print(f"\nInput chain:  {config['base_chain']}")
    print(f"Output chain: {output_chain}")
    print(f"Mapping log:  {log_file}")
    print()

    # Run the patcher
    chain = main(
        endf_gxs_dir=config['endf_gxs_dir'],
        base_chain_file=config['base_chain'],
        output_chain_file=output_chain,
        decay_file=config['decay_file'],
        mapping_mode=args.map,
        elis_rtol=args.rtol,
        elis_atol=args.atol,
        verbose=verbose,
        isomer_mapping_log_file=log_file,
        prune_nn_prime_self_loops=args.prune_nn_prime_self_loops,
        suppress_single_target_yields=args.suppress_single_target_yields
    )

    print("\n" + "=" * 70)
    print("Done. Chain saved to:", output_chain)
    print("=" * 70)
