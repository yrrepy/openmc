"""chain module.

This module contains information about a depletion chain.  A depletion chain is
loaded from an .xml file and all the nuclides are linked together.
"""

from io import StringIO
from itertools import chain
import math
import numpy as np
import re
from collections import defaultdict, namedtuple
from collections.abc import Mapping, Iterable
from numbers import Real, Integral
from pathlib import Path
from warnings import warn
from typing import List

import lxml.etree as ET

from openmc.checkvalue import check_type, check_length, check_greater_than, PathLike
from .._sparse_compat import csc_array, dok_array
from openmc.data import gnds_name, zam
from openmc.exceptions import DataError
from .nuclide import FissionYieldDistribution, Nuclide
from .._xml import get_text
import openmc.data


# tuple of (possible MT values, secondaries)
ReactionInfo = namedtuple('ReactionInfo', ('mts', 'secondaries'))

REACTIONS = {
    "(n,n')": ReactionInfo({4}, ()),  # Inelastic scattering (same Z, A)
    '(n,2nd)': ReactionInfo({11}, ('H2',)),
    '(n,2n)': ReactionInfo(set(chain([16], range(875, 892))), ()),
    '(n,3n)': ReactionInfo({17}, ()),
    '(n,na)': ReactionInfo({22}, ('He4',)),
    '(n,n3a)': ReactionInfo({23}, ('He4', 'He4', 'He4')),
    '(n,2na)': ReactionInfo({24}, ('He4',)),
    '(n,3na)': ReactionInfo({25}, ('He4',)),
    '(n,np)': ReactionInfo({28}, ('H1',)),
    '(n,n2a)': ReactionInfo({29}, ('He4', 'He4')),
    '(n,2n2a)': ReactionInfo({30}, ('He4', 'He4')),
    '(n,nd)': ReactionInfo({32}, ('H2',)),
    '(n,nt)': ReactionInfo({33}, ('H3',)),
    '(n,n3He)': ReactionInfo({34}, ('He3',)),
    '(n,nd2a)': ReactionInfo({35}, ('H2', 'He4', 'He4')),
    '(n,nt2a)': ReactionInfo({36}, ('H3', 'He4', 'He4')),
    '(n,4n)': ReactionInfo({37}, ()),
    '(n,2np)': ReactionInfo({41}, ('H1',)),
    '(n,3np)': ReactionInfo({42}, ('H1',)),
    '(n,n2p)': ReactionInfo({44}, ('H1', 'H1')),
    '(n,npa)': ReactionInfo({45}, ('H1', 'He4')),
    '(n,gamma)': ReactionInfo({102}, ()),
    '(n,p)': ReactionInfo(set(chain([103], range(600, 650))), ('H1',)),
    '(n,d)': ReactionInfo(set(chain([104], range(650, 700))), ('H2',)),
    '(n,t)': ReactionInfo(set(chain([105], range(700, 750))), ('H3',)),
    '(n,3He)': ReactionInfo(set(chain([106], range(750, 800))), ('He3',)),
    '(n,a)': ReactionInfo(set(chain([107], range(800, 850))), ('He4',)),
    '(n,2a)': ReactionInfo({108}, ('He4', 'He4')),
    '(n,3a)': ReactionInfo({109}, ('He4', 'He4', 'He4')),
    '(n,2p)': ReactionInfo({111}, ('H1', 'H1')),
    '(n,pa)': ReactionInfo({112}, ('H1', 'He4')),
    '(n,t2a)': ReactionInfo({113}, ('H3', 'He4', 'He4')),
    '(n,d2a)': ReactionInfo({114}, ('H2', 'He4', 'He4')),
    '(n,pd)': ReactionInfo({115}, ('H1', 'H2')),
    '(n,pt)': ReactionInfo({116}, ('H1', 'H3')),
    '(n,da)': ReactionInfo({117}, ('H2', 'He4')),
    '(n,5n)': ReactionInfo({152}, ()),
    '(n,6n)': ReactionInfo({153}, ()),
    '(n,2nt)': ReactionInfo({154}, ('H3',)),
    '(n,ta)': ReactionInfo({155}, ('H3', 'He4')),
    '(n,4np)': ReactionInfo({156}, ('H1',)),
    '(n,3nd)': ReactionInfo({157}, ('H2',)),
    '(n,nda)': ReactionInfo({158}, ('H2', 'He4')),
    '(n,2npa)': ReactionInfo({159}, ('H1', 'He4')),
    '(n,7n)': ReactionInfo({160}, ()),
    '(n,8n)': ReactionInfo({161}, ()),
    '(n,5np)': ReactionInfo({162}, ('H1',)),
    '(n,6np)': ReactionInfo({163}, ('H1',)),
    '(n,7np)': ReactionInfo({164}, ('H1',)),
    '(n,4na)': ReactionInfo({165}, ('He4',)),
    '(n,5na)': ReactionInfo({166}, ('He4',)),
    '(n,6na)': ReactionInfo({167}, ('He4',)),
    '(n,7na)': ReactionInfo({168}, ('He4',)),
    '(n,4nd)': ReactionInfo({169}, ('H2',)),
    '(n,5nd)': ReactionInfo({170}, ('H2',)),
    '(n,6nd)': ReactionInfo({171}, ('H2',)),
    '(n,3nt)': ReactionInfo({172}, ('H3',)),
    '(n,4nt)': ReactionInfo({173}, ('H3',)),
    '(n,5nt)': ReactionInfo({174}, ('H3',)),
    '(n,6nt)': ReactionInfo({175}, ('H3',)),
    '(n,2n3He)': ReactionInfo({176}, ('He3',)),
    '(n,3n3He)': ReactionInfo({177}, ('He3',)),
    '(n,4n3He)': ReactionInfo({178}, ('He3',)),
    '(n,3n2p)': ReactionInfo({179}, ('H1', 'H1')),
    '(n,3n2a)': ReactionInfo({180}, ('He4', 'He4')),
    '(n,3npa)': ReactionInfo({181}, ('H1', 'He4')),
    '(n,dt)': ReactionInfo({182}, ('H2', 'H3')),
    '(n,npd)': ReactionInfo({183}, ('H1', 'H2')),
    '(n,npt)': ReactionInfo({184}, ('H1', 'H3')),
    '(n,ndt)': ReactionInfo({185}, ('H2', 'H3')),
    '(n,np3He)': ReactionInfo({186}, ('H1', 'He3')),
    '(n,nd3He)': ReactionInfo({187}, ('H2', 'He3')),
    '(n,nt3He)': ReactionInfo({188}, ('H3', 'He3')),
    '(n,nta)': ReactionInfo({189}, ('H3', 'He4')),
    '(n,2n2p)': ReactionInfo({190}, ('H1', 'H1')),
    '(n,p3He)': ReactionInfo({191}, ('H1', 'He3')),
    '(n,d3He)': ReactionInfo({192}, ('H2', 'He3')),
    '(n,3Hea)': ReactionInfo({193}, ('He3', 'He4')),
    '(n,4n2p)': ReactionInfo({194}, ('H1', 'H1')),
    '(n,4n2a)': ReactionInfo({195}, ('He4', 'He4')),
    '(n,4npa)': ReactionInfo({196}, ('H1', 'He4')),
    '(n,3p)': ReactionInfo({197}, ('H1', 'H1', 'H1')),
    '(n,n3p)': ReactionInfo({198}, ('H1', 'H1', 'H1')),
    '(n,3n2pa)': ReactionInfo({199}, ('H1', 'H1', 'He4')),
    '(n,5n2p)': ReactionInfo({200}, ('H1', 'H1')),
}

__all__ = ["Chain", "REACTIONS"]


def _load_isomeric_branching_targets(root):
    """Load isomeric branching targets, LFS values, Q values, and embedded ratios.

    Returns (targets_data, lfs_data, q_data, embedded_data), each dict or None.

    Both chain-XML shapes are accepted for a branched ``<reaction>``:

    * legacy: scalar ``target``/``Q`` on the ``<reaction>`` element and an
      ``<isomeric_branching targets=... gendf_lfs=.../>`` child with no Q;
    * folded: the ``<reaction>`` carries only ``type`` and the child carries the
      per-pathway ``Q`` (or forward-compatible ``q_values``) parallel-list
      alongside ``targets``/``gendf_lfs``.

    The per-pathway Q list is retained in ``q_data`` purely so the writer can
    re-emit it via ``str()`` at full float precision (an exact round-trip); it
    is not consumed at runtime (Q does not enter :meth:`form_rxn_matrix`).
    """
    from openmc._xml import get_text

    targets_data = {}
    lfs_data = {}
    q_data = {}
    embedded_data = {}

    for nuclide_elem in root.findall('nuclide'):
        nuc_name = get_text(nuclide_elem, 'name')
        if not nuc_name:
            continue

        nuc_reactions = {}
        nuc_lfs = {}
        nuc_q = {}

        for reaction_elem in nuclide_elem.findall('reaction'):
            rx_type = reaction_elem.get('type')
            if not rx_type:
                continue

            iso_elem = reaction_elem.find('isomeric_branching')
            if iso_elem is not None:
                targets_attr = iso_elem.get('targets', '')
                targets = targets_attr.split()
                if targets:
                    nuc_reactions[rx_type] = targets
                    lfs_attr = iso_elem.get('gendf_lfs', '')
                    if lfs_attr:
                        try:
                            lfs_vals = [int(v) for v in lfs_attr.split()]
                        except ValueError:
                            warn(f"Malformed gendf_lfs '{lfs_attr}' for "
                                 f"{nuc_name}/{rx_type}, ignoring LFS")
                            lfs_vals = None
                        if lfs_vals is None:
                            pass
                        elif len(lfs_vals) == len(targets):
                            nuc_lfs[rx_type] = lfs_vals
                        else:
                            warn(
                                f"gendf_lfs count ({len(lfs_vals)}) != "
                                f"target count ({len(targets)}) for "
                                f"{nuc_name}/{rx_type}, ignoring LFS")
                    # Per-pathway Q parallel-list (folded form). Accept both the
                    # GENDF-fork ``Q`` attribute and the PENDF-style
                    # ``q_values``. A legacy child has neither -- the scalar Q on
                    # the <reaction> is used by the writer's replicate fallback.
                    q_attr = iso_elem.get('Q')
                    if q_attr is None:
                        q_attr = iso_elem.get('q_values')
                    if q_attr:
                        q_vals = q_attr.split()
                        if len(q_vals) == len(targets):
                            nuc_q[rx_type] = q_vals
                        else:
                            warn(
                                f"isomeric_branching Q count ({len(q_vals)}) != "
                                f"target count ({len(targets)}) for "
                                f"{nuc_name}/{rx_type}, ignoring Q")
                continue

            # Legacy <isomeric_yields> with embedded ratios. These blocks are
            # machine-generated (GENDF chain patcher), so a malformed ratio
            # block means a corrupted/hand-edited file -- raise loudly rather
            # than silently degrade to flags form and lose the ratios.
            legacy_elem = reaction_elem.find('isomeric_yields')
            if legacy_elem is not None:
                targets_elem = legacy_elem.find('targets')
                if targets_elem is None or not targets_elem.text:
                    continue
                targets = targets_elem.text.split()
                if not targets:
                    continue

                ctx = f"<isomeric_yields> for {nuc_name}/{rx_type}"
                energies_elem = legacy_elem.find('energies')
                branching_elem = legacy_elem.find('branching_ratios')
                if (energies_elem is None or not energies_elem.text
                        or branching_elem is None or not branching_elem.text):
                    raise ValueError(
                        f"Malformed {ctx}: missing <energies> or "
                        "<branching_ratios> text")

                try:
                    energies = np.array([float(e) for e in
                                         energies_elem.text.split()])
                    lines = []
                    for line in branching_elem.text.strip().split('\n'):
                        line = line.strip()
                        if line and not line.startswith('<!--'):
                            if '<!--' in line:
                                line = line[:line.index('<!--')].strip()
                            if line:
                                lines.append(line)
                    ratio_rows = [np.array([float(r) for r in line.split()])
                                  for line in lines]
                except ValueError as exc:
                    raise ValueError(
                        f"Malformed {ctx}: non-numeric ratio text ({exc})"
                    ) from exc

                if len(ratio_rows) != len(targets):
                    raise ValueError(
                        f"Malformed {ctx}: {len(ratio_rows)} ratio rows != "
                        f"{len(targets)} targets")
                br = {}
                for target, ratios in zip(targets, ratio_rows):
                    if len(ratios) != len(energies):
                        raise ValueError(
                            f"Malformed {ctx}: target {target} has "
                            f"{len(ratios)} ratios != {len(energies)} energies")
                    br[target] = ratios

                nuc_reactions[rx_type] = targets
                embedded_data[(nuc_name, rx_type)] = {
                    'energies': energies,
                    'targets': targets,
                    'branching_ratios': br,
                }

        if nuc_reactions:
            targets_data[nuc_name] = nuc_reactions
        if nuc_lfs:
            lfs_data[nuc_name] = nuc_lfs
        if nuc_q:
            q_data[nuc_name] = nuc_q

    targets_out = targets_data if targets_data else None
    lfs_out = lfs_data if lfs_data else None
    q_out = q_data if q_data else None
    embedded_out = embedded_data if embedded_data else None
    return targets_out, lfs_out, q_out, embedded_out


def _write_embedded_yields(reaction_elem, data):
    """Re-emit stored energy-dependent ratios as an ``<isomeric_yields>`` child."""
    yields_elem = ET.SubElement(reaction_elem, 'isomeric_yields')
    yields_elem.set('type', 'energy_dependent')

    energies = data['energies']
    targets = data['targets']
    ratios = data['branching_ratios']

    energies_elem = ET.SubElement(yields_elem, 'energies')
    energies_elem.text = ' '.join(f'{e:.6e}' for e in energies)

    targets_elem = ET.SubElement(yields_elem, 'targets')
    targets_elem.text = ' '.join(targets)

    ratios_elem = ET.SubElement(yields_elem, 'branching_ratios')
    lines = []
    for target in targets:
        vals = ratios.get(target)
        if vals is None:
            vals = [0.0] * len(energies)
        lines.append('          ' + ' '.join(f'{r:.6e}' for r in vals))
    ratios_elem.text = '\n' + '\n'.join(lines) + '\n        '


def _write_isomeric_branching_targets(root_elem, targets_data, lfs_data=None,
                                      q_data=None, embedded_data=None):
    """Write isomeric branching targets, LFS values, and per-pathway Q to XML.

    Emits the folded form: for every BRANCHED reaction the scalar ``target`` and
    ``Q`` are stripped from the ``<reaction>`` element and instead carried on the
    ``<isomeric_branching>`` child as parallel ``targets``/``gendf_lfs``/``Q``
    lists (one entry per pathway, ground first). Unbranched reactions are left
    untouched (they keep their scalar ``target``/``Q``).

    Exactly ONE ``<isomeric_branching>`` child is emitted per (nuclide, reaction
    type). A reaction represented by MULTIPLE same-type elements (official-chain
    style, one element per static pathway) gets the child on its first element
    and is NOT folded -- every entry keeps its scalar
    ``target``/``Q``/``branching_ratio`` (the legacy shape the reader accepts)
    so the static split round-trips.

    The child's ``Q`` list is the stored per-pathway Q (``q_data``) when known;
    for a chain loaded from the legacy shape -- which carried only a single
    reaction-level Q -- that scalar Q is REPLICATED across every target (the code
    already assumes "Q value is independent of target state", see
    :meth:`set_branch_ratios`), so no per-pathway Q is invented.

    Reactions carrying embedded energy-dependent ratios (``embedded_data``) are
    re-emitted as the legacy ``<isomeric_yields>`` child instead of the flags
    form, keeping their scalar ``target``/``Q`` so the ratios round-trip.
    """
    targets_data = targets_data or {}
    embedded_data = embedded_data or {}
    if not targets_data and not embedded_data:
        return

    nuc_names = set(targets_data) | {parent for parent, _rx in embedded_data}

    for nuclide_elem in root_elem.findall('nuclide'):
        nuc_name = nuclide_elem.get('name')
        if not nuc_name or nuc_name not in nuc_names:
            continue

        nuc_targets = targets_data.get(nuc_name, {})

        # Count same-type elements so multi-entry branched reactions are
        # emitted once and never folded
        reaction_elems = nuclide_elem.findall('reaction')
        type_counts = defaultdict(int)
        for elem in reaction_elems:
            type_counts[elem.get('type')] += 1
        emitted = set()

        for reaction_elem in reaction_elems:
            rx_type = reaction_elem.get('type')
            if not rx_type:
                continue
            has_embedded = (nuc_name, rx_type) in embedded_data
            if rx_type not in nuc_targets and not has_embedded:
                continue
            if rx_type in emitted:
                continue
            emitted.add(rx_type)

            # Embedded ratios take precedence: emit the legacy <isomeric_yields>
            # form (energies and ratios at %.6e, i.e. 7 significant figures) and
            # leave the reaction's scalar target/Q untouched.
            if has_embedded:
                _write_embedded_yields(reaction_elem,
                                       embedded_data[(nuc_name, rx_type)])
                continue

            targets = nuc_targets[rx_type]

            # Fold: the branched reaction's per-pathway data lives ONLY on the
            # child. Capture the scalar Q (for the replicate fallback) then drop
            # the scalar target/Q from the <reaction> element. Multi-entry
            # reactions keep their scalars (legacy shape) -- see docstring.
            scalar_q = reaction_elem.get('Q')
            if type_counts[rx_type] == 1:
                reaction_elem.attrib.pop('target', None)
                reaction_elem.attrib.pop('Q', None)

            iso_elem = ET.SubElement(reaction_elem, 'isomeric_branching')
            iso_elem.set('targets', ' '.join(targets))
            # Write gendf_lfs
            if (lfs_data is not None
                    and nuc_name in lfs_data
                    and rx_type in lfs_data[nuc_name]):
                lfs_vals = lfs_data[nuc_name][rx_type]
                iso_elem.set('gendf_lfs',
                             ' '.join(str(v) for v in lfs_vals))

            # Write the per-pathway Q parallel-list. Prefer stored per-pathway Q;
            # otherwise replicate the reaction's scalar Q across all targets.
            q_vals = None
            if (q_data is not None
                    and nuc_name in q_data
                    and rx_type in q_data[nuc_name]):
                stored = q_data[nuc_name][rx_type]
                if len(stored) == len(targets):
                    q_vals = [str(v) for v in stored]
            if q_vals is None and scalar_q is not None:
                q_vals = [scalar_q] * len(targets)
            if q_vals is not None:
                iso_elem.set('Q', ' '.join(q_vals))


def replace_missing(product, decay_data):
    """Replace missing product with suitable decay daughter.

    Parameters
    ----------
    product : str
        Name of product in GNDS format, e.g. 'Y86_m1'.
    decay_data : dict
        Dictionary of decay data

    Returns
    -------
    product : str
        Replacement for missing product in GNDS format.

    """
    # Determine atomic number, mass number, and metastable state
    Z, A, state = openmc.data.zam(product)
    symbol = openmc.data.ATOMIC_SYMBOL[Z]

    # Replace neutron with nothing
    if Z == 0:
        return None

    # First check if ground state is available
    if state:
        product = f'{symbol}{A}'

    # Find isotope with longest half-life
    half_life = 0.0
    for nuclide, data in decay_data.items():
        m = re.match(r'{}(\d+)(?:_m\d+)?'.format(symbol), nuclide)
        if m:
            # If we find a stable nuclide, stop search
            if data.nuclide['stable']:
                mass_longest_lived = int(m.group(1))
                break
            if data.half_life.nominal_value > half_life:
                mass_longest_lived = int(m.group(1))
                half_life = data.half_life.nominal_value

    # If mass number of longest-lived isotope is less than that of missing
    # product, assume it undergoes beta-. Otherwise assume beta+.
    beta_minus = (mass_longest_lived < A)

    # Iterate until we find an existing nuclide
    while product not in decay_data:
        if Z > 98:
            # Assume alpha decay occurs for Z=99 and above
            Z -= 2
            A -= 4
        else:
            # Otherwise assume a beta- or beta+
            if beta_minus:
                Z += 1
            else:
                Z -= 1
        product = f'{openmc.data.ATOMIC_SYMBOL[Z]}{A}'

    return product


def replace_missing_fpy(actinide, fpy_data, decay_data):
    """Replace missing fission product yields

    Parameters
    ----------
    actinide : str
        Name of actinide missing FPY data
    fpy_data : dict
        Dictionary of FPY data
    decay_data : dict
        Dictionary of decay data

    Returns
    -------
    str
        Actinide that can be used as replacement for FPY purposes

    """

    # Check if metastable state has data (e.g., Am242m)
    Z, A, m = zam(actinide)
    if m == 0:
        metastable = gnds_name(Z, A, 1)
        if metastable in fpy_data:
            return metastable

    # Try increasing Z, holding N constant
    isotone = actinide
    while isotone in decay_data:
        Z += 1
        A += 1
        isotone = gnds_name(Z, A, 0)
        if isotone in fpy_data:
            return isotone

    # Try decreasing Z, holding N constant
    isotone = actinide
    while isotone in decay_data:
        Z -= 1
        A -= 1
        isotone = gnds_name(Z, A, 0)
        if isotone in fpy_data:
            return isotone

    # If all else fails, use U235 yields
    return 'U235'


class Chain:
    """Full representation of a depletion chain.

    A depletion chain can be created by using the :meth:`from_endf` method which
    requires a list of ENDF incident neutron, decay, and neutron fission product
    yield sublibrary files. The depletion chain used during a depletion
    simulation is indicated by either an argument to
    :class:`openmc.deplete.CoupledOperator` or
    :class:`openmc.deplete.IndependentOperator`, or through
    openmc.config['chain_file'].

    Attributes
    ----------
    nuclides : list of openmc.deplete.Nuclide
        Nuclides present in the chain.
    reactions : list of str
        Reactions that are tracked in the depletion chain
    nuclide_dict : dict of str to int
        Maps a nuclide name to an index in nuclides.
    stable_nuclides : list of openmc.deplete.Nuclide
        List of stable nuclides available in the chain.
    unstable_nuclides : list of openmc.deplete.Nuclide
        List of unstable nuclides available in the chain.
    fission_yields : None or iterable of dict
        List of effective fission yields for materials. Each dictionary
        should be of the form ``{parent: {product: yield}}`` with
        types ``{str: {str: float}}``, where ``yield`` is the fission product
        yield for isotope ``parent`` producing isotope ``product``.
        A single entry indicates yields are constant across all materials.
        Otherwise, an entry can be added for each material to be burned.
        Ordering should be identical to how the operator orders reaction
        rates for burnable materials.
    """

    def __init__(self):
        self.nuclides: List[Nuclide] = []
        self.reactions = []
        self.nuclide_dict = {}
        self._fission_yields = None
        self._decay_matrix = None
        self.isomeric_branching_targets = None
        self.isomeric_branching_lfs = None
        self.isomeric_branching_q = None
        self.isomeric_branching_embedded = None
        self.reduce_pruned_targets = None

    def __contains__(self, nuclide):
        return nuclide in self.nuclide_dict

    def __getitem__(self, name):
        """Get a Nuclide by name."""
        return self.nuclides[self.nuclide_dict[name]]

    def __len__(self):
        """Number of nuclides in chain."""
        return len(self.nuclides)

    @property
    def stable_nuclides(self) -> List[Nuclide]:
        """List of stable nuclides available in the chain"""
        return [nuc for nuc in self.nuclides if nuc.half_life is None]

    @property
    def unstable_nuclides(self) -> List[Nuclide]:
        """List of unstable nuclides available in the chain"""
        return [nuc for nuc in self.nuclides if nuc.half_life is not None]

    def add_nuclide(self, nuclide: Nuclide):
        """Add a nuclide to the depletion chain

        Parameters
        ----------
        nuclide : openmc.deplete.Nuclide
            Nuclide to add

        """
        _invalidate_chain_cache(self)
        self.nuclide_dict[nuclide.name] = len(self.nuclides)
        self.nuclides.append(nuclide)

        # Check for reaction paths
        for rx in nuclide.reactions:
            if rx.type not in self.reactions:
                self.reactions.append(rx.type)

    @classmethod
    def from_endf(cls, decay_files, fpy_files, neutron_files,
        reactions=('(n,2n)', '(n,3n)', '(n,4n)', '(n,gamma)', '(n,p)', '(n,a)'),
        progress=True
    ):
        """Create a depletion chain from ENDF files.

        String arguments in ``decay_files``, ``fpy_files``, and
        ``neutron_files`` will be treated as file names to be read.
        Alternatively, :class:`openmc.data.endf.Evaluation` or
        ``endf.Material`` instances can be included in these arguments.

        Parameters
        ----------
        decay_files : list of str, openmc.data.endf.Evaluation, or endf.Material
            List of ENDF decay sub-library files
        fpy_files : list of str, openmc.data.endf.Evaluation, or endf.Material
            List of ENDF neutron-induced fission product yield sub-library files
        neutron_files : list of str, openmc.data.endf.Evaluation, or endf.Material
            List of ENDF neutron reaction sub-library files
        reactions : iterable of str, optional
            Transmutation reactions to include in the depletion chain, e.g.,
            `["(n,2n)", "(n,gamma)"]`. Note that fission is always included if
            it is present. A complete listing of transmutation reactions can be
            found in :data:`openmc.deplete.chain.REACTIONS`.

            .. versionadded:: 0.12.1
        progress : bool, optional
            Flag to print status messages during processing. Does not
            effect warning messages

        Returns
        -------
        Chain

        Notes
        -----
        When an actinide is missing fission product yield (FPY) data, yields will
        copied from a parent isotope, found according to:

        1. If the nuclide is in a ground state and a metastable state exists with
           fission yields, copy the yields from the metastable
        2. Find an isotone (same number of neutrons) and copy those yields
        3. Copy the yields of U235 if the previous two checks fail

        """
        transmutation_reactions = reactions

        # Create dictionary mapping target to filename
        if progress:
            print('Processing neutron sub-library files...')
        reactions = {}
        for f in neutron_files:
            evaluation = openmc.data.endf.as_evaluation(f)
            name = evaluation.gnds_name
            reactions[name] = {}
            for mf, mt, nc, mod in evaluation.reaction_list:
                if mf == 3:
                    file_obj = StringIO(evaluation.section[3, mt])
                    openmc.data.endf.get_head_record(file_obj)
                    q_value = openmc.data.endf.get_cont_record(file_obj)[1]
                    reactions[name][mt] = q_value

        # Determine what decay and FPY nuclides are available
        if progress:
            print('Processing decay sub-library files...')
        decay_data = {}
        for f in decay_files:
            data = openmc.data.Decay(f)
            # Skip decay data for neutron itself
            if data.nuclide['atomic_number'] == 0:
                continue
            decay_data[data.nuclide['name']] = data

        if progress:
            print('Processing fission product yield sub-library files...')
        fpy_data = {}
        for f in fpy_files:
            data = openmc.data.FissionProductYields(f)
            fpy_data[data.nuclide['name']] = data

        if progress:
            print('Creating depletion_chain...')
        missing_daughter = []
        missing_rx_product = []
        missing_fpy = []
        missing_fp = []

        chain = cls()
        for idx, parent in enumerate(sorted(decay_data, key=openmc.data.zam)):
            data = decay_data[parent]

            nuclide = Nuclide(parent)

            if not data.nuclide['stable'] and data.half_life.nominal_value != 0.0:
                nuclide.half_life = data.half_life.nominal_value
                nuclide.decay_energy = data.decay_energy.nominal_value
                branch_ratios = []
                branch_ids = []
                for mode in data.modes:
                    type_ = ','.join(mode.modes)
                    if mode.daughter in decay_data:
                        target = mode.daughter
                    else:
                        print('missing {} {} {}'.format(
                            parent, type_, mode.daughter))
                        target = replace_missing(mode.daughter, decay_data)
                    br = mode.branching_ratio.nominal_value
                    branch_ratios.append(br)
                    branch_ids.append((type_, target))

                if not math.isclose(sum(branch_ratios), 1.0):
                    max_br = max(branch_ratios)
                    max_index = branch_ratios.index(max_br)

                    # Adjust maximum branching ratio so they sum to unity
                    new_br = max_br - sum(branch_ratios) + 1.0
                    branch_ratios[max_index] = new_br
                    assert math.isclose(sum(branch_ratios), 1.0)

                # Append decay modes
                for br, (type_, target) in zip(branch_ratios, branch_ids):
                    nuclide.add_decay_mode(type_, target, br)

                nuclide.sources = data.sources

            fissionable = False
            if parent in reactions:
                reactions_available = set(reactions[parent].keys())
                for name in transmutation_reactions:
                    mts = REACTIONS[name].mts
                    delta_A, delta_Z = openmc.data.DADZ[name]
                    if mts & reactions_available:
                        A = data.nuclide['mass_number'] + delta_A
                        Z = data.nuclide['atomic_number'] + delta_Z
                        daughter = f'{openmc.data.ATOMIC_SYMBOL[Z]}{A}'

                        if daughter not in decay_data:
                            daughter = replace_missing(daughter, decay_data)
                            if daughter is None:
                                missing_rx_product.append((parent, name, daughter))

                        # Store Q value
                        for mt in sorted(mts):
                            if mt in reactions[parent]:
                                q_value = reactions[parent][mt]
                                break
                        else:
                            q_value = 0.0

                        nuclide.add_reaction(name, daughter, q_value, 1.0)

                if any(mt in reactions_available for mt in openmc.data.FISSION_MTS):
                    q_value = reactions[parent][18]
                    nuclide.add_reaction('fission', None, q_value, 1.0)
                    fissionable = True

            if fissionable:
                if parent in fpy_data:
                    fpy = fpy_data[parent]

                    if fpy.energies is not None:
                        yield_energies = fpy.energies
                    else:
                        yield_energies = [0.0]

                    yield_data = {}
                    for E, yield_table in zip(yield_energies, fpy.independent):
                        yield_replace = 0.0
                        yields = defaultdict(float)
                        for product, y in yield_table.items():
                            # Handle fission products that have no decay data
                            if product not in decay_data:
                                daughter = replace_missing(product, decay_data)
                                product = daughter
                                yield_replace += y.nominal_value

                            yields[product] += y.nominal_value

                        if yield_replace > 0.0:
                            missing_fp.append((parent, E, yield_replace))
                        yield_data[E] = yields

                    nuclide.yield_data = FissionYieldDistribution(yield_data)
                else:
                    nuclide._fpy = replace_missing_fpy(parent, fpy_data, decay_data)
                    missing_fpy.append((parent, nuclide._fpy))

            # Add nuclide to chain
            chain.add_nuclide(nuclide)

        # Replace missing FPY data
        for nuclide in chain.nuclides:
            if hasattr(nuclide, '_fpy'):
                nuclide.yield_data = chain[nuclide._fpy].yield_data

        # Display warnings
        if missing_daughter:
            print('The following decay modes have daughters with no decay data:')
            for mode in missing_daughter:
                print(f'  {mode}')
            print('')

        if missing_rx_product:
            print('The following reaction products have no decay data:')
            for vals in missing_rx_product:
                print('{} {} -> {}'.format(*vals))
            print('')

        if missing_fpy:
            print('The following fissionable nuclides have no fission product yields:')
            for parent, replacement in missing_fpy:
                print(f'  {parent}, replaced with {replacement}')
            print('')

        if missing_fp:
            print('The following nuclides have fission products with no decay data:')
            for vals in missing_fp:
                print('  {}, E={} eV (total yield={})'.format(*vals))

        return chain

    @classmethod
    def from_xml(cls, filename, fission_q=None):
        """Reads a depletion chain XML file.

        Parameters
        ----------
        filename : str
            The path to the depletion chain XML file.
        fission_q : dict, optional
            Dictionary of nuclides and their fission Q values [eV].
            If not given, values will be pulled from ``filename``

        """
        chain = cls()

        if fission_q is not None:
            check_type("fission_q", fission_q, Mapping)
        else:
            fission_q = {}

        # Load XML tree
        root = ET.parse(str(filename))

        for i, nuclide_elem in enumerate(root.findall('nuclide')):
            this_q = fission_q.get(get_text(nuclide_elem, "name"))

            nuc = Nuclide.from_xml(nuclide_elem, root, this_q)
            chain.add_nuclide(nuc)

        # Store path of XML file (used for handling cache invalidation)
        chain._xml_path = str(Path(filename).resolve())

        # Load isomeric branching data if present
        chain.isomeric_branching_targets, chain.isomeric_branching_lfs, \
            chain.isomeric_branching_q, chain.isomeric_branching_embedded = \
            _load_isomeric_branching_targets(root)

        # Pre-compute isomeric families cache for faster reduction operations
        chain._build_isomeric_families_cache()

        return chain

    def export_to_xml(self, filename):
        """Writes a depletion chain XML file.

        This method exports the complete depletion chain to XML format,
        including all nuclides, reactions, decay modes, fission yields,
        and energy-dependent isomeric branching data (if present).

        .. versionadded:: 0.15.3
            Isomeric branching data is now included in exported XML files.
            Flag-only branching (targets, ``gendf_lfs``, per-pathway Q)
            round-trips exactly, while embedded energy-dependent ratios
            round-trip to 7 significant figures (``%.6e``).

        Parameters
        ----------
        filename : str
            The path to the depletion chain XML file.
        """

        root_elem = ET.Element('depletion_chain')
        for nuclide in self.nuclides:
            root_elem.append(nuclide.to_xml_element())

        # Write isomeric branching data if present. Flags (targets + gendf_lfs)
        # round-trip exactly; embedded energy-dependent ratios round-trip to 7
        # significant figures (%.6e).
        if (self.isomeric_branching_targets is not None
                or self.isomeric_branching_embedded is not None):
            _write_isomeric_branching_targets(root_elem,
                                              self.isomeric_branching_targets,
                                              self.isomeric_branching_lfs,
                                              self.isomeric_branching_q,
                                              self.isomeric_branching_embedded)

        tree = ET.ElementTree(root_elem)
        tree.write(str(filename), encoding='utf-8', pretty_print=True)

    def get_default_fission_yields(self):
        """Return fission yields at lowest incident neutron energy

        Used as the default set of fission yields for :meth:`form_matrix`
        if ``fission_yields`` are not provided

        Returns
        -------
        fission_yields : dict
            Dictionary of ``{parent: {product: f_yield}}``
            where ``parent`` and ``product`` are both string
            names of nuclides with yield data and ``f_yield``
            is a float for the fission yield.
        """
        out = defaultdict(dict)
        for nuc in self.nuclides:
            if nuc.yield_data is None:
                continue
            yield_obj = nuc.yield_data[min(nuc.yield_energies)]
            out[nuc.name] = dict(yield_obj)
        return out

    @property
    def decay_matrix(self):
        """Sparse CSC decay transmutation matrix.

        Contains only terms from radioactive decay: diagonal loss terms
        and off-diagonal gain terms (branching ratios, alpha/proton
        production). Independent of reaction rates, so computed once and
        cached.

        See Also
        --------
        :meth:`form_rxn_matrix`, :meth:`form_matrix`
        """
        if self._decay_matrix is None:
            n = len(self)
            rows, cols, vals = [], [], []

            def setval(i, j, val):
                rows.append(i)
                cols.append(j)
                vals.append(val)

            for i, nuc in enumerate(self.nuclides):
                # Loss from radioactive decay
                if nuc.half_life is not None:
                    decay_constant = math.log(2) / nuc.half_life
                    if decay_constant != 0.0:
                        setval(i, i, -decay_constant)

                # Gain from radioactive decay
                if nuc.n_decay_modes != 0:
                    for decay_type, target, branching_ratio in nuc.decay_modes:
                        branch_val = branching_ratio * decay_constant

                        # Allow for total annihilation for debug purposes
                        if branch_val != 0.0:
                            if target is not None:
                                k = self.nuclide_dict[target]
                                setval(k, i, branch_val)

                            # Produce alphas and protons from decay
                            if 'alpha' in decay_type:
                                k = self.nuclide_dict.get('He4')
                                if k is not None:
                                    count = decay_type.count('alpha')
                                    setval(k, i, count * branch_val)
                            elif 'p' in decay_type:
                                k = self.nuclide_dict.get('H1')
                                if k is not None:
                                    count = decay_type.count('p')
                                    setval(k, i, count * branch_val)

            self._decay_matrix = csc_array((vals, (rows, cols)), shape=(n, n))
        return self._decay_matrix

    def form_rxn_matrix(self, rates, fission_yields=None, isomeric_branching=None):
        """Form the reaction-rate portion of the transmutation matrix.

        Builds only the terms that depend on reaction rates: transmutation
        reactions and fission product yields. Does not include radioactive
        decay terms (see :attr:`decay_matrix`).

        Parameters
        ----------
        rates : numpy.ndarray
            2D array indexed by (nuclide, reaction)
        fission_yields : dict, optional
            Option to use a custom set of fission yields. Expected
            to be of the form ``{parent : {product : f_yield}}``
            with string nuclide names for ``parent`` and ``product``,
            and ``f_yield`` as the respective fission yield
        isomeric_branching : dict, optional
            Isomeric branching ratios to use. Expected to be of the form
            ``{parent: {reaction: {target: branching_ratio}}}``

        Returns
        -------
        scipy.sparse.csc_array
            Sparse matrix representing reaction-rate terms.

        See Also
        --------
        :attr:`decay_matrix`, :meth:`form_matrix`
        """
        reactions = set()
        iso_applied = set()
        n = len(self)

        # Accumulate indices/values and then create the matrix at the end to
        # avoid expensive index checks scipy otherwise does.
        rows, cols, vals = [], [], []

        def setval(i, j, val):
            rows.append(i)
            cols.append(j)
            vals.append(val)

        if fission_yields is None:
            fission_yields = self.get_default_fission_yields()

        # Save local variables to avoid attribute lookups in loop
        index_nuc = rates.index_nuc
        index_rx = rates.index_rx
        iso_branching = isomeric_branching or {}

        for i, nuc in enumerate(self.nuclides):
            if nuc.name not in index_nuc:
                continue

            nuc_ind = index_nuc[nuc.name]
            nuc_rates = rates[nuc_ind, :]
            nuc_iso = iso_branching.get(nuc.name, {})

            for r_type, target, _, br in nuc.reactions:
                r_id = index_rx[r_type]
                path_rate = nuc_rates[r_id]

                # Loss term -- make sure we only count loss once for
                # reactions with branching ratios
                if r_type not in reactions:
                    reactions.add(r_type)
                    if path_rate != 0.0:
                        setval(i, i, -path_rate)

                # Gain term; allow for total annihilation for debug purposes
                if r_type != 'fission':
                    if target is not None and path_rate != 0.0:
                        # Isomeric branching only with a usable distribution;
                        # empty/all-zero/NaN dicts fall to the static target
                        iso_dist = nuc_iso.get(r_type)
                        if iso_dist and any(math.isfinite(v) and v > 0.0
                                            for v in iso_dist.values()):
                            # The distribution replaces the chain's static
                            # split for ALL same-type entries -- apply it once
                            # per reaction type (mirrors the loss-term guard)
                            # so multi-entry branched reactions (e.g. official
                            # chains with ground + metastable entries) do not
                            # double-count production
                            if r_type not in iso_applied:
                                iso_applied.add(r_type)
                                # Apply isomeric branching with strict validation
                                # br from isomeric data represents the complete branching
                                # distribution (sums to 1.0)
                                total_ratio = 0.0
                                missing_targets = []
                                for iso_target, iso_br in iso_dist.items():
                                    # Validate ratio value (guard against NaN/Inf from data corruption)
                                    if not math.isfinite(iso_br):
                                        warn(f"Invalid isomeric branching ratio for {nuc.name} {r_type} -> "
                                             f"{iso_target}: {iso_br}. Treating as 0.0.")
                                        iso_br = 0.0
                                    if iso_target not in self.nuclide_dict:
                                        # Critical error - missing target will cause mass conservation violation
                                        missing_targets.append(iso_target)
                                    else:
                                        k = self.nuclide_dict[iso_target]
                                        setval(k, i, path_rate * iso_br)
                                        total_ratio += iso_br
                                # Raise error if any targets are missing
                                if missing_targets:
                                    raise KeyError(
                                        f"Isomeric branching for {nuc.name} {r_type} references "
                                        f"target(s) not in chain: {missing_targets}. "
                                        f"This would cause mass conservation violations. "
                                        f"Check chain file or isomeric branching data.")
                                # Verify branching ratios sum to approximately 1.0
                                if total_ratio > 0.0 and not (0.98 <= total_ratio <= 1.02):
                                    warn(f"Isomeric branching ratios for {nuc.name} {r_type} "
                                         f"sum to {total_ratio:.4f}, not 1.0. This may indicate "
                                         f"incomplete or incorrect branching data.")
                        else:
                            if iso_dist is not None:
                                warn(f"Isomeric branching for {nuc.name} "
                                     f"{r_type} has no usable (finite, "
                                     f"positive) ratios; using the chain's "
                                     f"static target.")
                            # Original single target
                            k = self.nuclide_dict[target]
                            setval(k, i, path_rate * br)
                    # Determine light nuclide production, e.g., (n,d) should
                    # produce H2
                    light_nucs = REACTIONS[r_type].secondaries
                    for light_nuc in light_nucs:
                        k = self.nuclide_dict.get(light_nuc)
                        if k is not None:
                            setval(k, i, path_rate * br)

                else:
                    for product, y in fission_yields[nuc.name].items():
                        yield_val = y * path_rate
                        if yield_val != 0.0:
                            k = self.nuclide_dict[product]
                            setval(k, i, yield_val)

            reactions.clear()
            iso_applied.clear()

        return csc_array((vals, (rows, cols)), shape=(n, n))

    def form_matrix(self, rates, fission_yields=None, isomeric_branching=None):
        """Form the full transmutation matrix (decay + reactions).

        Parameters
        ----------
        rates : numpy.ndarray
            2D array indexed by (nuclide, reaction)
        fission_yields : dict, optional
            Option to use a custom set of fission yields. Expected
            to be of the form ``{parent : {product : f_yield}}``
            with string nuclide names for ``parent`` and ``product``,
            and ``f_yield`` as the respective fission yield
        isomeric_branching : dict, optional
            Isomeric branching ratios to use. Expected to be of the form
            ``{parent: {reaction: {target: branching_ratio}}}``

        Returns
        -------
        scipy.sparse.csc_array
            Sparse matrix representing depletion.

        See Also
        --------
        :attr:`decay_matrix`, :meth:`form_rxn_matrix`,
        :meth:`get_default_fission_yields`
        """
        return self.decay_matrix + self.form_rxn_matrix(rates, fission_yields, isomeric_branching)

    def add_redox_term(self, matrix, buffer, oxidation_states):
        r"""Adds a redox term to the depletion matrix from data contained in
        the matrix itself and a few user-inputs.

        The redox term to add to the buffer nuclide :math:`N_j` can be written
        as:

        .. math::
            \frac{dN_j(t)}{dt} = \cdots - \frac{1}{OS_j}\sum_i N_i a_{ij}
            \cdot OS_i

        where :math:`OS` is the oxidation states vector and :math:`a_{ij}` the
        corresponding term in the Bateman matrix.

        Parameters
        ----------
        matrix : scipy.sparse.csc_array
            Sparse matrix representing depletion
        buffer : dict
            Dictionary of buffer nuclides used to maintain anoins net balance.
            Keys are nuclide names (strings) and values are their respective
            fractions (float) that collectively sum to 1.
        oxidation_states : dict
            User-defined oxidation states for elements. Keys are element symbols
            (e.g., 'H', 'He'), and values are their corresponding oxidation
            states as integers (e.g., +1, 0).
        Returns
        -------
        matrix : scipy.sparse.csc_array
            Sparse matrix with redox term added
        """
        # Elements list with the same size as self.nuclides
        elements = [re.split(r'\d+', nuc.name)[0] for nuc in self.nuclides]

        # Match oxidation states with all elements and add 0 if not data
        os = np.array([oxidation_states[elm] if elm in oxidation_states else 0
                       for elm in elements])

        # Buffer idx with nuclide index as value
        buffer_idx = {nuc: self.nuclide_dict[nuc] for nuc in buffer}
        array = matrix.toarray()
        redox_change = np.array([])

        # calculate the redox array
        for i in range(len(self)):
            # Net redox impact of reaction: multiply the i-th column of the
            # depletion matrix by the oxidation states
            redox_change = np.append(redox_change, sum(array[:, i]*os))

        # Subtract redox vector to the buffer nuclides in the matrix scaling by
        # their respective oxidation states
        for nuc, idx in buffer_idx.items():
            array[idx] -= redox_change * buffer[nuc] / os[idx]

        return csc_array(array)

    def form_rr_term(self, tr_rates, current_timestep, mats):
        """Function to form the transfer rate term matrices.

        .. versionadded:: 0.14.0

        Parameters
        ----------
        tr_rates : openmc.deplete.TransferRates
            Instance of openmc.deplete.TransferRates
        current_timestep : int
            Current timestep index
        mats : string or two-tuple of strings
            Two cases are possible:

            1) Material ID as string:
            Nuclide transfer only. In this case the transfer rate terms will be
            subtracted from the respective depletion matrix

            2) Two-tuple of material IDs as strings:
            Nuclide transfer from one material into another.
            The pair is assumed to be
            ``(destination_material, source_material)``, where
            ``destination_material`` and ``source_material`` are the nuclide
            receiving and losing materials, respectively.
            The transfer rate terms get placed in the final matrix with indexing
            position corresponding to the ID of the materials set.

        Returns
        -------
        scipy.sparse.csc_array
            Sparse matrix representing transfer term.

        """
        # Use DOK as intermediate representation
        n = len(self)
        matrix = dok_array((n, n))

        check_type("mats", mats, (tuple, str))
        if not isinstance(mats, str):
            check_type("mats", mats, tuple, str)
            check_length("mats", mats, 2, 2)
            dest_mat, mat = mats
        else:
            mat = mats
            dest_mat = None

        # Build transfer term
        components = tr_rates.get_components(mat, current_timestep, dest_mat)

        for i, nuc in enumerate(self.nuclides):
            elm = re.split(r'\d+', nuc.name)[0]
            if elm in components:
                key = elm
            elif nuc.name in components:
                key = nuc.name
            else:
                continue
            matrix[i, i] = sum(tr_rates.get_external_rate(mat, key, current_timestep, dest_mat))

        # Return CSC instead of DOK
        return matrix.tocsc()

    def form_ext_source_term(self, ext_source_rates, current_timestep, mat):
        """Function to form the external source rate term vectors.

        .. versionadded:: 0.15.3

        Parameters
        ----------
        ext_source_rates : openmc.deplete.ExternalSourceRates
            Instance of openmc.deplete.ExternalSourceRates
        current_timestep : int
            Current timestep index
        mat : string
            Material id

        Returns
        -------
        scipy.sparse.csc_array
            Sparse vector representing external source term.

        """
        if not ext_source_rates.get_components(mat, current_timestep):
            return
        # Use DOK as intermediate representation
        n = len(self)
        vector = dok_array((n, 1))
        components = ext_source_rates.get_components(mat, current_timestep)

        for i, nuc in enumerate(self.nuclides):
            # Build source term vector
            if nuc.name in components:
                vector[i] = sum(ext_source_rates.get_external_rate(
                    mat, nuc.name, current_timestep))

        # Return CSC instead of DOK
        return vector.tocsc()

    def get_branch_ratios(self, reaction="(n,gamma)"):
        """Return a dictionary with reaction branching ratios

        Parameters
        ----------
        reaction : str, optional
            Reaction name like ``"(n,gamma)"`` [default], or
            ``"(n,alpha)"``.

        Returns
        -------
        branches : dict
            nested dict of parent nuclide keys with reaction targets and
            branching ratios. Consider the capture, ``"(n,gamma)"``,
            reaction for Am241::

                {"Am241": {"Am242": 0.91, "Am242_m1": 0.09}}

        See Also
        --------
        :meth:`set_branch_ratios`
        """

        capt = {}
        for nuclide in self.nuclides:
            nuc_capt = {}
            for rx in nuclide.reactions:
                if rx.type == reaction and rx.branching_ratio != 1.0:
                    nuc_capt[rx.target] = rx.branching_ratio
            if len(nuc_capt) > 0:
                capt[nuclide.name] = nuc_capt
        return capt

    def set_branch_ratios(self, branch_ratios, reaction="(n,gamma)",
                          strict=True, tolerance=1e-5):
        """Set the branching ratios for a given reactions

        Parameters
        ----------
        branch_ratios : dict of {str: {str: float}}
            Capture branching ratios to be inserted.
            First layer keys are names of parent nuclides, e.g.
            ``"Am241"``. The branching ratios for these
            parents will be modified. Corresponding values are
            dictionaries of ``{target: branching_ratio}``
        reaction : str, optional
            Reaction name like ``"(n,gamma)"`` [default], or
            ``"(n, alpha)"``.
        strict : bool, optional
            Error control. If this evalutes to ``True``, then errors will
            be raised if inconsistencies are found. Otherwise, warnings
            will be raised for most issues.
        tolerance : float, optional
            Tolerance on the sum of all branching ratios for a
            single parent. Will be checked with::

                1 - tol < sum_br < 1 + tol

        Raises
        ------
        IndexError
            If no isotopes were found on the chain that have the requested
            reaction
        KeyError
            If ``strict`` evaluates to ``False`` and a parent isotope in
            ``branch_ratios`` does not exist on the chain
        AttributeError
            If ``strict`` evaluates to ``False`` and a parent isotope in
            ``branch_ratios`` does not have the requested reaction
        ValueError
            If ``strict`` evalutes to ``False`` and the sum of one parents
            branch ratios is outside  1 +/- ``tolerance``

        See Also
        --------
        :meth:`get_branch_ratios`
        """
        _invalidate_chain_cache(self)
        # Store some useful information through the validation stage

        sums = {}
        rxn_ix_map = {}
        grounds = {}

        tolerance = abs(tolerance)

        missing_parents = set()
        missing_products = {}
        missing_reaction = set()
        bad_sums = {}

        # Secondary products, like alpha particles, should not be modified
        secondary = REACTIONS[reaction].secondaries

        # Check for validity before manipulation

        for parent, sub in branch_ratios.items():
            if parent not in self:
                if strict:
                    raise KeyError(parent)
                missing_parents.add(parent)
                continue

            # Make sure all products are present in the chain

            prod_flag = False

            for product in sub:
                if product not in self:
                    if strict:
                        raise KeyError(product)
                    missing_products[parent] = product
                    prod_flag = True
                    break

            if prod_flag:
                continue

            # Make sure this nuclide has the reaction

            indexes = []
            for ix, rx in enumerate(self[parent].reactions):
                if rx.type == reaction and rx.target not in secondary:
                    indexes.append(ix)
                    if "_m" not in rx.target:
                        grounds[parent] = rx.target

            if len(indexes) == 0:
                if strict:
                    raise AttributeError(
                        f"Nuclide {parent} does not have {reaction} reactions")
                missing_reaction.add(parent)
                continue

            this_sum = sum(sub.values())
            # sum of branching ratios can be lower than 1 if no ground
            # target is given, but never greater
            if (this_sum >= 1 + tolerance or (grounds[parent] in sub
                                              and this_sum <= 1 - tolerance)):
                if strict:
                    msg = ("Sum of {} branching ratios for {} "
                           "({:7.3f}) outside tolerance of 1 +/- "
                           "{:5.3e}".format(
                               reaction, parent, this_sum, tolerance))
                    raise ValueError(msg)
                bad_sums[parent] = this_sum
            else:
                rxn_ix_map[parent] = indexes
                sums[parent] = this_sum

        if len(rxn_ix_map) == 0:
            raise IndexError(
                f"No {reaction} reactions found in this {self.__class__.__name__}")

        if len(missing_parents) > 0:
            warn("The following nuclides were not found in {}: {}".format(
                 self.__class__.__name__, ", ".join(sorted(missing_parents))))

        if len(missing_reaction) > 0:
            warn("The following nuclides did not have {} reactions: "
                 "{}".format(reaction, ", ".join(sorted(missing_reaction))))

        if len(missing_products) > 0:
            tail = (f"{k} -> {v}"
                    for k, v in sorted(missing_products.items()))
            warn("The following products were not found in the {} and "
                 "parents were unmodified: \n{}".format(
                     self.__class__.__name__, ", ".join(tail)))

        if len(bad_sums) > 0:
            tail = (f"{k}: {s:5.3f}"
                    for k, s in sorted(bad_sums.items()))
            warn("The following parent nuclides were given {} branch ratios "
                 "with a sum outside tolerance of 1 +/- {:5.3e}:\n{}".format(
                     reaction, tolerance, "\n".join(tail)))

        # Insert new ReactionTuples with updated branch ratios

        for parent_name, rxn_index in rxn_ix_map.items():

            parent = self[parent_name]
            new_ratios = branch_ratios[parent_name]
            rxn_index = rxn_ix_map[parent_name]

            # Assume Q value is independent of target state
            rxn_Q = parent.reactions[rxn_index[0]].Q

            # Remove existing reactions
            for ix in reversed(rxn_index):
                parent.reactions.pop(ix)

            # Add new reactions
            all_meta = True
            for target, br in new_ratios.items():
                all_meta = all_meta and ("_m" in target)
                parent.add_reaction(reaction, target, rxn_Q, br)

            # If branching ratios don't add to unity, add reaction to ground
            # with remainder of branching ratio
            if all_meta and sums[parent_name] != 1.0:
                ground_br = 1.0 - sums[parent_name]
                ground_target = grounds.get(parent_name)
                if ground_target is None:
                    pz, pa, pm = zam(parent_name)
                    ground_target = gnds_name(pz, pa + 1, 0)
                new_ratios[ground_target] = ground_br
                parent.add_reaction(reaction, ground_target, rxn_Q, ground_br)

    @property
    def fission_yields(self):
        if self._fission_yields is None:
            self._fission_yields = [self.get_default_fission_yields()]
        return self._fission_yields

    @fission_yields.setter
    def fission_yields(self, yields):
        _invalidate_chain_cache(self)
        if yields is not None:
            if isinstance(yields, Mapping):
                yields = [yields]
            check_type("fission_yields", yields, Iterable, Mapping)
        self._fission_yields = yields

    def validate(self, strict=True, quiet=False, tolerance=1e-4):
        """Search for possible inconsistencies

        The following checks are performed for all nuclides present:

            1) For all non-fission reactions, does the sum of branching
               ratios equal about one?
            2) For fission reactions, does the sum of fission yield
               fractions equal about two?

        Parameters
        ----------
        strict : bool, optional
            Raise exceptions at the first inconsistency if true.
            Otherwise mark a warning
        quiet : bool, optional
            Flag to suppress warnings and return immediately at
            the first inconsistency. Used only if
            ``strict`` does not evaluate to ``True``.
        tolerance : float, optional
            Absolute tolerance for comparisons. Used to compare computed
            value ``x`` to intended value ``y`` as::

                valid = (y - tolerance <= x <= y + tolerance)

        Returns
        -------
        valid : bool
            True if no inconsistencies were found

        Raises
        ------
        ValueError
            If ``strict`` evaluates to ``True`` and an inconistency was
            found

        See Also
        --------
        openmc.deplete.Nuclide.validate
        """
        check_type("tolerance", tolerance, Real)
        check_greater_than("tolerance", tolerance, 0.0, True)
        valid = True
        # Sort through nuclides by name
        for name in sorted(self.nuclide_dict):
            stat = self[name].validate(strict, quiet, tolerance)
            if quiet and not stat:
                return stat
            valid = valid and stat
        return valid

    def _get_base_name(self, nuclide_name):
        """Extract base name without isomeric state suffix.

        Parameters
        ----------
        nuclide_name : str
            Full nuclide name

        Returns
        -------
        str
            Base name without '_m1', '_m2', etc.

        Examples
        --------
        >>> chain._get_base_name('Ir191')
        'Ir191'
        >>> chain._get_base_name('Ir191_m1')
        'Ir191'
        """
        if '_m' in nuclide_name:
            return nuclide_name.rsplit('_m', 1)[0]
        return nuclide_name

    def _get_isomeric_siblings(self, nuclide_name):
        """Get all isomeric siblings (same element-mass, different states).

        Parameters
        ----------
        nuclide_name : str
            Nuclide name (e.g., 'Ir191', 'Ir191_m1')

        Returns
        -------
        list of str
            All siblings including ground and metastable states
        """
        if not hasattr(self, '_isomeric_families'):
            self._build_isomeric_families_cache()
        return self._isomeric_families.get(nuclide_name, [nuclide_name])

    def _build_isomeric_families_cache(self):
        """Build cache of isomeric families for fast lookup."""
        self._isomeric_families = {}

        # Group nuclides by base name
        families = {}
        for nuc in self.nuclides:
            base = self._get_base_name(nuc.name)
            if base not in families:
                families[base] = []
            families[base].append(nuc.name)

        # Map each nuclide to its family
        for family_list in families.values():
            for nuc_name in family_list:
                self._isomeric_families[nuc_name] = family_list

    def _expand_with_isomeric_siblings(self, isotope_set):
        """Expand isotope set to include isomeric siblings and branching targets.

        This method adds:
        1. All isomeric siblings for every nuclide in the set
        2. If isomeric_branching_targets exists: all branching targets and their siblings

        This ensures that when ``keep_isomeric_siblings=True`` is used during
        chain reduction, no isomeric branching targets are excluded, which would
        otherwise cause renormalization warnings.

        Parameters
        ----------
        isotope_set : set of str
            Set of nuclide names to expand (modified in-place).

        Returns
        -------
        set of str
            The expanded isotope set (same object as input, modified in-place).
        """
        to_check = list(isotope_set)
        checked = set()

        while to_check:
            nuc_name = to_check.pop()
            if nuc_name in checked:
                continue
            checked.add(nuc_name)

            # Add all siblings for this nuclide
            siblings = self._get_isomeric_siblings(nuc_name)
            for sibling in siblings:
                if sibling in self.nuclide_dict and sibling not in isotope_set:
                    isotope_set.add(sibling)
                    to_check.append(sibling)

            # If isomeric branching targets exist, also add targets and their siblings
            if self.isomeric_branching_targets and nuc_name in self.isomeric_branching_targets:
                for rx_type, targets in self.isomeric_branching_targets[nuc_name].items():
                    for target in targets:
                        if target in self.nuclide_dict and target not in isotope_set:
                            isotope_set.add(target)
                            to_check.append(target)
                        # Add siblings for target
                        for sibling in self._get_isomeric_siblings(target):
                            if sibling in self.nuclide_dict and sibling not in isotope_set:
                                isotope_set.add(sibling)
                                to_check.append(sibling)

        return isotope_set

    def reduce(self, initial_isotopes, level=None, keep_isomeric_siblings=True):
        """Reduce the size of the chain by following transmutation paths

        As an example, consider a simple chain with the following
        isotopes and transmutation paths::

            U235 (n,gamma) U236
                 (n,fission) (Xe135, I135, Cs135)
            I135 (beta decay) Xe135 (beta decay) Cs135
            Xe135 (n,gamma) Xe136

        Calling ``chain.reduce(["I135"])`` will produce a depletion
        chain that contains only isotopes that would originate from
        I135: I135, Xe135, Cs135, and Xe136. U235 and U236 will not
        be included, but multiple isotopes can be used to start
        the search.

        The ``level`` value controls the depth of the search.
        ``chain.reduce(["U235"], level=1)`` would return a chain
        with all isotopes except Xe136, since it is two transmutations
        removed from U235 in this case.

        While targets will not be included in the new chain, the
        total destruction rate and decay rate of included isotopes
        will be preserved.

        .. versionadded:: 0.15.3
            Isomeric branching data is now filtered and preserved in reduced
            chains. When isomeric targets are partially excluded the surviving
            targets are retained (and pruned ones recorded); the ratios
            themselves are renormalized to sum to 1.0 later -- at runtime for
            flag-only chains and at patch time for embedded ratios -- not here.

        Parameters
        ----------
        initial_isotopes : iterable of str
            Start the search based on the contents of these isotopes
        level : int, optional
            Depth of transmuation path to follow. Must be greater than
            or equal to zero. A value of zero returns a chain with
            ``initial_isotopes``. The default value of None implies
            that all isotopes that appear in the transmutation paths
            of the initial isotopes and their progeny should be
            explored
        keep_isomeric_siblings : bool, optional
            Whether to keep all isomeric state siblings together. Only takes
            effect for chains that carry isomeric-branching metadata; for a
            vanilla/official chain without it this flag is a no-op and
            ``reduce()`` matches the upstream (sibling-independent) result.

            - True (default): Always keep all isomeric siblings (ground +
              metastables) when any state is reachable. Required for correct
              isomeric branching calculations. May increase chain size by
              10-30%.

            - False: Original behavior. Isomeric states treated independently.
              May cause isomeric branching failures with partial exclusions.
              Use this only if you don't need isomeric branching or accept
              renormalization warnings.

        Returns
        -------
        Chain
            Depletion chain containing isotopes that would appear
            after following up to ``level`` reactions and decay paths.
            If the original chain contains isomeric branching data,
            the reduced chain will include the filtered isomeric branching
            (targets/LFS/Q and embedded ratios) for reactions where at least
            one target is retained.

        Notes
        -----
        **Isomeric Branching Handling:**

        Energy-dependent isomeric branching data is filtered based on which
        isotopes are included in the reduced chain:

        - **All targets retained**: branching data preserved unchanged
        - **All targets excluded**: isomeric branching entry dropped for that reaction
        - **Partial exclusion**: surviving targets kept, pruned ones recorded in
          ``reduce_pruned_targets``; the remaining ratios are renormalized to sum
          to 1.0 downstream (at runtime for flag-only chains, at patch time for
          embedded ratios), preserving mass conservation

        This behavior is consistent with fission yield handling, where products
        are filtered and yields are implicitly renormalized.

        """
        check_type("initial_isotopes", initial_isotopes, Iterable, str)
        if level is None:
            level = math.inf
        else:
            check_type("level", level, Integral)
            check_greater_than("level", level, 0, equality=True)

        # Validate keep_isomeric_siblings is a bool
        if not isinstance(keep_isomeric_siblings, bool):
            raise TypeError(
                f"keep_isomeric_siblings must be bool, got "
                f"{type(keep_isomeric_siblings).__name__}"
            )

        # Sibling expansion only matters for chains that actually carry
        # isomeric-branching metadata; gating on it keeps reduce() byte-identical
        # to upstream for vanilla/official chains (no spurious 10-30% growth).
        has_iso_metadata = self.isomeric_branching_targets is not None
        expand_siblings = keep_isomeric_siblings and has_iso_metadata

        # First pass: Follow reactions, including isomeric branching targets
        # when expansion is active
        all_isotopes = self._follow(set(initial_isotopes), level,
                                   include_isomeric_targets=expand_siblings)

        # Expand to include isomeric siblings if requested
        if expand_siblings:
            self._expand_with_isomeric_siblings(all_isotopes)

        # Avoid re-sorting for fission yields
        name_sort = sorted(all_isotopes)

        new_chain = type(self)()

        for idx, iso in enumerate(sorted(all_isotopes, key=openmc.data.zam)):
            previous = self[iso]
            new_nuclide = Nuclide(previous.name)
            new_nuclide.half_life = previous.half_life
            new_nuclide.decay_energy = previous.decay_energy
            new_nuclide.sources = previous.sources.copy()
            if hasattr(previous, '_fpy'):
                new_nuclide._fpy = previous._fpy

            for mode in previous.decay_modes:
                if mode.target in all_isotopes:
                    new_nuclide.add_decay_mode(*mode)
                else:
                    new_nuclide.add_decay_mode(mode.type, None, mode.branching_ratio)

            for rx in previous.reactions:
                if rx.target in all_isotopes:
                    new_nuclide.add_reaction(*rx)
                elif rx.type == "fission":
                    new_yields = new_nuclide.yield_data = (
                        previous.yield_data.restrict_products(name_sort))
                    if new_yields is not None:
                        new_nuclide.add_reaction(*rx)
                # Maintain total destruction rates but set no target
                else:
                    new_nuclide.add_reaction(rx.type, None, rx.Q, rx.branching_ratio)

            new_chain.add_nuclide(new_nuclide)

        # Doesn't appear that the ordering matters for the reactions,
        # just the contents
        new_chain.reactions = sorted(new_chain.reactions)

        # Filter isomeric branching targets for reduced chain
        if self.isomeric_branching_targets:
            new_targets = {}
            new_lfs = {}
            new_q = {}
            pruned = {}
            for parent, reactions in self.isomeric_branching_targets.items():
                if parent not in all_isotopes:
                    continue
                new_reactions = {}
                new_lfs_reactions = {}
                new_q_reactions = {}
                pruned_reactions = {}
                for rx_type, target_list in reactions.items():
                    retained_indices = [i for i, t in enumerate(target_list)
                                        if t in all_isotopes]
                    retained = [target_list[i] for i in retained_indices]
                    removed = [t for t in target_list if t not in all_isotopes]
                    if retained:
                        new_reactions[rx_type] = retained
                        # Keep LFS in sync
                        if (self.isomeric_branching_lfs
                                and parent in self.isomeric_branching_lfs
                                and rx_type in self.isomeric_branching_lfs[parent]):
                            lfs_list = self.isomeric_branching_lfs[parent][rx_type]
                            new_lfs_reactions[rx_type] = [lfs_list[i]
                                                          for i in retained_indices]
                        # Keep per-pathway Q in sync (parallel to targets/LFS)
                        if (self.isomeric_branching_q
                                and parent in self.isomeric_branching_q
                                and rx_type in self.isomeric_branching_q[parent]):
                            q_list = self.isomeric_branching_q[parent][rx_type]
                            new_q_reactions[rx_type] = [q_list[i]
                                                        for i in retained_indices]
                    if removed:
                        pruned_reactions[rx_type] = removed
                if new_reactions:
                    new_targets[parent] = new_reactions
                if new_lfs_reactions:
                    new_lfs[parent] = new_lfs_reactions
                if new_q_reactions:
                    new_q[parent] = new_q_reactions
                if pruned_reactions:
                    pruned[parent] = pruned_reactions
            new_chain.isomeric_branching_targets = new_targets or None
            new_chain.isomeric_branching_lfs = new_lfs or None
            new_chain.isomeric_branching_q = new_q or None
            new_chain.reduce_pruned_targets = pruned or None
        else:
            new_chain.isomeric_branching_targets = None
            new_chain.isomeric_branching_lfs = None
            new_chain.isomeric_branching_q = None
            new_chain.reduce_pruned_targets = None

        # Filter embedded branching ratios for reduced chain
        if self.isomeric_branching_embedded:
            new_embedded = {}
            for (parent, rx_type), data in self.isomeric_branching_embedded.items():
                if parent not in all_isotopes:
                    continue
                retained = [t for t in data['targets'] if t in all_isotopes]
                if not retained:
                    continue
                new_embedded[(parent, rx_type)] = {
                    'energies': data['energies'],
                    'targets': retained,
                    'branching_ratios': {t: data['branching_ratios'][t]
                                         for t in retained
                                         if t in data['branching_ratios']}
                }
            new_chain.isomeric_branching_embedded = new_embedded or None
        else:
            new_chain.isomeric_branching_embedded = None

        # Pre-compute isomeric families cache for the reduced chain
        new_chain._build_isomeric_families_cache()

        return new_chain

    def _follow(self, isotopes, level, include_isomeric_targets=True):
        """Return all isotopes present up to depth level

        Parameters
        ----------
        isotopes : set of str
            Initial isotopes to follow
        level : int
            Maximum depth to follow
        include_isomeric_targets : bool
            If True, include all isomeric branching targets.
            If False, only include the primary reaction target.

        Returns
        -------
        set of str
            All isotopes reachable within the specified level
        """
        found = isotopes.copy()
        remaining = set(self.nuclide_dict)
        if not found.issubset(remaining):
            raise IndexError(
                "The following isotopes were not found in the chain: "
                "{}".format(", ".join(found - remaining)))

        if level == 0:
            return found

        remaining -= found

        depth = 0
        next_iso = set()

        while depth < level and remaining:
            # Exhaust all isotopes at this level
            while isotopes:
                iso = isotopes.pop()
                found.add(iso)
                nuclide = self[iso]

                # Follow all transmutation paths for this nuclide
                for rxn in nuclide.reactions + nuclide.decay_modes:
                    if rxn.type == "fission":
                        continue

                    # Figure out if this reaction produces light nuclides
                    if rxn.type in REACTIONS:
                        secondaries = REACTIONS[rxn.type].secondaries
                    else:
                        secondaries = []

                    # Only include secondaries if they are present in original chain
                    secondaries = [x for x in secondaries if x in self]

                    for product in chain([rxn.target], secondaries):
                        if product is None:
                            continue
                        # Skip if we've already come across this isotope
                        elif (product in next_iso or product in found
                              or product in isotopes):
                            continue
                        next_iso.add(product)

                    # Follow isomeric branching targets (co-products at same depth)
                    if (include_isomeric_targets
                        and self.isomeric_branching_targets is not None
                        and iso in self.isomeric_branching_targets
                        and rxn.type in self.isomeric_branching_targets[iso]):

                        for iso_target in self.isomeric_branching_targets[iso][rxn.type]:
                            if iso_target not in self:
                                continue
                            if iso_target == rxn.target:
                                continue
                            if (iso_target in next_iso
                                or iso_target in found
                                or iso_target in isotopes):
                                continue
                            next_iso.add(iso_target)

                if nuclide.yield_data is not None:
                    for product in nuclide.yield_data.products:
                        if (product in next_iso
                                or product in found or product in isotopes):
                            continue
                        next_iso.add(product)

            if not next_iso:
                # No additional isotopes to process, nor to update the
                # current set of discovered isotopes
                return found

            # Prepare for next dig
            depth += 1
            isotopes |= next_iso
            remaining -= next_iso
            next_iso.clear()

        # Process isotope that would have started next depth
        found.update(isotopes)

        return found


# A global cache for Chain objects
_CHAIN_CACHE = {}


def _get_chain(
    chain_file: PathLike | Chain | None = None,
    fission_q: dict | None = None
) -> Chain:
    """Get a depletion chain from a file or the runtime configuration.

    Parameters
    ----------
    chain_file : PathLike or Chain, optional
        Path to depletion chain XML file, a Chain instance, or None to use
        the file specified in ``openmc.config['chain_file']``.
    fission_q : dict, optional
        Dictionary of nuclides and their fission Q values [eV]. If not given,
        values will be pulled from the ``chain_file``.

    Returns
    -------
    Chain
        Depletion chain instance.
    """
    # If chain_file is already a Chain, return it directly
    if isinstance(chain_file, Chain):
        return chain_file

    # Resolve chain_file based on config if None
    if chain_file is None:
        chain_file = openmc.config.get('chain_file')
        if 'chain_file' not in openmc.config:
            raise DataError(
                "No depletion chain specified and could not find depletion "
                "chain in openmc.config['chain_file']"
            )
    elif not isinstance(chain_file, PathLike):
        raise TypeError("chain_file must be path-like, a Chain, or None")

    # Determine the key for the cache, which consists of the absolute path, the
    # file modification time, the file size, and the fission Q values.
    chain_path = Path(chain_file).resolve()
    stat_result = chain_path.stat()
    fq_tuple = tuple(sorted(fission_q.items())) if fission_q else ()
    key = (chain_path, stat_result.st_mtime, stat_result.st_size, fq_tuple)

    # Check the global cache. If not cached, load the chain from XML and store
    global _CHAIN_CACHE
    if key not in _CHAIN_CACHE:
        _CHAIN_CACHE[key] = Chain.from_xml(chain_path, fission_q)
    return _CHAIN_CACHE[key]


def _invalidate_chain_cache(chain):
    """Invalidate the cache for a specific Chain (when it is modifed)."""
    chain._decay_matrix = None
    if hasattr(chain, '_xml_path'):
        # Remove all entries with the same path as self._xml_path
        for key in list(_CHAIN_CACHE.keys()):
            if str(key[0]) == chain._xml_path:
                del _CHAIN_CACHE[key]
