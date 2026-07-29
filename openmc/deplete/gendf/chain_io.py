"""GENDF isomeric-branching chain XML I/O."""

import numpy as np
from collections import defaultdict
from warnings import warn

import lxml.etree as ET


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
