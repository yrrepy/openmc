"""Build a PENDF-native depletion chain with isomeric pathway reactions.

The base chain (nuclides, half-lives, decay modes) is built from ENDF decay
data via :meth:`openmc.deplete.Chain.from_endf`. Transmutation reactions are
then taken directly from a preprocessed PENDF HDF5 library (schema in the
PENDF implementation plan Sec. 4.1): each MF=10 partial that carries a baked
``product`` name becomes a full reaction with branching ratio 1.0 to that
single product. The ground channel keeps the canonical reaction name
(``(n,gamma)``); metastable channels are product-qualified (``(n,gamma)_m1``,
``(n,gamma)_m2``, ...) so the reaction ``type`` matches the per-target rows a
PENDF MicroXS carries. No branching ratios are stored.
"""

from pathlib import Path

import h5py

import openmc.data
from .chain import Chain, REACTIONS

__all__ = ["chain_from_pendf"]

# GNDS metastable ordinal -> ENDF decay-file suffix (m1->'m', m2->'n', ...)
_META_SUFFIX = " mnopqrs"


def _decay_filename(name):
    """Return the decay-sublibrary filename for a GNDS nuclide name."""
    z, a, m = openmc.data.zam(name)
    suffix = '' if m == 0 else _META_SUFFIX[m]
    return f'{openmc.data.ATOMIC_SYMBOL[z]}{a:03d}{suffix}'


def _decay_closure(decay_dir, nuclides):
    """Return ``nuclides`` plus every decay daughter reachable from them.

    ``Chain.from_endf`` resolves each decay mode's daughter against the decay
    data it was given; restricting to a bare seed set leaves daughters (and
    thus whole elements) unrepresented and trips ``replace_missing``. Walking
    the decay closure keeps the base chain self-consistent.
    """
    closure = {}
    frontier = list(nuclides)
    while frontier:
        name = frontier.pop()
        if name in closure:
            continue
        path = decay_dir / _decay_filename(name)
        if not path.is_file():
            continue
        closure[name] = str(path)
        for mode in openmc.data.Decay(str(path)).modes:
            if mode.daughter and mode.daughter not in closure:
                frontier.append(mode.daughter)
    return closure


def chain_from_pendf(pendf_h5, decay_dir, nuclides=None):
    """Build a depletion chain with PENDF isomeric pathway reactions.

    Parameters
    ----------
    pendf_h5 : path-like
        Preprocessed PENDF HDF5 library (read only).
    decay_dir : path-like
        Directory of ENDF decay sub-library files (one per nuclide).
    nuclides : iterable of str, optional
        Restrict the chain to these GNDS nuclide names (and whatever decay
        modes reference). If ``None``, every decay file in ``decay_dir`` is
        used.

    Returns
    -------
    openmc.deplete.Chain
        Chain whose ``reactions`` carry product-qualified isomeric pathways.
        A ``coverage`` attribute (list of dicts with keys ``parent``,
        ``reaction``, ``product``, ``reason``) records pathways the physics
        implies but the data cannot supply.

    """
    decay_dir = Path(decay_dir)
    if nuclides is not None:
        decay_files = sorted(_decay_closure(decay_dir, nuclides).values())
    else:
        decay_files = [str(p) for p in sorted(decay_dir.iterdir()) if p.is_file()]

    # Base chain: decay structure only (no neutron files -> no reactions).
    chain = Chain.from_endf(decay_files, [], [], reactions=(), progress=False)
    chain_names = {nuc.name for nuc in chain.nuclides}

    # Reverse MT -> canonical chain reaction name.
    mt_to_name = {}
    for name, info in REACTIONS.items():
        for mt in info.mts:
            mt_to_name.setdefault(mt, name)

    coverage = []
    with h5py.File(pendf_h5, 'r') as h5:
        for nuclide in chain.nuclides:
            if nuclide.name not in h5:
                continue
            z, a, _ = openmc.data.zam(nuclide.name)
            nuc_group = h5[nuclide.name]
            for mt_key in nuc_group:
                if not mt_key.startswith('MT'):
                    continue
                mt = int(mt_key[2:])
                name = mt_to_name.get(mt)
                if name is None:
                    continue
                mt_group = nuc_group[mt_key]
                q_value = float(mt_group.attrs['QI'])

                # Ground product from DADZ (Sym{A}), drives coverage checks.
                delta_a, delta_z = openmc.data.DADZ[name]
                ground = f'{openmc.data.ATOMIC_SYMBOL[z + delta_z]}{a + delta_a}'

                lfs_keys = sorted(k for k in mt_group if k.startswith('LFS'))
                products_added = set()
                if lfs_keys:
                    # One reaction per MF=10 pathway with a baked product name.
                    for lfs_key in lfs_keys:
                        sub = mt_group[lfs_key]
                        if 'product' not in sub.attrs:
                            coverage.append(dict(
                                parent=nuclide.name, reaction=name, product=None,
                                reason=f'MF=10 {mt_key}/{lfs_key} has no mapped product'))
                            continue
                        product = sub.attrs['product']
                        if isinstance(product, bytes):
                            product = product.decode()
                        liso = openmc.data.zam(product)[2]
                        r_type = name if liso == 0 else f'{name}_m{liso}'
                        if product in chain_names:
                            nuclide.add_reaction(r_type, product, q_value, 1.0)
                            products_added.add(product)
                        else:
                            coverage.append(dict(
                                parent=nuclide.name, reaction=r_type, product=product,
                                reason='pathway target not in chain nuclide set'))
                else:
                    # No MF=10 -> single canonical reaction to the ground product.
                    if ground in chain_names:
                        nuclide.add_reaction(name, ground, q_value, 1.0)
                        products_added.add(ground)
                    else:
                        coverage.append(dict(
                            parent=nuclide.name, reaction=name, product=ground,
                            reason='ground product not in chain nuclide set'))

                # Metastable siblings decay data knows about but MF=10 lacks.
                for m in range(1, len(_META_SUFFIX)):
                    meta = f'{ground}_m{m}'
                    if meta in chain_names and meta not in products_added:
                        coverage.append(dict(
                            parent=nuclide.name, reaction=f'{name}_m{m}', product=meta,
                            reason='metastable product in decay data but no MF=10 pathway'))

    chain.coverage = coverage
    return chain
