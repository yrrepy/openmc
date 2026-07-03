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

.. versionadded:: 0.15.4
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


def _reaction_products(h5, name, mt_to_name):
    """Yield the transmutation product names of ``name`` from the PENDF library.

    Mirrors the reaction extraction in :func:`chain_from_pendf`: every baked
    MF=10 ``product`` for a mapped MT, plus the DADZ ground product for a mapped
    MT that carries no MF=10 pathway. ``name`` absent from the library yields
    nothing.
    """
    if name not in h5:
        return
    z, a, _ = openmc.data.zam(name)
    nuc_group = h5[name]
    for mt_key in nuc_group:
        if not mt_key.startswith('MT'):
            continue
        r_name = mt_to_name.get(int(mt_key[2:]))
        if r_name is None:
            continue
        mt_group = nuc_group[mt_key]
        lfs_keys = [k for k in mt_group if k.startswith('LFS')]
        if lfs_keys:
            for lfs_key in lfs_keys:
                product = mt_group[lfs_key].attrs.get('product')
                if product is None:
                    continue
                if isinstance(product, bytes):
                    product = product.decode()
                yield product
        else:
            delta_a, delta_z = openmc.data.DADZ[r_name]
            # Exotic multi-particle MTs on low-Z targets can push the product
            # below Z=1; no such nuclide exists, so skip it.
            if (z + delta_z) in openmc.data.ATOMIC_SYMBOL:
                yield f'{openmc.data.ATOMIC_SYMBOL[z + delta_z]}{a + delta_a}'


def _chain_closure(decay_dir, nuclides, h5, mt_to_name):
    """Return the transmutation+decay closure of ``nuclides``.

    Maps each reachable nuclide name to its decay-sublibrary path. Restricting
    to a bare seed set silently loses the seed's own activation products:
    capture/transmutation products are *not* decay daughters of the seed, so a
    decay-only walk leaves them out of the chain and the reaction pathways fall
    into the coverage report. This walks BOTH the decay daughters and the PENDF
    transmutation products (:func:`_reaction_products`) to a fixed point, so a
    seeded target keeps its whole activation network. Only nuclides with a
    decay file join the closure (unchanged rule); products lacking one never
    enter ``chain_names`` and are surfaced as coverage by the caller. The
    nuclide set is finite, so the walk terminates.
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
            if mode.daughter:
                frontier.append(mode.daughter)
        frontier.extend(_reaction_products(h5, name, mt_to_name))
    return closure


def chain_from_pendf(pendf_h5, decay_dir, nuclides=None):
    """Build a depletion chain with PENDF isomeric pathway reactions.

    .. versionadded:: 0.15.4

    Parameters
    ----------
    pendf_h5 : path-like
        Preprocessed PENDF HDF5 library (read only).
    decay_dir : path-like
        Directory of ENDF decay sub-library files (one per nuclide).
    nuclides : iterable of str, optional
        Restrict the chain to these GNDS nuclide names and their full
        transmutation+decay closure: the walk follows both decay daughters and
        PENDF reaction products (MF=10 baked products and DADZ ground products)
        to a fixed point, so seeding only a target keeps that target's whole
        activation network instead of dropping its capture pathways to the
        coverage report. If ``None``, every decay file in ``decay_dir`` is used.

    Returns
    -------
    openmc.deplete.Chain
        Chain whose ``reactions`` carry product-qualified isomeric pathways.
        A ``coverage`` attribute (list of dicts with keys ``parent``,
        ``reaction``, ``product``, ``reason``) records pathways the physics
        implies but the data cannot supply.

    """
    decay_dir = Path(decay_dir)

    # Reverse MT -> canonical chain reaction name (built once, reused below).
    mt_to_name = {}
    for name, info in REACTIONS.items():
        for mt in info.mts:
            mt_to_name.setdefault(mt, name)

    coverage = []
    with h5py.File(pendf_h5, 'r') as h5:
        if nuclides is not None:
            decay_files = sorted(
                _chain_closure(decay_dir, nuclides, h5, mt_to_name).values())
        else:
            decay_files = [str(p) for p in sorted(decay_dir.iterdir())
                           if p.is_file()]

        # Base chain: decay structure only (no neutron files -> no reactions).
        chain = Chain.from_endf(decay_files, [], [], reactions=(), progress=False)
        chain_names = {nuc.name for nuc in chain.nuclides}

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

                # Ground product from DADZ (Sym{A}), drives coverage checks.
                delta_a, delta_z = openmc.data.DADZ[name]
                if (z + delta_z) not in openmc.data.ATOMIC_SYMBOL:
                    # Exotic multi-particle MT drove the product below Z=1.
                    coverage.append(dict(
                        parent=nuclide.name, reaction=name, product=None,
                        reason='product Z out of range'))
                    continue
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
                            if 'QI' not in sub.attrs:
                                coverage.append(dict(
                                    parent=nuclide.name, reaction=r_type, product=product,
                                    reason='LFS QI absent; Q defaulted to 0.0'))
                            lfs_q = float(sub.attrs.get('QI', 0.0))
                            nuclide.add_reaction(r_type, product, lfs_q, 1.0)
                            products_added.add(product)
                        else:
                            coverage.append(dict(
                                parent=nuclide.name, reaction=r_type, product=product,
                                reason='pathway target not in chain nuclide set'))
                else:
                    # No MF=10 -> single canonical reaction to the ground
                    # product. The MT-group QI drives this ground-only branch;
                    # read it here (only branch that uses it) and default to 0.0
                    # with a coverage note if the file omits it.
                    if 'QI' not in mt_group.attrs:
                        coverage.append(dict(
                            parent=nuclide.name, reaction=name, product=ground,
                            reason='MT-group QI absent; Q defaulted to 0.0'))
                    q_value = float(mt_group.attrs.get('QI', 0.0))
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

    # Reactions were attached to the Nuclide objects directly, bypassing
    # Chain.add_nuclide, so the top-level chain.reactions (built by from_endf
    # when no reactions existed yet) is still empty and IndependentOperator
    # would read nothing from it. Rebuild it exactly as Chain.from_xml would:
    # first-appearance order across nuclides in chain order, then within each
    # nuclide's reaction list (mirrors Chain.add_nuclide, chain.py).
    chain.reactions = []
    for nuclide in chain.nuclides:
        for rx in nuclide.reactions:
            if rx.type not in chain.reactions:
                chain.reactions.append(rx.type)
    return chain
