.. _io_depletion_chain:

============================
Depletion Chain -- chain.xml
============================

A depletion chain file has a ``<depletion_chain>`` root element with one or more
``<nuclide>`` child elements. The decay, reaction, and fission product data for
each nuclide appears as child elements of ``<nuclide>``.

---------------------
``<nuclide>`` Element
---------------------

The ``<nuclide>`` element contains information on the decay modes, reactions,
and fission product yields for a given nuclide in the depletion chain. This
element may have the following attributes:

  :name:
    Name of the nuclide

  :half_life:
    Half-life of the nuclide in [s]

  :decay_modes:
    Number of decay modes present

  :decay_energy:
    Decay energy released in [eV]

  :reactions:
    Number of reactions present

For each decay mode, a :ref:`io_chain_decay` appears as a child of
``<nuclide>``. For each reaction present, a :ref:`io_chain_reaction` appears as
a child of ``<nuclide>``. If the nuclide is fissionable, a :ref:`io_chain_nfy`
appears as well.

.. _io_chain_decay:

-------------------
``<decay>`` Element
-------------------

The ``<decay>`` element represents a single decay mode and has the following
attributes:

  :type:
    The type of the decay, e.g. 'ec/beta+'

  :target:
    The daughter nuclide produced from the decay

  :branching_ratio:
    The branching ratio for this decay mode

.. _io_chain_reaction:

--------------------
``<source>`` Element
--------------------

The ``<source>`` element represents photon and electron sources associated with
the decay of a nuclide and contains information to construct an
:class:`openmc.stats.Univariate` object that represents this emission as an
energy distribution. This element has the following attributes:

  :type:
    The type of :class:`openmc.stats.Univariate` source term.

  :particle:
    The type of particle emitted, e.g., 'photon' or 'electron'

  :parameters:
    The parameters of the source term, e.g., for a
    :class:`openmc.stats.Discrete` source, the energies (in [eV]) at which the
    particles are emitted and their relative intensities in [Bq/atom] (in other
    words, decay constants).

----------------------
``<reaction>`` Element
----------------------

The ``<reaction>`` element represents a single transmutation reaction. This
element has the following attributes:

  :type:
    The type of the reaction, e.g., '(n,gamma)'

  :Q:
    The Q value of the reaction in [eV]

  :target:
    The nuclide produced in the reaction (absent if the type is 'fission')

  :branching_ratio:
    The branching ratio for the reaction

.. _io_chain_isomeric_branching:

--------------------------------
``<isomeric_branching>`` Element
--------------------------------

The optional ``<isomeric_branching>`` element is a child of a ``<reaction>``
element and records that a single reaction populates more than one isomeric
state of the product nuclide (for example ``(n,gamma)`` producing both the
ground state and one or more metastable states). It is used together with a
GENDF cross-section library so that the split between states is computed from
multigroup cross sections and the local flux spectrum at run time; see
:ref:`gendf_depletion`. Exactly one ``<isomeric_branching>`` element is written
per (nuclide, reaction type). The element has the following attributes:

  :targets:
    Space-separated list of product nuclides, ground state first (e.g.
    ``Ag110 Ag110_m1``).

  :gendf_lfs:
    Space-separated list of final-state (LFS) indices, one per entry in
    ``targets``, used to select the matching GENDF ``MF=10`` production section
    for each state. The list length must equal the number of ``targets``.

  :Q:
    Optional space-separated list of Q values in [eV], one per target. This is
    retained only for round-trip fidelity; it is not used when forming the
    transmutation matrix. The alias ``q_values`` is also accepted on read.

Because the ``<isomeric_branching>`` element carries only *flags* (the targets
and their LFS indices) and not the branching ratios themselves, the ratios are
recomputed at run time from the GENDF cross sections weighted by the flux
spectrum. This keeps a single chain valid across different spectra.

When a branched reaction is represented by a single ``<reaction>`` element, the
scalar ``target`` and ``Q`` attributes are omitted from that ``<reaction>`` and
the per-pathway lists on the ``<isomeric_branching>`` child are used instead. A
reaction represented by several same-type ``<reaction>`` elements (one per
static pathway) instead keeps its scalar ``target``, ``Q``, and
``branching_ratio`` on each element and carries the ``<isomeric_branching>``
child on the first of them.

.. note::

   The ``keep_isomeric_siblings`` argument of
   :class:`~openmc.deplete.CoupledOperator` and
   :class:`~openmc.deplete.IndependentOperator` only takes effect when a chain
   contains this metadata. For chains without any ``<isomeric_branching>`` or
   ``<isomeric_yields>`` elements the flag is a no-op and chain reduction
   matches the upstream behavior.

.. _io_chain_isomeric_yields:

-----------------------------
``<isomeric_yields>`` Element
-----------------------------

The ``<isomeric_yields>`` element is a legacy child of a ``<reaction>`` element
that stores explicit, energy-dependent branching ratios embedded in the chain
(rather than computing them from GENDF at run time). It has a ``type`` attribute
of ``energy_dependent`` and the following sub-elements:

  :energies:
    Space-separated energies in [eV] at which branching ratios are tabulated.

  :targets:
    Space-separated product nuclides, one column of ``branching_ratios`` each.

  :branching_ratios:
    One line of ratios per target (in the same order as ``targets``); each line
    lists one ratio per entry in ``energies``.

.. _io_chain_nfy:

------------------------------------
``<neutron_fission_yields>`` Element
------------------------------------

The ``<neutron_fission_yields>`` element provides yields of fission products for
fissionable nuclides. Normally, it has the follow sub-elements:

  :energies:
    Energies in [eV] at which yields for products are tabulated

  :fission_yields:

    Fission product yields for a single energy point. This element itself has a
    number of attributes/sub-elements:

      :energy:
        Energy in [eV] at which yields are tabulated

      :products:
        Names of fission products

      :data:
        Independent yields for each fission product

In the event that a nuclide doesn't have any known fission product yields, it is
possible to have that nuclide borrow yields from another nuclide by indicating
the other nuclide in a single `parent` attribute. For example:

.. code-block:: xml

    <neutron_fission_yields parent="U235"/>
