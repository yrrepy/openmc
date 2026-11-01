.. _usersguide_parallel:

===================
Running in Parallel
===================

If you are running a simulation on a computer with multiple cores, multiple
sockets, or multiple nodes (i.e., a cluster), you can benefit from the fact that
OpenMC is able to use all available hardware resources if configured
correctly. OpenMC is capable of using both distributed-memory (`MPI
<https://www.mpi-forum.org/>`_) and shared-memory (`OpenMP
<https://www.openmp.org/>`_) parallelism. If you are on a single-socket
workstation or a laptop, using shared-memory parallelism is likely
sufficient. On a multi-socket node, cluster, or supercomputer, chances are you
will need to use both distributed-memory (across nodes) and shared-memory
(within a single node) parallelism.

----------------------------------
Shared-Memory Parallelism (OpenMP)
----------------------------------

When using OpenMP, multiple threads will be launched and each is capable of
simulating a particle independently of all other threads. The primary benefit of
using OpenMP within a node is that it requires very little extra memory per
thread. OpenMP can be turned on or off at configure-time; by default it is
turned on. The only requirement is that the C++ compiler you use must support
the OpenMP 3.1 or higher standard. Most recent compilers do support the use of
OpenMP.

To specify the number of threads at run-time, you can use the ``threads``
argument to :func:`openmc.run`::

  openmc.run(threads=8)

If you're running :ref:`scripts_openmc` directly from the command line, you can
use the ``-s`` or ``--threads`` command-line argument. Alternatively, you can
use the :envvar:`OMP_NUM_THREADS` environment variable. If you do not specify
the number of threads, the OpenMP library will try to determine how many
hardware threads are available on your system and use that many threads.

In general, it is recommended to use as many OpenMP threads as you have hardware
threads on your system. Notably, on a system with Intel hyperthreading, the
hyperthreads should be used and can be expected to provide a 10--30% performance
improvement over not using hyperthreads.

------------------------------------
Distributed-Memory Parallelism (MPI)
------------------------------------

MPI defines a library specification for message-passing between processes. There
are two major implementations of MPI, `OpenMPI <https://www.open-mpi.org/>`_ and
`MPICH <https://www.mpich.org/>`_. Both implementations are known to work with
OpenMC; there is no obvious reason to prefer one over the other. Building OpenMC
with support for MPI requires that you have one of these implementations
installed on your system. For instructions on obtaining MPI, see
:ref:`prerequisites`. Once you have an MPI implementation installed, compile
OpenMC following :ref:`usersguide_compile_mpi`.

To run a simulation using MPI, :ref:`scripts_openmc` needs to be called using
the `mpiexec <https://www.mpich.org/static/docs/v3.1/www1/mpiexec.html>`_
wrapper. For example, to run OpenMC using 32 processes:

.. code-block:: sh

   mpiexec -n 32 openmc

The same thing can be achieved from the Python API by supplying the ``mpi_args``
argument to :func:`openmc.run`::

   openmc.run(mpi_args=['mpiexec', '-n', '32'])

Distributed memory calculations with the random ray solver for DAGMC geometries 
require MOAB version 5.2.0 or later.

.. _usersguide_exatallies:

----------------------------------------
Large Tallies on Many Ranks (ExaTallies)
----------------------------------------

A tally with a fine mesh filter and a fine energy filter can need tens of
gigabytes per copy. By default every MPI rank keeps its own copy, so on a node
with many ranks the tallies, not the particles, decide whether a run fits in
memory. ExaTallies adds two per-tally settings that change where a tally is
stored and how much of it is kept. They do not change how particles are
tracked or how scores are made, and the tally means are those of a default run
to within floating-point round-off.

* The **accumulator mode** selects where the per-batch accumulator is stored:
  ``replicated`` (one copy per rank, the default), ``shared`` (one copy per
  node, in MPI-3 shared memory) or ``rma`` (one copy in total, distributed
  across all ranks by filter-bin row, together with the moments).
* The **moments mode** selects which cross-batch moments are kept: ``full``
  (sum and sum of squares, the default), ``mean`` (sum only) or ``none`` (no
  moments during the run; the run total is collected only when a statepoint is
  written and at the end of the run).

Set the global default on :class:`openmc.Settings` and override it per tally
with :class:`openmc.ExaTally`::

   settings.exatally = openmc.ExaTally(accumulator='shared')

   flux = openmc.Tally(name='fine flux')
   flux.filters = [mesh_filter, energy_filter]
   flux.scores = ['flux']
   flux.exatally = openmc.ExaTally(moments='none')

   coarse = openmc.Tally(name='coarse flux')
   coarse.filters = [coarse_mesh_filter]
   coarse.scores = ['flux']
   coarse.exatally = openmc.ExaTally(moments='full')

In XML the same settings are the ``<exatally>`` element of :ref:`settings.xml
<settings_exatally>` and of each tally in :ref:`tallies.xml <io_tallies>`.
Modes can be mixed across the tallies of one run, as above: the large tally
carries no moments, and the small tally on the coarse mesh provides the
uncertainty.

**How much memory a tally needs.** One *plane* is one double per tally bin:
the number of filter bins times the number of score bins times 8 bytes. The
table gives the tally memory of a flux tally on a mesh of 200,000 voxels with
16,000 energy groups, for which one plane is 25.6 GB, on nodes that run 48 MPI
ranks each. The full memory model is in :ref:`methods_parallel_exatallies`.

.. list-table::
   :header-rows: 1
   :widths: 24 40 36

   * - Accumulator / moments
     - Tally memory
     - What is held
   * - ``replicated`` / ``full``
     - 1.28 TB on the node of rank 0, 1.23 TB on every other node
     - one accumulator plane per rank, plus two moment planes on rank 0
   * - ``shared`` / ``full``
     - 77 GB on the node of rank 0, 26 GB on every other node
     - one accumulator plane per node, plus two moment planes on rank 0
   * - ``shared`` / ``mean``
     - 51 GB on the node of rank 0, 26 GB on every other node
     - one accumulator plane per node, plus one moment plane on rank 0
   * - ``shared`` / ``none``
     - 26 GB per node, plus 26 GB on the node of rank 0 while a statepoint is
       written and at the end of the run
     - one accumulator plane per node, no moment plane during the run
   * - ``rma`` / ``full``
     - 102 GB in total, divided evenly over all ranks
     - two accumulator planes and two moment planes, each distributed
   * - ``rma`` / ``none``
     - 51 GB in total, divided evenly over all ranks, plus 26 GB in total
       while a statepoint is written and at the end of the run
     - two accumulator planes, each distributed, no moment plane during the
       run

**Choosing the accumulator mode.**

- Use ``shared`` whenever several MPI ranks run on each node and one plane
  fits in the memory of a node. It is the mode to try first: tally memory
  stops growing with the number of ranks per node, and the scoring rate does
  not drop, because the per-batch reduction shrinks to one plane per node. It
  needs a build with both MPI and OpenMP and the Monte Carlo solver, 
  and it cannot be combined with ``no_reduce`` or CMFD.
- Use ``rma`` when one plane does not fit on a node, or when the run spans
  many nodes. ``rma`` holds about four planes (two with moments ``none``) in
  total over all ranks, against one per node for ``shared``, so it saves
  memory over ``shared`` only beyond about four nodes (two with ``none``), and
  it pays for the saving with scoring rate: nearly every score goes to a bin
  owned by another rank and crosses the network. It has the requirements of
  ``shared`` and is not available in event-based mode. On runs with more than
  one rank, the results of an ``rma`` tally are available only in the
  statepoint file, not in ``tallies.out`` or through :mod:`openmc.lib`.
- Global tallies, including :math:`k_\text{eff}`, always stay replicated, and
  the tally that generates weight windows always keeps ``replicated`` and
  ``full``, whatever the global default.

**Choosing the moments mode.**

- Keep ``full`` when you need the standard deviation of the tally, tally
  triggers or higher moments.
- Use ``mean`` when the mean is enough. It drops one moment plane on rank 0
  and otherwise behaves like ``full``, including restart from a statepoint.
- Use ``none`` for the largest tallies whose result is consumed as a mean,
  such as a fine flux spectrum that is later collapsed for an activation
  calculation. The tally then causes no communication at all between batches.
  A run cannot restart from a statepoint that contains such a tally, and the
  mode needs tally reduction (no ``no_reduce``) and the Monte Carlo solver.
- With ``mean`` or ``none`` the uncertainty must come from a companion tally
  with ``full`` on a coarser mesh or energy grid, or from independent runs with
  different random number seeds (see the ``moments`` entry of
  :ref:`settings_exatally`).

**Running and reading the results.** ``shared`` lets you run many MPI ranks
per node, for example one per NUMA domain or one per core, without
multiplying tally memory; the shared plane is spread over the NUMA domains of
the node. ``rma`` needs MPI initialized with at least
``MPI_THREAD_SERIALIZED`` when more than one OpenMP thread is used; OpenMC
requests this itself, and a program that initializes MPI before OpenMC (for
example through ``mpi4py``) must do the same. Results are read as always with
:class:`openmc.StatePoint`. For a tally with moments ``mean`` or ``none``,
:attr:`openmc.Tally.std_dev` is NaN, :attr:`openmc.Tally.sum_sq` is None, and
``tallies.out`` prints the mean without an uncertainty. The statepoint records
the moments mode of each tally, and :class:`openmc.StatePoint` sets
:attr:`openmc.Tally.exatally` on the tallies whose mode is ``mean`` or
``none``.

----------------------
Maximizing Performance
----------------------

There are a number of things you can do to ensure that you obtain optimal
performance on a machine when running in parallel:

- **Use OpenMP within each NUMA node**. Some large server processors have so
  many cores that the last level cache is split to reduce memory latency. For
  example, the Intel Xeon Haswell-EP_ architecture uses a snoop mode called
  *cluster on die* where the L3 cache is split in half. Thus, in general, you
  should use one MPI process per socket (and OpenMP within each socket), but for
  these large processors, you will want to go one step further and use one
  process per NUMA node. The Xeon Phi Knights Landing architecture uses a
  similar concept called `sub NUMA clustering
  <https://colfaxresearch.com/knl-numa/>`_.
- **Use a sufficiently large number of particles per generation**. Between
  fission generations, a number of synchronization tasks take place. If the
  number of particles per generation is too low and you are using many
  processes/threads, the synchronization time may become non-negligible.
- **Use hardware threading if available**.
- **Use process binding**. When running with MPI, you should ensure that
  processes are bound_ to a specific hardware region. This can be set using the
  ``-bind-to`` (MPICH) or ``--bind-to`` (OpenMPI) option to ``mpiexec``.
- **Use the ExaTallies modes for large tallies**. A tally with millions of
  bins should use the ``shared`` or ``rma`` accumulator mode, and the ``mean``
  or ``none`` moments mode when its standard deviation is not needed, so that
  tally memory does not grow with the number of ranks per node. See
  :ref:`usersguide_exatallies`.
- **Turn off generation of tallies.out**. For large simulations with millions of
  tally bins or more, generating this ASCII file might consume considerable
  time. You can turn off generation of ``tallies.out`` via the
  :attr:`Settings.output` attribute::

     settings = openmc.Settings()
     settings.output = {'tallies': False}

.. _Haswell-EP: https://www.anandtech.com/show/8423/intel-xeon-e5-version-3-up-to-18-haswell-ep-cores-/4
.. _bound: https://github.com/pmodels/mpich/blob/main/doc/wiki/how_to/Using_the_Hydra_Process_Manager.md#process-core-binding
