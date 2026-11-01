from collections.abc import Iterable, Mapping

import lxml.etree as ET

import openmc
import openmc.checkvalue as cv
from ._xml import get_text


# Valid accumulator modes: where a tally's per-batch accumulator is homed
ACCUMULATOR_MODES = {'replicated', 'shared', 'rma'}

# Valid moments modes: which cross-batch moments a tally keeps per bin
MOMENTS_MODES = {'full', 'mean', 'none'}


class ExaTally:
    """ExaTallies accumulator and moments modes of a tally, and its projections.

    An instance assigned to :attr:`openmc.Settings.exatally` sets the global
    default; one assigned to :attr:`openmc.Tally.exatally` overrides that
    default for one tally. A mode left as None is not written, so it inherits
    the global default (for a tally) or the built-in default (for the
    settings).

    .. versionadded:: 0.16.1

    Parameters
    ----------
    accumulator : {'replicated', 'shared', 'rma'}, optional
        Where the per-batch accumulator is homed across MPI ranks.
        'replicated' (the built-in default) keeps a full private copy on every
        rank, 'shared' keeps one plane per shared-memory node, and 'rma'
        block-distributes the accumulator and the moments across all ranks.
    moments : {'full', 'mean', 'none'}, optional
        Which cross-batch moments are kept per bin. 'full' (the built-in
        default) keeps the sum and the sum of squares (plus the higher moments
        when requested), so the mean and its standard deviation are available.
        'mean' keeps the sum only: one moment plane, the mean but no standard
        deviation (:attr:`openmc.Tally.std_dev` is NaN), and no triggers or
        higher moments. 'none' keeps no moments during the run: the
        accumulator holds the run total, which is combined and scaled only for
        a statepoint and at the end of the run; it adds to the 'mean'
        restrictions that a run cannot restart from its statepoint, and that
        it requires tally reduction (no ``no_reduce``) and the Monte Carlo
        solver. See :ref:`settings_exatally`.
    projections : dict of str to list of openmc.Filter, optional
        Projections of the tally (per tally only; not allowed in
        :attr:`openmc.Settings.exatally`). Each maps a name (unique within
        the tally; no '/') to a filter list in the tally's filter order, in
        which each of the tally's filters is either kept (the tally's own
        filter object), collapsed (a coarser :class:`openmc.EnergyFilter`
        whose boundaries are boundaries of the tally's energy filter), or
        left out (summed over). An empty list sums over every filter. For
        each projection the run keeps the sum and the sum of squares of the
        projected batch realization, so the projected quantity gets an exact
        standard deviation that includes the covariance of the fine bins it
        sums. Not allowed with moments 'none'. The results are read as
        :attr:`openmc.Tally.projections`. See the ``exatally`` entry of
        :ref:`io_tallies`.

        .. versionadded:: 0.16.1

    Attributes
    ----------
    accumulator : {'replicated', 'shared', 'rma'} or None
        Where the per-batch accumulator is homed across MPI ranks
    moments : {'full', 'mean', 'none'} or None
        Which cross-batch moments are kept per bin
    projections : dict of str to list of openmc.Filter or None
        Projections of the tally, by name

    """

    def __init__(self, accumulator=None, moments=None, projections=None):
        self.accumulator = accumulator
        self.moments = moments
        self.projections = projections

    def _projection_key(self):
        """Comparable form of the projections: name -> [(id, filter), ...]."""
        if not self.projections:
            return {}
        return {name: [(f.id, f) for f in filters]
                for name, filters in self.projections.items()}

    def __eq__(self, other):
        if not isinstance(other, ExaTally):
            return NotImplemented
        return (self.accumulator == other.accumulator
                and self.moments == other.moments
                and self._projection_key() == other._projection_key())

    def __repr__(self):
        string = 'ExaTally\n'
        string += '{: <16}=\t{}\n'.format('\tAccumulator', self._accumulator)
        string += '{: <16}=\t{}\n'.format('\tMoments', self._moments)
        if self._projections:
            proj = {name: [f.id for f in filters]
                    for name, filters in self._projections.items()}
            string += '{: <16}=\t{}\n'.format('\tProjections', proj)
        return string

    @property
    def accumulator(self):
        return self._accumulator

    @accumulator.setter
    def accumulator(self, accumulator):
        if accumulator is not None:
            cv.check_value('exatally accumulator', accumulator,
                           ACCUMULATOR_MODES)
        self._accumulator = accumulator

    @property
    def moments(self):
        return self._moments

    @moments.setter
    def moments(self, moments):
        if moments is not None:
            cv.check_value('exatally moments', moments, MOMENTS_MODES)
        self._moments = moments

    @property
    def projections(self):
        return self._projections

    @projections.setter
    def projections(self, projections):
        if projections is None:
            self._projections = None
            return
        cv.check_type('exatally projections', projections, Mapping)
        checked = {}
        for name, filters in projections.items():
            cv.check_type('exatally projection name', name, str)
            if not name or name == '.' or '/' in name:
                raise ValueError(
                    f'Invalid ExaTally projection name {name!r}: a name '
                    "must be non-empty, not '.', and contain no '/'.")
            cv.check_type(f'filters of projection {name!r}', filters,
                          Iterable, openmc.Filter)
            checked[name] = list(filters)
        self._projections = checked

    def to_xml_element(self):
        """Return XML representation of the ExaTallies modes

        Returns
        -------
        element : lxml.etree._Element or None
            XML element containing the modes and projections that are set,
            or None if nothing is set (nothing needs to be written)

        """
        if (self.accumulator is None and self.moments is None
                and not self.projections):
            return None
        element = ET.Element("exatally")
        if self.accumulator is not None:
            subelement = ET.SubElement(element, "accumulator")
            subelement.text = self.accumulator
        if self.moments is not None:
            subelement = ET.SubElement(element, "moments")
            subelement.text = self.moments
        for name, filters in (self.projections or {}).items():
            subelement = ET.SubElement(element, "projection")
            subelement.set("name", name)
            subelement.set("filters", ' '.join(str(f.id) for f in filters))
        return element

    @classmethod
    def from_xml_element(cls, elem, filters=None):
        """Generate ExaTallies modes from an XML element

        Parameters
        ----------
        elem : lxml.etree._Element
            ``<exatally>`` XML element
        filters : dict of int to openmc.Filter, optional
            Filters by ID, to resolve the filters of ``<projection>``
            elements (required when the element has any)

        Returns
        -------
        openmc.ExaTally
            ExaTallies modes generated from the XML element

        """
        accumulator = get_text(elem, "accumulator")
        moments = get_text(elem, "moments")
        projections = None
        for proj in elem.findall("projection"):
            if filters is None:
                raise ValueError('<projection> elements need the filters by '
                                 'ID to be resolved.')
            if projections is None:
                projections = {}
            ids = proj.get("filters", "").split()
            projections[proj.get("name")] = [filters[int(i)] for i in ids]
        return cls(
            accumulator=accumulator.strip() if accumulator is not None else None,
            moments=moments.strip() if moments is not None else None,
            projections=projections)
