import lxml.etree as ET

import openmc.checkvalue as cv
from ._xml import get_text


# Valid accumulator modes: where a tally's per-batch accumulator is homed
ACCUMULATOR_MODES = {'replicated', 'shared', 'rma'}

# Valid moments modes: which cross-batch moments a tally keeps per bin
MOMENTS_MODES = {'full', 'mean', 'none'}


class ExaTally:
    """ExaTallies accumulator and moments modes of a tally.

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

    Attributes
    ----------
    accumulator : {'replicated', 'shared', 'rma'} or None
        Where the per-batch accumulator is homed across MPI ranks
    moments : {'full', 'mean', 'none'} or None
        Which cross-batch moments are kept per bin

    """

    def __init__(self, accumulator=None, moments=None):
        self.accumulator = accumulator
        self.moments = moments

    def __eq__(self, other):
        if not isinstance(other, ExaTally):
            return NotImplemented
        return (self.accumulator == other.accumulator
                and self.moments == other.moments)

    def __repr__(self):
        string = 'ExaTally\n'
        string += '{: <16}=\t{}\n'.format('\tAccumulator', self._accumulator)
        string += '{: <16}=\t{}\n'.format('\tMoments', self._moments)
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

    def to_xml_element(self):
        """Return XML representation of the ExaTallies modes

        Returns
        -------
        element : lxml.etree._Element or None
            XML element containing the modes that are set, or None if no mode
            is set (nothing needs to be written)

        """
        if self.accumulator is None and self.moments is None:
            return None
        element = ET.Element("exatally")
        if self.accumulator is not None:
            subelement = ET.SubElement(element, "accumulator")
            subelement.text = self.accumulator
        if self.moments is not None:
            subelement = ET.SubElement(element, "moments")
            subelement.text = self.moments
        return element

    @classmethod
    def from_xml_element(cls, elem):
        """Generate ExaTallies modes from an XML element

        Parameters
        ----------
        elem : lxml.etree._Element
            ``<exatally>`` XML element

        Returns
        -------
        openmc.ExaTally
            ExaTallies modes generated from the XML element

        """
        accumulator = get_text(elem, "accumulator")
        moments = get_text(elem, "moments")
        return cls(
            accumulator=accumulator.strip() if accumulator is not None else None,
            moments=moments.strip() if moments is not None else None)
