"""DICOM file comparison utilities.

This module provides tools for comparing two DICOM files at the attribute level,
identifying which attributes are shared, exclusive to one file, and whether
shared attributes have matching or conflicting values.

Example:
    >>> dcm1 = pydicom.dcmread("path/to/first.dcm")
    >>> dcm2 = pydicom.dcmread("path/to/second.dcm")
    >>> comparer = DicomComparer(dcm1, dcm2)
    >>> comparison = comparer.compare()
    >>> comparison.summary()
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union, overload
import hashlib

import pydicom


def _compute_value_hash(value: Any) -> str:
    """Computes a hash string for any DICOM value, including non-hashable types.

    Handles numpy arrays (pixel data), sequences, and other complex types
    by converting them to bytes and computing an MD5 hash.

    Args:
        value: The DICOM element value to hash.

    Returns:
        A hex string representing the hash of the value.
    """
    try:
        # Try to check if it's a numpy array (pixel data)
        if hasattr(value, 'tobytes'):
            # numpy array - hash the raw bytes
            return hashlib.md5(value.tobytes()).hexdigest()

        # Try to check if it's a pydicom Sequence
        if isinstance(value, pydicom.Sequence):
            # Hash each item in the sequence recursively
            seq_hashes: list[str] = [_compute_value_hash(item) for item in value]
            return hashlib.md5("".join(seq_hashes).encode()).hexdigest()

        # Try to check if it's a pydicom Dataset (nested)
        if isinstance(value, pydicom.Dataset):
            # Hash all elements in the dataset
            elem_hashes: list[str] = []
            for elem in value:
                elem_hashes.append(f"{elem.tag}:{_compute_value_hash(elem.value)}")
            return hashlib.md5("".join(elem_hashes).encode()).hexdigest()

        # For bytes, hash directly
        if isinstance(value, bytes):
            return hashlib.md5(value).hexdigest()

        # For other iterables (but not strings), try to hash contents
        if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
            try:
                items: list[str] = [_compute_value_hash(item) for item in value]
                return hashlib.md5("".join(items).encode()).hexdigest()
            except (TypeError, RecursionError):
                pass

        # For simple hashable types, use their string representation
        return hashlib.md5(str(value).encode()).hexdigest()

    except Exception:
        # Fallback: hash the string representation
        return hashlib.md5(str(value).encode()).hexdigest()


# --- Data Structures ---

@dataclass(frozen=True, eq=True)
class FileAttribute:
    """Represents a single DICOM attribute (tag + name).

    An immutable identifier for a DICOM attribute, used as a key for
    comparing attributes across files.

    Attributes:
        key: The DICOM tag identifying this attribute.
        name: Human-readable name of the attribute.
    """

    key: pydicom.tag.BaseTag  # type: ignore
    name: str

    def _str(self) -> str:
        """Returns a formatted string of the tag and name."""
        return f"{self.key} {self.name}"

    def __repr__(self) -> str:
        return f"FileAttribute({self._str()})"

    @property
    def _hash_str(self) -> str:
        """Returns a unique string for hashing purposes."""
        return f"attr:{str(self.key)}:{str(self.name)}"

    def __hash__(self) -> int:
        return hash(self._hash_str)


@dataclass(frozen=True, eq=True)
class AttributeValue:
    """Represents a DICOM attribute paired with its value.

    Links a FileAttribute to its actual value in a specific DICOM file,
    enabling value-level comparisons between files. Comparison is based on
    the actual value content hash, not just the string representation.

    Attributes:
        attr: The FileAttribute this value belongs to.
        repval: String representation of the value (for display).
        value: The actual value stored in the DICOM element.
        value_hash: MD5 hash of the actual value content for comparison.
    """

    attr: FileAttribute
    repval: str
    value: Any = field(compare=False)  # Exclude from auto-generated __eq__
    value_hash: str = field(default="")  # Hash of actual value content

    def __repr__(self) -> str:
        return f"AttributeValue([{self.attr._str()}]: {self.repval})"

    @property
    def _hash_str(self) -> str:
        """Returns a unique string for hashing purposes based on actual value content."""
        return f"val:{self.attr._hash_str}:{self.value_hash}"

    def __hash__(self) -> int:
        return hash(self._hash_str)

    def __eq__(self, other: object) -> bool:
        """Compares AttributeValues based on attribute and value hash."""
        if not isinstance(other, AttributeValue):
            return NotImplemented
        return self.attr == other.attr and self.value_hash == other.value_hash


@dataclass(frozen=True, eq=True)
class AttrValComparison:
    """Represents a comparison of values for a single attribute across two files.

    Used to track whether an attribute's value matches or conflicts between
    two DICOM files being compared.

    Attributes:
        attr: The FileAttribute being compared.
        value1: The AttributeValue from the first DICOM file.
        value2: The AttributeValue from the second DICOM file.
    """

    attr: FileAttribute
    value1: AttributeValue
    value2: AttributeValue

    def __repr__(self) -> str:
        return f"AttrValComparison([{self.attr._str()}]: (1: {self.value1.repval}, 2: {self.value2.repval}))"

    @property
    def _hash_str(self) -> str:
        """Returns a unique string for hashing purposes."""
        return f"comp:{self.value1._hash_str}:{self.value2._hash_str}"

    def __hash__(self) -> int:
        return hash(self._hash_str)


# --- Comparison Result Containers ---

@dataclass
class ComparisonDetails:
    """Stores the results of comparing intersecting attribute values.

    Separates attributes into those with matching values and those with
    conflicting values between the two files.

    Attributes:
        matches: Comparisons where both files have the same value.
        conflicts: Comparisons where the files have different values.
    """

    matches: list[AttrValComparison] = field(default_factory=list)
    conflicts: list[AttrValComparison] = field(default_factory=list)

    @property
    def n_matches(self) -> int:
        """Returns the count of matching attribute values."""
        return len(self.matches)

    @property
    def n_conflicts(self) -> int:
        """Returns the count of conflicting attribute values."""
        return len(self.conflicts)

    def __repr__(self) -> str:
        return f"ComparisonDetails({self.n_matches} matches, {self.n_conflicts} conflicts)"


@dataclass
class IntersectionData:
    """Stores data about attributes present in both DICOM files.

    Attributes:
        attributes: Set of FileAttributes that exist in both files.
        comparison: Detailed comparison results for shared attributes.
    """

    attributes: set[FileAttribute] = field(default_factory=set)
    comparison: ComparisonDetails = field(default_factory=ComparisonDetails)

    @property
    def n(self) -> int:
        """Returns the count of intersecting attributes."""
        return len(self.attributes)

    def __repr__(self) -> str:
        n_matches: int = self.comparison.n_matches
        n_conflicts: int = self.comparison.n_conflicts

        comp_str: str = ""
        if n_matches + n_conflicts > 0:
            comp_str = f", matches: {n_matches}/{self.n} ({n_matches / self.n:.1%})"

        return f"IntersectionData({self.n} attributes{comp_str})"


@dataclass
class ExclusiveData:
    """Stores data about attributes exclusive to one DICOM file.

    Attributes:
        attributes: Set of FileAttributes unique to this file.
        values: Set of AttributeValues for the exclusive attributes.
    """

    attributes: set[FileAttribute] = field(default_factory=set)
    values: set[AttributeValue] = field(default_factory=set)

    @property
    def n(self) -> int:
        """Returns the count of exclusive attributes."""
        return len(self.attributes)

    def __repr__(self) -> str:
        return f"ExclusiveData({self.n} attributes)"


@dataclass
class FileComparison:
    """Complete comparison result between two DICOM files.

    Contains all data about attribute overlap, exclusive attributes, and
    value comparisons between the two files.

    Attributes:
        intersection: Data about attributes present in both files.
        exclusive_to_1: Data about attributes only in the first file.
        exclusive_to_2: Data about attributes only in the second file.
    """

    intersection: IntersectionData = field(default_factory=IntersectionData)
    exclusive_to_1: ExclusiveData = field(default_factory=ExclusiveData)
    exclusive_to_2: ExclusiveData = field(default_factory=ExclusiveData)

    @staticmethod
    def convert_vals_to_dict(
        vals: set[AttributeValue],
    ) -> dict[FileAttribute, AttributeValue]:
        """Converts a set of AttributeValues to a dict keyed by FileAttribute.

        Args:
            vals: Set of AttributeValue objects to convert.

        Returns:
            Dictionary mapping each FileAttribute to its AttributeValue.
        """
        return {v.attr: v for v in list(vals)}

    def register_attributes(
        self, attrs1: set[FileAttribute], attrs2: set[FileAttribute]
    ) -> None:
        """Compares and registers attribute sets from both DICOM files.

        Determines which attributes are shared between files and which are
        exclusive to each file.

        Args:
            attrs1: Set of FileAttributes from the first DICOM file.
            attrs2: Set of FileAttributes from the second DICOM file.

        Raises:
            ValueError: If either attribute set is None.
        """
        if attrs1 is None:
            raise ValueError("No attributes in DICOM #1!")
        elif attrs2 is None:
            raise ValueError("No attributes in DICOM #2!")

        inter_attrs, excl_attrs1, excl_attrs2 = compare_sets(
            attrs1, attrs2, verbose=False, output=True
        )

        self.intersection.attributes = inter_attrs
        self.exclusive_to_1.attributes = excl_attrs1
        self.exclusive_to_2.attributes = excl_attrs2

    def register_values(
        self, vals1: set[AttributeValue], vals2: set[AttributeValue]
    ) -> None:
        """Registers and categorizes attribute values from both DICOM files.

        Separates values into exclusive sets and triggers intersection evaluation
        for shared attributes.

        Args:
            vals1: Set of AttributeValues from the first DICOM file.
            vals2: Set of AttributeValues from the second DICOM file.
        """
        val_dict1: dict[FileAttribute, AttributeValue] = self.convert_vals_to_dict(vals1)
        val_dict2: dict[FileAttribute, AttributeValue] = self.convert_vals_to_dict(vals2)

        excl_vals1: set[AttributeValue] = set(
            [val_dict1[a] for a in self.exclusive_to_1.attributes]
        )
        excl_vals2: set[AttributeValue] = set(
            [val_dict2[a] for a in self.exclusive_to_2.attributes]
        )

        self.exclusive_to_1.values = excl_vals1
        self.exclusive_to_2.values = excl_vals2

        inter_vals1: set[AttributeValue] = vals1 - excl_vals1
        inter_vals2: set[AttributeValue] = vals2 - excl_vals2

        self.evaluate_intersection(inter_vals1, inter_vals2)

    def evaluate_intersection(
        self, inter_vals1: set[AttributeValue], inter_vals2: set[AttributeValue]
    ) -> None:
        """Evaluates intersecting attributes for matches and conflicts.

        Compares values for attributes present in both files and categorizes
        them as matching (same value) or conflicting (different values).

        Args:
            inter_vals1: AttributeValues for shared attributes from file 1.
            inter_vals2: AttributeValues for shared attributes from file 2.
        """
        match_vals, conflict_vals1, conflict_vals2 = compare_sets(
            inter_vals1, inter_vals2, verbose=False, output=True
        )

        for val in list(match_vals):
            av_comp: AttrValComparison = AttrValComparison(val.attr, val, val)
            self.intersection.comparison.matches.append(av_comp)

        cval_dict1: dict[FileAttribute, AttributeValue] = self.convert_vals_to_dict(conflict_vals1)
        cval_dict2: dict[FileAttribute, AttributeValue] = self.convert_vals_to_dict(conflict_vals2)

        conflict_attrs: set[FileAttribute] = set(
            list(cval_dict1.keys()) + list(cval_dict2.keys())
        )

        for attr in list(conflict_attrs):
            cval1: AttributeValue = cval_dict1[attr]
            cval2: AttributeValue = cval_dict2[attr]
            av_comp: AttrValComparison = AttrValComparison(attr, cval1, cval2)
            self.intersection.comparison.conflicts.append(av_comp)

    def __repr__(self) -> str:
        return f"FileComparison(intersection: {self.intersection.n}, 1: {self.exclusive_to_1.n}, 2: {self.exclusive_to_2.n})"

    def _make_bar(self, value: int, total: int, width: int = 30, fill: str = "█", empty: str = "░") -> str:
        """Creates an ASCII progress bar.

        Args:
            value: Current value to represent.
            total: Maximum value (100%).
            width: Character width of the bar.
            fill: Character for filled portion.
            empty: Character for empty portion.

        Returns:
            ASCII bar string.
        """
        if total == 0:
            return empty * width
        filled: int = int((value / total) * width)
        return fill * filled + empty * (width - filled)

    def _make_comparison_bar(self, excl1: int, shared: int, excl2: int, width: int = 40) -> list[str]:
        """Creates a visual comparison bar showing overlap between two files.

        Args:
            excl1: Count exclusive to file 1.
            shared: Count shared between files.
            excl2: Count exclusive to file 2.
            width: Character width of the bar.

        Returns:
            List of strings forming the visual comparison.
        """
        total: int = excl1 + shared + excl2
        if total == 0:
            return ["No attributes to compare"]

        w1: int = max(1, int((excl1 / total) * width)) if excl1 > 0 else 0
        ws: int = max(1, int((shared / total) * width)) if shared > 0 else 0
        w2: int = max(1, int((excl2 / total) * width)) if excl2 > 0 else 0

        # Adjust to exactly match width
        diff: int = width - (w1 + ws + w2)
        if diff != 0:
            if ws > 0:
                ws += diff
            elif w1 > 0:
                w1 += diff
            else:
                w2 += diff

        bar: str = "─" * w1 + "═" * ws + "─" * w2

        # Build indicator line
        indicators: str = ""
        if w1 > 0:
            indicators += "1" + " " * (w1 - 1)
        if ws > 0:
            mid: int = ws // 2
            indicators += " " * mid + "∩" + " " * (ws - mid - 1)
        if w2 > 0:
            indicators += " " * (w2 - 1) + "2"

        return [
            f"    ┌{'─' * width}┐",
            f"    │{bar}│",
            f"    └{'─' * width}┘",
            f"     {indicators}",
        ]

    def summary(self) -> None:
        """Prints a formatted visual summary of the comparison results."""
        n_inter: int = self.intersection.n
        n_excl1: int = self.exclusive_to_1.n
        n_excl2: int = self.exclusive_to_2.n
        n1: int = n_inter + n_excl1
        n2: int = n_inter + n_excl2
        n_total: int = n_excl1 + n_inter + n_excl2

        lines: list[str] = [
            "",
            "╔══════════════════════════════════════════════════════════════╗",
            "║                   DICOM COMPARISON SUMMARY                   ║",
            "╚══════════════════════════════════════════════════════════════╝",
            "",
        ]

        # Attribute overview table
        lines += [
            "┌─────────────────────────────────────────────────────────────┐",
            "│  ATTRIBUTE OVERVIEW                                         │",
            "├─────────────────────────────────────────────────────────────┤",
            f"│  DICOM #1 total attributes:  {n1:>6,d}                         │",
            f"│  DICOM #2 total attributes:  {n2:>6,d}                         │",
            "└─────────────────────────────────────────────────────────────┘",
            "",
        ]

        # Visual overlap representation
        lines += [
            "  ATTRIBUTE OVERLAP VISUALIZATION",
            "  ─────────────────────────────────",
            "",
            "    DICOM #1 only ─────┬───── Shared ─────┬───── DICOM #2 only",
        ]
        lines += self._make_comparison_bar(n_excl1, n_inter, n_excl2)
        lines += [""]

        # Overlap statistics table
        pct1_inter: float = (n_inter / n1 * 100) if n1 > 0 else 0
        pct2_inter: float = (n_inter / n2 * 100) if n2 > 0 else 0
        pct1_excl: float = (n_excl1 / n1 * 100) if n1 > 0 else 0
        pct2_excl: float = (n_excl2 / n2 * 100) if n2 > 0 else 0

        lines += [
            "┌───────────────────┬──────────┬─────────────┬─────────────────────────────────┐",
            "│ Category          │   Count  │  % of File  │ Distribution                    │",
            "├───────────────────┼──────────┼─────────────┼─────────────────────────────────┤",
            f"│ Shared            │  {n_inter:>6,d}  │ {pct1_inter:>5.1f}%/{pct2_inter:<5.1f}%│ {self._make_bar(n_inter, n_total, 30, '█', '░')} │",
            f"│ DICOM #1 only     │  {n_excl1:>6,d}  │ {pct1_excl:>5.1f}%      │ {self._make_bar(n_excl1, n_total, 30, '▓', '░')} │",
            f"│ DICOM #2 only     │  {n_excl2:>6,d}  │ {pct2_excl:>5.1f}%      │ {self._make_bar(n_excl2, n_total, 30, '▒', '░')} │",
            "└───────────────────┴──────────┴─────────────┴─────────────────────────────────┘",
            "",
        ]

        # Value comparison section (only if there are shared attributes)
        if n_inter > 0:
            n_matches: int = self.intersection.comparison.n_matches
            n_conflicts: int = self.intersection.comparison.n_conflicts
            pct_match: float = (n_matches / n_inter * 100) if n_inter > 0 else 0
            pct_conflict: float = (n_conflicts / n_inter * 100) if n_inter > 0 else 0

            lines += [
                "┌─────────────────────────────────────────────────────────────┐",
                "│  VALUE COMPARISON (Shared Attributes)                       │",
                "├─────────────────────────────────────────────────────────────┤",
            ]

            # Match/conflict bar
            match_bar: str = self._make_bar(n_matches, n_inter, 40, "✓", " ")
            conflict_bar: str = self._make_bar(n_conflicts, n_inter, 40, "✗", " ")

            lines += [
                f"│                                                             │",
                f"│  Matching values:    {n_matches:>6,d} ({pct_match:>5.1f}%)                      │",
                f"│  [{match_bar}]   │",
                f"│                                                             │",
                f"│  Conflicting values: {n_conflicts:>6,d} ({pct_conflict:>5.1f}%)                      │",
                f"│  [{conflict_bar}]   │",
                f"│                                                             │",
                "└─────────────────────────────────────────────────────────────┘",
                "",
            ]

            # Summary verdict
            if n_conflicts == 0:
                verdict: str = "✓ All shared attributes have matching values!"
                verdict_style: str = "SUCCESS"
            elif pct_conflict < 10:
                verdict = f"⚠ Minor differences: {n_conflicts} conflicting attribute(s)"
                verdict_style = "WARNING"
            else:
                verdict = f"✗ Significant differences: {n_conflicts} conflicting attribute(s)"
                verdict_style = "ALERT"

            lines += [
                "┌─────────────────────────────────────────────────────────────┐",
                f"│  {verdict_style:^59}  │",
                f"│  {verdict:<59}  │",
                "└─────────────────────────────────────────────────────────────┘",
            ]

        lines += [""]
        print("\n".join(lines))


# --- DICOM Wrapper ---

@dataclass
class Dicom:
    """Wrapper for a pydicom FileDataset that extracts attributes and values.

    Parses the DICOM file on initialization to extract all attributes and
    their values into hashable, comparable structures.

    Attributes:
        data: The underlying pydicom FileDataset.
        n: Count of attributes in the file.
        attrs: Set of FileAttribute objects extracted from the file.
        vals: Set of AttributeValue objects extracted from the file.
    """

    data: pydicom.FileDataset
    n: int = 0
    attrs: set[FileAttribute] = field(default_factory=set)
    vals: set[AttributeValue] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Extracts attributes and values from the DICOM data."""
        elements: list[pydicom.dataelem.DataElement] = list(self.data.values())  # type: ignore

        for e in elements:
            tag: pydicom.tag.BaseTag = e.tag  # type: ignore
            try:
                name: str = e.name
            except AttributeError:
                name = "NAME MISSING"

            repval: str = str(e.repval)
            val: Any = e.value
            val_hash: str = _compute_value_hash(val)

            attr: FileAttribute = FileAttribute(tag, name)
            attr_val: AttributeValue = AttributeValue(attr, repval, val, val_hash)

            self.attrs.add(attr)
            self.vals.add(attr_val)

        self.n = len(self.attrs)

    def __repr__(self) -> str:
        return f"Dicom({self.n:,d} key{'' if self.n == 1 else 's'})"


# --- Main Comparer ---

class DicomComparer:
    """Orchestrates comparison between two DICOM files.

    Provides a simple interface for comparing two DICOM files and generating
    a detailed FileComparison result.

    Example:
        >>> dcm1 = pydicom.dcmread("path/to/first.dcm")
        >>> dcm2 = pydicom.dcmread("path/to/second.dcm")
        >>> comparer = DicomComparer(dcm1, dcm2)
        >>> comparison = comparer.compare()
        >>> comparison.summary()
        >>> for c in comparison.intersection.comparison.conflicts:
        ...     print(c)

    Attributes:
        label1: Label for the first DICOM file.
        label2: Label for the second DICOM file.
        dcm1: Parsed Dicom wrapper for the first file.
        dcm2: Parsed Dicom wrapper for the second file.
    """

    def __init__(
        self,
        dcm1: Optional[pydicom.FileDataset] = None,
        dcm2: Optional[pydicom.FileDataset] = None,
        dcms: Optional[dict[str, pydicom.FileDataset]] = None,
    ) -> None:
        """Initializes the comparer with two DICOM files.

        Args:
            dcm1: First DICOM file (used with dcm2).
            dcm2: Second DICOM file (used with dcm1).
            dcms: Alternative input as a dict of {label: FileDataset} with exactly 2 entries.

        Raises:
            ValueError: If inputs are invalid or insufficient.
        """
        _dcms: dict[str, pydicom.FileDataset] = self._handle_inputs(dcm1, dcm2, dcms)
        (label1, dcm1), (label2, dcm2) = list(_dcms.items())

        self.label1: str = label1
        self.dcm1: Dicom = Dicom(dcm1)

        self.label2: str = label2
        self.dcm2: Dicom = Dicom(dcm2)

    def _handle_inputs(
        self,
        dcm1: Optional[pydicom.FileDataset] = None,
        dcm2: Optional[pydicom.FileDataset] = None,
        dcms: Optional[dict[str, pydicom.FileDataset]] = None,
    ) -> dict[str, pydicom.FileDataset]:
        """Validates and normalizes input arguments.

        Args:
            dcm1: First DICOM file.
            dcm2: Second DICOM file.
            dcms: Dictionary of labeled DICOM files.

        Returns:
            Normalized dictionary with exactly two label:FileDataset entries.

        Raises:
            ValueError: If inputs are invalid or insufficient.
        """
        if (dcms is None) and ((dcm1 is None) or (dcm2 is None)):
            raise ValueError(
                "Please specify a 'dcms' dictionary with two label:DICOM entries, "
                "or provide 'dcm1' and 'dcm2' DICOMs separately!"
            )

        if (dcms is not None) and (len(dcms) != 2):
            raise ValueError(
                "'dcms' dictionary must contain exactly two label:DICOM entries!"
            )

        if dcms is not None:
            return dcms

        if (dcm1 is None) or (dcm2 is None):
            raise ValueError("'dcm1' and 'dcm2' must both be specified")

        return {
            "DICOM #1": dcm1,
            "DICOM #2": dcm2,
        }

    def compare(self) -> FileComparison:
        """Performs the comparison between the two DICOM files.

        Returns:
            FileComparison object containing all comparison results.
        """
        comparison: FileComparison = FileComparison()

        comparison.register_attributes(self.dcm1.attrs, self.dcm2.attrs)
        comparison.register_values(self.dcm1.vals, self.dcm2.vals)

        return comparison


# --- Utility Functions ---

@overload
def compare_sets(
    set1: set, set2: set, verbose: bool = False, *, output: Literal[True]
) -> tuple[set, set, set]: ...


@overload
def compare_sets(
    set1: set, set2: set, verbose: bool = False, output: Literal[False] = False
) -> None: ...


def compare_sets(
    set1: set, set2: set, verbose: bool = False, output: bool = True
) -> Union[None, tuple[set, set, set]]:
    """Compares two sets and returns their intersection and exclusive elements.

    Args:
        set1: First set to compare.
        set2: Second set to compare.
        verbose: If True, prints comparison statistics.
        output: If True, returns the comparison results.

    Returns:
        If output is True: Tuple of (intersection, exclusive_to_set1, exclusive_to_set2).
        If output is False: None.
    """
    intersection: set = set1.intersection(set2)
    exclusive1: set = set1.difference(set2)
    exclusive2: set = set2.difference(set1)

    if verbose:
        n_set1: int = len(set1)
        n_set2: int = len(set2)
        n_inter: int = len(intersection)
        n_excl1: int = len(exclusive1)
        n_excl2: int = len(exclusive2)

        print(f"N in Set 1: {n_set1}")
        print(f"N in Set 2: {n_set2}")
        print(f"N present in both: {n_inter} ({n_inter/n_set1:.1%}, {n_inter/n_set2:.1%})")
        print(f"N exclusive to Set 1: {n_excl1} ({n_excl1/n_set1:.1%})")
        print(f"N exclusive to Set 2: {n_excl2} ({n_excl2/n_set2:.1%})")

    if output:
        return intersection, exclusive1, exclusive2
