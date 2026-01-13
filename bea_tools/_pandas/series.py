import pandas as pd
from pandas.api.extensions import register_series_accessor
from typing import Union, Optional, overload, Literal

# handle display import for non-Jupyter environments
try:
    from IPython.display import display  # type: ignore
except ImportError:

    def display(obj):
        print(obj)


@register_series_accessor("bea")
class BeaSeriesTools:
    """Custom tools for working with pd.Series objects. Made by, and for, Bea c:"""

    def __init__(self, pandas_object: pd.Series) -> None:
        self._obj = pandas_object

    # Overload 1: output=False -> Returns None (Display only)
    @overload
    def value_counts(
        self,
        normalize: bool = ...,
        with_proportion: bool = ...,
        n_decimals: int = ...,
        sort: Union[bool, list[any]] = ...,
        output: Literal[False] = ...,
    ) -> None: ...

    # Overload 2: with_proportion=True -> Returns dict[any, str] (Strings)
    @overload
    def value_counts(
        self,
        normalize: bool = ...,
        with_proportion: Literal[True] = ...,
        n_decimals: int = ...,
        sort: Union[bool, list[any]] = ...,
        output: Literal[True] = ...,
    ) -> dict[any, str]: ...

    # Overload 3: normalize=True -> Returns dict[any, float] (Floats)
    @overload
    def value_counts(
        self,
        normalize: Literal[True] = ...,
        with_proportion: Literal[False] = ...,
        n_decimals: int = ...,
        sort: Union[bool, list[any]] = ...,
        output: Literal[True] = ...,
    ) -> dict[any, float]: ...

    # Overload 4: Default -> Returns dict[any, int] (Integers)
    @overload
    def value_counts(
        self,
        normalize: Literal[False] = ...,
        with_proportion: Literal[False] = ...,
        n_decimals: int = ...,
        sort: Union[bool, list[any]] = ...,
        output: Literal[True] = ...,
    ) -> dict[any, int]: ...

    def value_counts(
        self,
        normalize: bool = False,
        with_proportion: bool = False,
        n_decimals: int = 1,
        sort: Union[bool, list[any]] = True,
        output: bool = False,
        **kwargs,
    ) -> Optional[dict[any, any]]:
        """
        Calculates value counts with options for custom sorting, normalization,
        and formatted string output.

        Args:
            normalize (bool, optional): If True, returns proportions (0.0 to 1.0)
                instead of counts. Defaults to False.
            with_proportion (bool, optional): If True, returns formatted strings
                in the format "n (n/total%)". Overrides 'normalize'. Defaults to False.
            n_decimals (int, optional): Number of decimal places to use for
                percentages when with_proportion is True. Defaults to 1.
            sort (Union[bool, list[any]], optional): If True/False, uses standard
                Pandas sorting. If a list is provided, the result is reindexed
                to match the list order, filling missing values with 0. Defaults to True.
            output (bool, optional): If True, returns the result as a dictionary.
                If False, displays the result using IPython's display() and returns None.
                Defaults to True.

        Returns:
            Optional[dict[any, any]]:
                - dict[any, int] if standard counts.
                - dict[any, float] if normalize=True.
                - dict[any, str] if with_proportion=True.
                - None if output=False.
        """
        # base Counts Calculation
        if isinstance(sort, list):
            # calculate unsorted first for speed
            counts = self._obj.value_counts(sort=False, **kwargs)
            # reindex handles filtering, ordering, and filling zeros simultaneously
            counts = counts.reindex(sort, fill_value=0)
        else:
            counts = self._obj.value_counts(sort=sort, **kwargs)

        # determine Output Data Format
        result_series: pd.Series

        if with_proportion:
            # calculate total from the original object to ensure percentages are correct
            # even if the custom sort list filtered out data.
            total = len(self._obj)
            pcts = (counts / total) * 100

            # vectorized string formatting
            result_series = (
                counts.astype(str) + " (" + pcts.round(n_decimals).astype(str) + "%)"
            )

        elif normalize:
            total = len(self._obj)
            result_series = counts / total

        else:
            result_series = counts

        # 3. Output Handling
        if output:
            return result_series.to_dict()
        else:
            display(result_series)
            return None
