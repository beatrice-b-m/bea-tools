import pandas as pd
import random
import math
from typing import Literal, Optional, Any, Union

# --- Data Structures ---


class Level:
    """Represents a single category/value within a Feature.

    Holds the configuration for a specific level, including how to match it
    in the dataframe, its target weight, and any conditional logic.

    Attributes:
        feature: Name of the parent feature.
        match_type: Strategy to match values ('contains', 'equals', 'between').
        name: The value(s) defining this level.
        weight: Target proportion for this level.
        count: Explicit target count (overrides weight).
        cond_weights: Conditional weights dependent on parent path choices.
        label: Output label for this level.
        strict: If True, this level cannot accept spillover samples.
        resampling_weight: Multiplier for prioritizing spillover absorption.
        query: Compiled pandas query string.
    """

    def __init__(
        self,
        feature: str,
        match_type: Literal["contains", "equals", "between"],
        name: Any,
        weight: float,
        count: Optional[int],
        cond_weights: Optional[dict[str, dict[str, float]]],
        label: Optional[str],
        strict: bool = False,
        resampling_weight: float = 1.0,
    ):
        self.feature = feature
        self.match_type = match_type
        self.name = name
        self.weight = weight
        self.count = count
        self.cond_weights = cond_weights
        self.label = label
        self.strict = strict
        self.resampling_weight = resampling_weight
        self.query = self._build_query()

    def _build_query(self) -> str:
        """Constructs the pandas query string based on match type."""
        val = f"'{self.name}'" if isinstance(self.name, str) else self.name

        if self.match_type == "equals":
            return f"{self.feature} == {val}"
        elif self.match_type == "contains":
            return f"{self.feature}.str.contains({val})"
        elif self.match_type == "between":
            if not isinstance(val, tuple):
                raise ValueError("Between match_type requires tuple values.")
            return f"({self.feature} > {val[0]}) & ({self.feature} <= {val[1]})"
        else:
            raise ValueError(f"Unknown match_type: {self.match_type}")


class Feature:
    """Defines a dimension of stratification.

    Orchestrates the creation of Levels and manages the distribution of
    weights and configuration across them.

    Attributes:
        name: Name of the column in the dataframe.
        match_type: Strategy to match values.
        levels: list of Level objects generated from configuration.
        label_col: Optional output column name for the matched label.
        strict: If True, levels in this feature won't accept spillover.
        resampling_weight: Priority multiplier for spillover absorption.
        balanced: If True, ignores weights/counts and ensures equal distribution across levels.
    """

    def __init__(
        self,
        name: str,
        match_type: Literal["contains", "equals", "between"],
        levels: list[Any],
        weights: Optional[list[float]] = None,
        conditional_weights: Optional[list[dict]] = None,
        counts: Optional[list[Optional[int]]] = None,
        labels: Optional[list[Optional[str]]] = None,
        label_col: Optional[str] = None,
        strict: bool = False,
        resampling_weight: float = 1.0,
        balanced: bool = False,
    ):
        """Initializes the Feature and generates child Level objects.

        Args:
            name: DataFrame column name.
            match_type: 'contains', 'equals', or 'between'.
            levels: list of values defining the levels.
            weights: list of float proportions (must align with levels).
            conditional_weights: Complex dictionary for path-dependent weights.
            counts: Explicit integer counts (overrides weights).
            labels: Output labels for the levels.
            label_col: Column to write labels to.
            strict: If True, prevents spillover re-balancing for this feature.
            resampling_weight: Multiplier for maintaining weight ratios.
            balanced: If True, ignores weights/counts and distributes equally across levels.
        """
        self.name = name
        self.label_col = label_col
        self.strict = strict
        self.resampling_weight = resampling_weight
        self.balanced = balanced

        n = len(levels)

        if weights is not None:
            _weights: list[float] = weights
        else:
            _weights: list[float] = [1.0 / n] * n

        if counts is not None:
            _counts: list[Optional[int]] = counts
        else:
            _counts: list[Optional[int]] = [None] * n

        if labels is not None:
            _labels: list[Optional[str]] = labels
        else:
            _labels: list[Optional[str]] = (
                [str(l) for l in levels] if label_col else [None] * n
            )

        # process conditional weights into lookup: {level_val: {target_feat: {target_level: weight}}}
        cond_lookup = {}
        if conditional_weights:
            for cw in conditional_weights:
                feat = cw["feature"]
                for lvl_name, weight_list in cw["weights"].items():
                    # mapping level index to weight index
                    for i, l_val in enumerate(levels):
                        if l_val not in cond_lookup:
                            cond_lookup[l_val] = {}
                        if feat not in cond_lookup[l_val]:
                            cond_lookup[l_val][feat] = {}
                        try:
                            cond_lookup[l_val][feat][lvl_name] = weight_list[i]
                        except IndexError:
                            pass

        self.levels = []
        for i in range(n):
            self.levels.append(
                Level(
                    feature=name,
                    match_type=match_type,
                    name=levels[i],
                    weight=_weights[i],
                    count=_counts[i],
                    cond_weights=cond_lookup.get(levels[i]),
                    label=_labels[i],
                    strict=strict,
                    resampling_weight=resampling_weight,
                )
            )


# --- Tree Logic ---


class SamplingNode:
    """A node in the hierarchical sampling tree.

    Manages a subset of data defined by the path from the root. Responsible for
    balancing sample targets with available capacity via hierarchical spillover.

    Attributes:
        name: Debug name (e.g., "Gender=Male").
        data: DataFrame subset at this node.
        target_n: The goal sample size for this node.
        capacity: The max available samples (unique patients/exams).
        route: The dictionary path of features taken to reach this node.
        children: list of child SamplingNodes.
        strict: If True, this node cannot accept extra samples (spillover).
        resampling_weight: Multiplier for spillover priority.
    """

    def __init__(
        self,
        name: str,
        data: pd.DataFrame,
        target_n: int,
        count_col: str,
        single_per_patient: bool,
        route: dict[str, str],
        strict: bool = False,
        resampling_weight: float = 1.0,
        sort_col: Optional[str] = "studydate_anon",
    ):
        self.name: str = name
        self.data: pd.DataFrame = data
        self.target_n: int = target_n
        self.route: dict[str, str] = route
        self.strict: bool = strict
        self.resampling_weight: float = resampling_weight
        self.sort_col: Optional[str] = sort_col

        # tree linkage
        self.children: list["SamplingNode"] = []

        # capacity calculation
        self.count_col: str = count_col
        self.single_per_patient: bool = single_per_patient

        self.ids: list[int] = data[count_col].unique().tolist()
        self.capacity: int = len(self.ids)
        self.original_target: int = target_n

    @property
    def is_leaf(self) -> bool:
        """Returns True if the node has no children."""
        return len(self.children) == 0

    @property
    def excess_n(self) -> int:
        """Returns the number of spare samples available (Capacity - Target)."""
        return self.capacity - self.target_n

    def add_child(self, node: "SamplingNode"):
        """Appends a child node."""
        self.children.append(node)

    def balance(self) -> int:
        """Performs Hierarchical Spillover to resolve deficits.

        Propagates deficits up from children and attempts to resolve them by
        distributing the load to 'wealthy' (surplus) siblings.

        Returns:
            int: The remaining unresolvable deficit for this subtree.
        """
        # 1. ask all children to balance themselves first (bottom-up)
        total_child_deficit = 0
        for child in self.children:
            total_child_deficit += child.balance()

        # 2. if i am a leaf, calculate my own simple deficit
        if self.is_leaf:
            if self.target_n > self.capacity:
                deficit = self.target_n - self.capacity
                self.target_n = self.capacity  # clip target to reality
                return deficit
            return 0

        # 3. if i am a parent node and my children have deficits
        if total_child_deficit > 0:
            # find wealthy siblings (children with surplus)
            # STRICT LOGIC: strict nodes cannot accept work, so they are not 'wealthy'
            wealthy_children = [
                c for c in self.children if c.excess_n > 0 and not c.strict
            ]
            total_surplus = sum(c.excess_n for c in wealthy_children)

            if total_surplus > 0:
                # distribute deficit to wealthy siblings
                remaining_deficit = total_child_deficit

                for child in wealthy_children:
                    available = child.excess_n
                    take_amount = min(remaining_deficit, available)

                    # push the extra work down to the wealthy child
                    child.absorb_surplus(take_amount)

                    remaining_deficit -= take_amount
                    if remaining_deficit == 0:
                        break

                return remaining_deficit  # pass remaining up to grandparent

            else:
                # no siblings have money, pass full bill to grandparent
                return total_child_deficit

        return 0

    def absorb_surplus(self, amount: int):
        """Recursively distributes extra work (samples) to children.

        Uses 'resampling_weight' to prioritize which children receive the
        extra load.

        Args:
            amount: The number of extra samples this node must take.
        """
        # strict check safety (should be handled by caller, but good for robustness)
        if self.strict:
            return

        self.target_n += amount
        if self.is_leaf:
            return

        remaining = amount

        # filter children eligible for spillover
        candidates = [c for c in self.children if not c.strict]
        if not candidates:
            return

        # --- PASS 1: Weighted Proportional Distribution ---
        # try to maintain the relative ratios, adjusted by resampling_weight
        # weighted target = target * resampling_weight
        total_weighted_target = sum(
            c.target_n * c.resampling_weight for c in candidates
        )

        if total_weighted_target > 0:
            for child in candidates:
                if remaining == 0:
                    break

                # calculate share based on weighted ratio
                weight_factor = child.target_n * child.resampling_weight
                ratio = weight_factor / total_weighted_target
                share = math.ceil(amount * ratio)

                # 1. don't take more than we have left to give
                share = min(share, remaining)

                # 2. don't overflow the child's physical capacity
                max_absorbable = child.capacity - child.target_n
                share = min(share, max_absorbable)

                if share > 0:
                    child.absorb_surplus(share)
                    remaining -= share

        # --- PASS 2: Greedy Cleanup ---
        # if 'remaining' is still > 0, dump into ANY eligible node with space
        if remaining > 0:
            # sort children by who has the most space left
            sorted_children = sorted(
                candidates, key=lambda c: (c.capacity - c.target_n), reverse=True
            )

            for child in sorted_children:
                if remaining == 0:
                    break

                max_absorbable = child.capacity - child.target_n
                take = min(remaining, max_absorbable)

                if take > 0:
                    child.absorb_surplus(take)
                    remaining -= take

    def collect_leaves(self) -> list["SamplingNode"]:
        """Recursively collects all leaf nodes."""
        if self.is_leaf:
            return [self]
        leaves = []
        for c in self.children:
            leaves.extend(c.collect_leaves())
        return leaves

    def refresh_ids(self, exclude_patients: list[int]):
        """Recalculates capacity by excluding used patients (for greedy sampler)."""
        if self.single_per_patient:
            current_ids = set(self.ids)
            used = set(exclude_patients)
            # update available ids and capacity
            self.ids = list(current_ids - used)
            self.capacity = len(self.ids)

    def sample(self, rng: random.Random) -> pd.DataFrame:
        """Extracts the final random sample from this node's data."""
        n = min(self.capacity, self.target_n)
        if n <= 0:
            return pd.DataFrame()

        sampled_ids = rng.sample(self.ids, n)

        # single-per-patient only (sorted)
        if self.single_per_patient and (self.sort_col is not None):
            return self.data[self.data[self.count_col].isin(sampled_ids)].sort_values(self.sort_col).drop_duplicates("empi_anon", keep="first")  # type: ignore

        # single-per-patient only (arbitrary selection)
        # TODO: default here is to select first, do we add ability to randomly choose etc.?
        elif self.single_per_patient:
            return self.data[self.data[self.count_col].isin(sampled_ids)].drop_duplicates("empi_anon", keep="first")  # type: ignore

        # multiple-per-patient
        else:
            return self.data[self.data[self.count_col].isin(sampled_ids)]  # type: ignore

    def __repr__(self):
        return f"Node({self.name} | Target: {self.target_n} | Cap: {self.capacity} | Strict: {self.strict})"


# --- Main Controller ---


class TreeSampler:
    """Orchestrates the stratified sampling process.

    Builds the tree, balances targets, and executes the greedy sampling.

    Attributes:
        n: Total target sample size.
        features: list of Feature configurations.
        seed: Random seed.
        count_col: Column used for ID/Capacity counting (e.g., patient ID).
        single_per_patient: If True, only one row per count_col ID is sampled.
    """

    def __init__(
        self,
        n: int,
        features: list[Feature],
        seed: int = 13,
        count_col: str = "empi_anon",
        sort_col: Optional[str] = "studydate_anon",
        single_per_patient: bool = True,
    ):
        self.n = n
        self.features = features
        self.seed = seed
        self.count_col = count_col
        self.single_per_patient = single_per_patient
        self.rng = random.Random(seed)
        self.patients = []  # track used patients across sampling steps
        self.sort_col: Optional[str] = sort_col

    def sample_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Executes the full sampling pipeline."""
        print("1. Building Tree...")
        # create a dummy root node to hold the structure
        root = SamplingNode(
            "ROOT",
            data,
            self.n,
            self.count_col,
            self.single_per_patient,
            {},
            sort_col=self.sort_col,
        )

        # recursively build tree (pass 1)
        self._build_tree(root, data, 0)

        print("2. Balancing Tree (Hierarchical Spillover)...")
        # this modifies target_n in place across the tree
        unresolved = root.balance()
        if unresolved > 0:
            print(
                f"Warning: Could not satisfy total sample size. Short by {unresolved}"
            )

        print("3. Sampling...")
        # collect all leaves to perform the actual extraction
        leaves = root.collect_leaves()

        # standard greedy sampling loop
        # we make a copy of leaves to work on
        active_leaves = [l for l in leaves if l.target_n > 0]
        final_samples = []

        while active_leaves:
            # sort: prioritize nodes with least 'safety margin' (Capacity - Target)
            active_leaves.sort(key=lambda x: x.capacity - x.target_n)

            # pick the most critical node
            node = active_leaves[0]

            # sample
            sample_df = node.sample(self.rng)
            final_samples.append(sample_df)

            # record used patients
            new_patients = sample_df[self.count_col].unique().tolist()
            self.patients.extend(new_patients)

            # remove this node from active list
            active_leaves.pop(0)

            # refresh other nodes (update capacities based on used patients)
            if self.single_per_patient:
                for l in active_leaves:
                    l.refresh_ids(self.patients)

        return (
            pd.concat(final_samples, ignore_index=True)
            if final_samples
            else pd.DataFrame()
        )

    def _build_tree(self, parent_node: SamplingNode, data: pd.DataFrame, f_idx: int):
        """Recursively constructs the sampling tree."""
        if f_idx >= len(self.features):
            return

        feature = self.features[f_idx]

        # calculate theoretical targets for children
        parent_n = parent_node.target_n
        remaining_n = parent_n

        # --- BALANCED MODE: Equal distribution across levels ---
        if feature.balanced:
            n_levels = len(feature.levels)
            base_target = parent_n // n_levels
            extra = parent_n % n_levels

            for i, level in enumerate(feature.levels):
                # Distribute remainder across first 'extra' levels
                target = base_target + (1 if i < extra else 0)

                # slice data
                level_df = data.query(level.query)
                if feature.label_col:
                    level_df = level_df.copy()
                    level_df[feature.label_col] = level.label

                # update route
                new_route = parent_node.route.copy()
                new_route[feature.name] = str(level.name)

                # create node
                child_node = SamplingNode(
                    name=f"{feature.name}={level.name}",
                    data=level_df,
                    target_n=target,
                    count_col=self.count_col,
                    single_per_patient=self.single_per_patient,
                    route=new_route,
                    strict=level.strict,
                    resampling_weight=level.resampling_weight,
                    sort_col=self.sort_col,
                )

                parent_node.add_child(child_node)

                # recurse
                self._build_tree(child_node, level_df, f_idx + 1)

        # --- STANDARD MODE: Use weights/counts ---
        else:
            for i, level in enumerate(feature.levels):
                is_last_level = i == len(feature.levels) - 1

                # --- Target Calculation ---
                # determine target, then check if we need to force-fix it
                # to ensure conservation of mass

                if level.count is not None and f_idx == 0:
                    target = level.count
                elif level.cond_weights and parent_node.route:
                    w = 1.0
                    for req_feat, weight_map in level.cond_weights.items():
                        prev_val = parent_node.route.get(req_feat)
                        w *= weight_map.get(prev_val, 1.0)
                    target = int(w * parent_n)
                else:
                    target = int(level.weight * parent_n)

                # force last level to take remainder
                if is_last_level:
                    target = remaining_n

                remaining_n -= target

                # slice data
                level_df = data.query(level.query)
                if feature.label_col:
                    level_df = level_df.copy()
                    level_df[feature.label_col] = level.label

                # update route
                new_route = parent_node.route.copy()
                new_route[feature.name] = str(level.name)

                # create node
                child_node = SamplingNode(
                    name=f"{feature.name}={level.name}",
                    data=level_df,
                    target_n=target,
                    count_col=self.count_col,
                    single_per_patient=self.single_per_patient,
                    route=new_route,
                    strict=level.strict,
                    resampling_weight=level.resampling_weight,
                    sort_col=self.sort_col,
                )

                parent_node.add_child(child_node)

                # recurse
                self._build_tree(child_node, level_df, f_idx + 1)
