from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any
import pandas as pd
import pulp
import warnings
import uuid
import itertools


# core structures


@dataclass(frozen=True)
class PathStep:
    """represents a single decision node in the feature hierarchy.

    Attributes:
        feature_name: name of the feature this step represents.
        level_name: the specific level/value within the feature.
        label_val: the label to use in output for this step.
        output_col: column name for output labeling, none if not requested.
    """

    feature_name: str
    level_name: str
    label_val: str
    output_col: Optional[str]  # none if no label column was requested for this feature


# unique sequence of decisions that assigns a row to a specific leaf
PathSignature = tuple[PathStep, ...]


class BaseConstraint(ABC):
    """protocol that all constraints must implement."""

    @abstractmethod
    def apply(
        self,
        prob: pulp.LpProblem,
        data: pd.DataFrame,
        lp_vars: dict[int, dict[PathSignature, pulp.LpVariable]],
        penalty_terms: list[Any],
    ) -> None:
        """injects specific linear equations into the problem.

        Args:
            prob: the active PuLP problem.
            data: the source dataframe (read-only).
            lp_vars: registry of variables {row_idx: {PathSignature: LpVariable}}.
            penalty_terms: list to accumulate penalty expressions.
        """
        pass


# constraint classes


class UniquenessConstraint(BaseConstraint):
    """enforces cardinality constraints (e.g., 'max n rows per patient').

    always treated as a hard constraint.

    Attributes:
        id_col: column name to group rows by.
        n: maximum number of rows allowed per unique id.
    """

    id_col: str
    n: int

    def __init__(self, id_col: str, n: int = 1) -> None:
        """initializes uniqueness constraint.

        Args:
            id_col: column name to group rows by.
            n: maximum number of rows allowed per unique id. defaults to 1.
        """
        self.id_col = id_col
        self.n = n

    def apply(
        self,
        prob: pulp.LpProblem,
        data: pd.DataFrame,
        lp_vars: dict[int, dict[PathSignature, pulp.LpVariable]],
        penalty_terms: list[Any],
    ) -> None:
        # group active variables by id (build dict once to avoid n^2 lookups)
        vars_by_id: dict[Any, list[pulp.LpVariable]] = {}

        for row_idx, paths in lp_vars.items():
            uid: Any = data.at[row_idx, self.id_col]
            if uid not in vars_by_id:
                vars_by_id[uid] = []
            vars_by_id[uid].extend(paths.values())

        # apply constraints
        for uid, variables in vars_by_id.items():
            # only add constraint if group size could violate it
            if len(variables) > self.n:
                prob += pulp.lpSum(variables) <= self.n, f"Unique_{self.id_col}_{uid}"


class FeatureConstraint(BaseConstraint):
    """hierarchical distribution constraint.

    supports linking children, fuzzy matching ('contains'), and output labeling.

    Attributes:
        name: feature column name in the dataframe.
        levels: list of possible values/levels for this feature.
        weights: target distribution weights for each level (must sum to 1.0).
        how: matching strategy, either 'equals' or 'contains'.
        strictness: constraint strictness (1.0 = hard, <1.0 = soft).
        label_col: optional output column name for labeling selected rows.
        labels: custom labels for each level (defaults to level names).
        children: nested child constraints mapped by level.
    """

    name: str
    levels: list[str]
    weights: list[float]
    how: str
    strictness: float
    label_col: Optional[str]
    labels: list[str]
    children: dict[str, list[FeatureConstraint]]

    def __init__(
        self,
        name: str,
        levels: list[str],
        weights: Optional[list[float]] = None,
        how: str = "equals",
        strictness: float = 1.0,
        label_col: Optional[str] = None,
        labels: Optional[list[str]] = None,
    ) -> None:
        """initializes feature constraint.

        Args:
            name: feature column name in the dataframe.
            levels: list of possible values/levels for this feature.
            weights: target distribution weights for each level. defaults to uniform.
            how: matching strategy, either 'equals' or 'contains'. defaults to 'equals'.
            strictness: constraint strictness (1.0 = hard, <1.0 = soft). defaults to 1.0.
            label_col: optional output column name for labeling selected rows.
            labels: custom labels for each level. defaults to level names.

        Raises:
            ValueError: if levels and weights have different lengths.
            ValueError: if levels and labels have different lengths.
        """
        self.name = name
        self.levels = levels
        self.weights = weights or [1.0 / len(levels)] * len(levels)
        self.how = how  # 'equals' or 'contains'
        self.strictness = strictness
        self.label_col = label_col
        self.labels = labels if labels else levels

        # hierarchy: map level_name -> list[child_constraints]
        self.children: dict[str, list[FeatureConstraint]] = {
            lvl: [] for lvl in self.levels
        }

        # validation
        if len(self.levels) != len(self.weights):
            raise ValueError(f"Feature '{name}': Mismatch between levels and weights.")
        if labels and len(labels) != len(levels):
            raise ValueError(f"Feature '{name}': Mismatch between levels and labels.")

    def link(self, child: "FeatureConstraint", levels: Optional[list[str]] = None) -> None:
        """attaches a child constraint to specific levels of this feature.

        Args:
            child: the child feature_constraint to attach.
            levels: specific levels to attach child to. defaults to all levels.

        Raises:
            ValueError: if specified level is not found in this feature.
        """
        target_levels: list[str] = levels if levels else self.levels
        for lvl in target_levels:
            if lvl in self.children:
                self.children[lvl].append(child)
            else:
                raise ValueError(f"Level '{lvl}' not found in feature '{self.name}'")

    def _match_row(self, val: Any, level: str) -> bool:
        """helper for equality vs contains logic.

        Args:
            val: the value from the dataframe to match.
            level: the level string to match against.

        Returns:
            true if the value matches the level according to self.how.
        """
        val_str: str = str(val)
        if self.how == "equals":
            return val_str == level
        elif self.how == "contains":
            return level in val_str
        return False

    def get_valid_paths(self, row_idx: int, data: pd.DataFrame) -> list[list[PathStep]]:
        """recursively finds all valid assignment paths for a single row.

        Args:
            row_idx: index of the row in the dataframe.
            data: the source dataframe.

        Returns:
            list of paths, where each path is a list of path_steps.
        """
        row_val: Any = data.at[row_idx, self.name]
        valid_paths: list[list[PathStep]] = []

        for i, lvl in enumerate(self.levels):
            if self._match_row(row_val, lvl):
                # create the step for this node
                current_step: PathStep = PathStep(
                    feature_name=self.name,
                    level_name=lvl,
                    label_val=self.labels[i],
                    output_col=self.label_col,
                )

                children: list[FeatureConstraint] = self.children[lvl]

                if not children:
                    # leaf node
                    valid_paths.append([current_step])
                else:
                    # branch node: recursively collect child paths
                    # assumes AND logic: row must satisfy at least one path in every child chain
                    # for a single chain, just concatenate
                    # if multiple independent children exist, need cartesian product

                    # collect all valid paths for each child
                    child_path_groups: Optional[list[list[list[PathStep]]]] = []
                    for child in children:
                        c_paths: list[list[PathStep]] = child.get_valid_paths(row_idx, data)
                        if not c_paths:
                            child_path_groups = None  # failed a required child constraint
                            break
                        child_path_groups.append(c_paths)

                    if child_path_groups is None:
                        continue  # skip this level, row doesn't satisfy children

                    # cartesian product to combine paths from multiple children
                    # e.g., child_a has 2 valid paths, child_b has 1 -> 2 total combinations
                    for combination in itertools.product(*child_path_groups):
                        # combination is a tuple of lists (paths), flatten it
                        combined_suffix: list[PathStep] = []
                        for p in combination:
                            combined_suffix.extend(p)
                        valid_paths.append([current_step] + combined_suffix)

        return valid_paths

    def apply(
        self,
        prob: pulp.LpProblem,
        data: pd.DataFrame,
        lp_vars: dict[int, dict[PathSignature, pulp.LpVariable]],
        penalty_terms: list[Any],
    ) -> None:
        """entry point for applying distribution constraints.

        Args:
            prob: the active PuLP problem.
            data: the source dataframe (read-only).
            lp_vars: registry of variables {row_idx: {PathSignature: LpVariable}}.
            penalty_terms: list to accumulate penalty expressions.
        """
        self._apply_recursive(prob, lp_vars, penalty_terms, parent_vars=None)

    def _apply_recursive(
        self,
        prob: pulp.LpProblem,
        lp_vars: dict[int, dict[PathSignature, pulp.LpVariable]],
        penalty_terms: list[Any],
        parent_vars: Optional[list[pulp.LpVariable]],
    ) -> None:
        """recursively applies distribution constraints to this feature and children.

        Args:
            prob: the active PuLP problem.
            lp_vars: registry of variables {row_idx: {PathSignature: LpVariable}}.
            penalty_terms: list to accumulate penalty expressions.
            parent_vars: variables from parent scope, none for root level.
        """
        # filter variables: get all vars that passed through the parent node (if any)
        # and group them by the levels of this feature
        vars_by_level: dict[str, list[pulp.LpVariable]] = {lvl: [] for lvl in self.levels}
        all_vars_in_scope: list[pulp.LpVariable] = []

        for row_idx, paths in lp_vars.items():
            for path, var in paths.items():
                # optimization: in large-scale systems, would index this better
                # here we scan the path signature

                # check if this var belongs to the parent scope
                if parent_vars is not None and var not in parent_vars:
                    continue

                # find the step corresponding to this feature
                step: Optional[PathStep] = next((s for s in path if s.feature_name == self.name), None)
                if step:
                    vars_by_level[step.level_name].append(var)
                    all_vars_in_scope.append(var)

        if not all_vars_in_scope:
            return

        # apply distribution constraints
        total_scope_sum: pulp.LpAffineExpression = pulp.lpSum(all_vars_in_scope)

        for i, lvl in enumerate(self.levels):
            vars_in_level: list[pulp.LpVariable] = vars_by_level[lvl]
            target_weight: float = self.weights[i]

            # equation: sum(level) == target * sum(total)
            lhs: pulp.LpAffineExpression = pulp.lpSum(vars_in_level)
            rhs: pulp.LpAffineExpression = target_weight * total_scope_sum

            uid: str = str(uuid.uuid4())
            self._add_soft_equality(
                prob,
                lhs,
                rhs,
                self.strictness,
                penalty_terms,
                f"{self.name}_{lvl}_{uid}",
            )

            # recurse to children
            for child in self.children[lvl]:
                child._apply_recursive(prob, lp_vars, penalty_terms, vars_in_level)

    def _add_soft_equality(
        self,
        prob: pulp.LpProblem,
        lhs: pulp.LpAffineExpression,
        rhs: pulp.LpAffineExpression,
        strictness: float,
        penalty_terms: list[Any],
        name: str,
    ) -> None:
        """adds a soft or hard equality constraint based on strictness.

        Args:
            prob: the active PuLP problem.
            lhs: left-hand side expression.
            rhs: right-hand side expression.
            strictness: constraint strictness (>=1.0 = hard, <1.0 = soft).
            penalty_terms: list to accumulate penalty expressions.
            name: unique name for the constraint.
        """
        if strictness >= 1.0:
            prob += lhs == rhs, f"HardDist_{name}"
        elif strictness > 0.0:
            pos: pulp.LpVariable = pulp.LpVariable(f"slack_pos_{name}", lowBound=0)
            neg: pulp.LpVariable = pulp.LpVariable(f"slack_neg_{name}", lowBound=0)

            prob += lhs - rhs + pos - neg == 0, f"SoftDist_{name}"

            # append to list instead of +=
            penalty_weight: float = 100.0 * (1 + strictness * 10)
            penalty_terms.append(penalty_weight * (pos + neg))


class HomogeneityConstraint(BaseConstraint):
    """encourages a specific feature to have the same distribution across groups.

    uses symmetric global comparison (group vs global average) for stability.
    respects strictness >= 1.0 as a hard constraint (==).

    Attributes:
        feature: the feature column that should be homogeneous across groups.
        group_by: the feature column that defines groups.
        weights: target size weights for each group.
        strictness: constraint strictness (>=1.0 = hard, <1.0 = soft).
    """

    feature: str
    group_by: str
    weights: dict[str, float]
    strictness: float

    def __init__(
        self,
        feature: str,
        group_by: str,
        group_weights: dict[str, float],
        strictness: float = 0.5,
    ) -> None:
        """initializes homogeneity constraint.

        Args:
            feature: the feature column that should be homogeneous across groups.
            group_by: the feature column that defines groups.
            group_weights: target size weights for each group.
            strictness: constraint strictness (>=1.0 = hard, <1.0 = soft). defaults to 0.5.
        """
        self.feature = feature
        self.group_by = group_by
        self.weights = group_weights
        self.strictness = strictness

    def apply(
        self,
        prob: pulp.LpProblem,
        data: pd.DataFrame,
        lp_vars: dict[int, dict[PathSignature, pulp.LpVariable]],
        penalty_terms: list[Any],
    ) -> None:
        """applies homogeneity constraints to the problem.

        Args:
            prob: the active PuLP problem.
            data: the source dataframe (read-only).
            lp_vars: registry of variables {row_idx: {PathSignature: LpVariable}}.
            penalty_terms: list to accumulate penalty expressions.
        """
        if self.strictness <= 0.0:
            return

        # bucket variables by {feature_value -> {group -> [vars]}}
        target_levels: list[str] = sorted(data[self.feature].astype(str).unique())
        groups: list[str] = list(self.weights.keys())
        buckets: dict[str, dict[str, list[pulp.LpVariable]]] = {lvl: {g: [] for g in groups} for lvl in target_levels}

        # track all variables relevant to this constraint for global sums
        all_relevant_vars: dict[str, list[pulp.LpVariable]] = {lvl: [] for lvl in target_levels}

        for row_idx, paths in lp_vars.items():
            row_feat_val: str = str(data.at[row_idx, self.feature])

            # skip rows that don't have the target feature value
            if row_feat_val not in buckets:
                continue

            for path, var in paths.items():
                # find which group this variable is assigned to
                group_step: Optional[PathStep] = next(
                    (s for s in path if s.feature_name == self.group_by), None
                )

                if group_step and group_step.level_name in groups:
                    buckets[row_feat_val][group_step.level_name].append(var)
                    all_relevant_vars[row_feat_val].append(var)

        # build symmetric balance equations
        # target: count(group, val) should equal weight(group) * total_count(val)
        # linear form: count(group) - weight(group) * sum(all_relevant_vars) == 0

        for lvl in target_levels:
            # sum of all groups for this feature level to act as the "global total"
            # note: must use pulp.lpSum to delay evaluation until solve time
            total_count_expr: pulp.LpAffineExpression = pulp.lpSum(all_relevant_vars[lvl])

            # if no vars exist for this level at all, skip
            if len(all_relevant_vars[lvl]) == 0:
                continue

            for group in groups:
                group_vars: list[pulp.LpVariable] = buckets[lvl][group]
                group_weight: float = self.weights[group]

                # lhs: the actual count in this group
                lhs: pulp.LpAffineExpression = pulp.lpSum(group_vars)

                # rhs: the 'ideal' count based on the group's size weight
                rhs: pulp.LpAffineExpression = group_weight * total_count_expr

                # create constraint
                if self.strictness >= 1.0:
                    prob += lhs - rhs == 0, f"HardHomo_{self.feature}_{lvl}_{group}"
                else:
                    pos: pulp.LpVariable = pulp.LpVariable(f"h_pos_{lvl}_{group}", lowBound=0)
                    neg: pulp.LpVariable = pulp.LpVariable(f"h_neg_{lvl}_{group}", lowBound=0)

                    prob += (
                        lhs - rhs + pos - neg == 0,
                        f"SoftHomo_{self.feature}_{lvl}_{group}",
                    )

                    # exponential penalty (strictness 0.99 -> massive weight)
                    weight: float = 10.0 * (10.0 ** (self.strictness * 5.0))
                    penalty_terms.append(weight * (pos + neg))


# orchestrator


class LPSampler:
    """linear programming based sampler for constrained dataset sampling.

    uses integer linear programming to sample rows from a dataframe while
    satisfying distribution, uniqueness, and homogeneity constraints.

    Attributes:
        solver: the PuLP solver instance (highs or cbc).
    """

    solver: pulp.LpSolver

    def __init__(
        self,
        solver_timeout: int = 300,
        solver_path: str = "highs",
        verbose_solver: bool = False,
    ) -> None:
        """initializes the lp sampler with solver configuration.

        Args:
            solver_timeout: maximum solver time in seconds. defaults to 300.
            solver_path: path to solver executable. defaults to "highs".
            verbose_solver: whether to show solver output. defaults to false.
        """
        # highs is essential for performance on large datasets
        try:
            print("Trying to get the HiGHS_CMD solver...")
            self.solver = pulp.getSolver(
                "HiGHS_CMD",
                path=solver_path,
                timeLimit=solver_timeout,
                msg=verbose_solver,
            )
            print("HiGHS_CMD solver retrieved.")

        except AttributeError:
            warnings.warn(
                "highs solver unavailable. falling back to default cbc. performance may degrade."
            )
            print("Trying to get the PULP_CBC_CMD solver...")
            self.solver = pulp.getSolver(
                "PULP_CBC_CMD", timeLimit=solver_timeout, msg=verbose_solver
            )
            print("PULP_CBC_CMD solver retrieved.")

    def sample_data(
        self,
        data: pd.DataFrame,
        features: FeatureConstraint,
        constraints: list[BaseConstraint],
        n: int,
        strict: bool = True,
    ) -> pd.DataFrame:
        """main execution pipeline for sampling data.

        Args:
            data: source dataframe to sample from.
            features: the root feature_constraint object defining hierarchy.
            constraints: list of global constraints (uniqueness, homogeneity).
            n: desired sample size.
            strict: if true, fails if exact n cannot be met. if false, maximizes size up to n.

        Returns:
            sampled dataframe with n rows (or fewer if strict=false and infeasible).

        Raises:
            ValueError: if no rows match the root feature constraints.
        """

        # initialize problem
        prob: pulp.LpProblem = pulp.LpProblem("GranularSampler", pulp.LpMaximize)

        # generate candidate variables
        # structure: lp_vars[row_index][path_signature] = lpvariable
        lp_vars: dict[int, dict[PathSignature, pulp.LpVariable]] = {}
        all_vars_flat: list[pulp.LpVariable] = []

        # optimization: only iterate rows once
        for idx in data.index:
            valid_paths: list[list[PathStep]] = features.get_valid_paths(idx, data)

            if valid_paths:
                row_vars: dict[PathSignature, pulp.LpVariable] = {}
                for path in valid_paths:
                    path_tuple: PathSignature = tuple(path)
                    # unique name for the variable
                    var_name: str = f"x_{idx}_{abs(hash(path_tuple))}"
                    var: pulp.LpVariable = pulp.LpVariable(var_name, cat=pulp.LpBinary)

                    row_vars[path_tuple] = var
                    all_vars_flat.append(var)

                lp_vars[idx] = row_vars

        if not all_vars_flat:
            raise ValueError("no rows in data matched the root feature constraints.")

        # apply constraints
        penalty_terms: list[Any] = []

        # root feature (recursive)
        features.apply(prob, data, lp_vars, penalty_terms)

        # global constraints
        for constr in constraints:
            constr.apply(prob, data, lp_vars, penalty_terms)

        # define objective function
        total_selected: pulp.LpAffineExpression = pulp.lpSum(all_vars_flat)
        total_penalty: pulp.LpAffineExpression = pulp.lpSum(penalty_terms)  # sum the list into one expression

        if strict:
            prob += total_selected == n, "StrictSize"
            prob += -1 * total_penalty
        else:
            prob += total_selected <= n, "MaxSize"
            prob += (1000 * total_selected) - total_penalty

        # solve
        print("\nStarting solver...")
        prob.solve(self.solver)
        print("Solver finished.\n")

        if prob.status != pulp.LpStatusOptimal:
            msg: str = f"solver status: {pulp.LpStatus[prob.status]}"
            if strict:
                warnings.warn(
                    f"{msg}. strict requirements could not be met. returning empty dataframe."
                )
                return pd.DataFrame()
            else:
                warnings.warn(f"{msg}. returning best partial result.")

        # reconstruct and label
        selected_rows: list[int] = []
        labels_to_inject: dict[int, dict[str, str]] = {}

        for idx, paths in lp_vars.items():
            for path_tuple, var in paths.items():
                if var.varValue and var.varValue > 0.5:
                    selected_rows.append(idx)

                    # extract labels from the selected path
                    row_labels: dict[str, str] = {}
                    for step in path_tuple:
                        if step.output_col:
                            row_labels[step.output_col] = step.label_val

                    if row_labels:
                        labels_to_inject[idx] = row_labels

                    # break loop: a row can only be selected once
                    # (uniqueness_constraint ensures this, but we break to be safe)
                    break

        # build result dataframe
        result: pd.DataFrame = data.loc[selected_rows].copy()

        # inject labels
        # identify all label columns that were used
        all_label_cols: set[str] = set()
        for labs in labels_to_inject.values():
            all_label_cols.update(labs.keys())

        # initialize cols with none if they don't exist
        for col in all_label_cols:
            if col not in result.columns:
                result[col] = None

        # fill values
        for idx, labs in labels_to_inject.items():
            for col, val in labs.items():
                result.at[idx, col] = val

        return result
