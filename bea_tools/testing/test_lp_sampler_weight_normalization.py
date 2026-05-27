from __future__ import annotations

import pandas as pd
import pulp
import pytest

from bea_tools import FeatureConstraint, LPSampler, UniquenessConstraint


@pytest.fixture
def balanced_feature_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": list(range(8)),
            "group": ["A"] * 4 + ["B"] * 4,
        }
    )


def _cbc_sampler() -> LPSampler:
    sampler = LPSampler(verbose_solver=False)
    sampler.solver = pulp.getSolver("PULP_CBC_CMD", timeLimit=30, msg=False)
    return sampler


def test_non_unit_weights_still_yield_feasible_strict_sample(
    balanced_feature_data: pd.DataFrame,
) -> None:
    feature = FeatureConstraint(
        name="group",
        levels=["A", "B"],
        weights=[0.5005, 0.5005],
        strictness=1.0,
    )

    sampled = _cbc_sampler().sample_data(
        data=balanced_feature_data,
        features=feature,
        constraints=[UniquenessConstraint(id_col="row_id")],
        n=4,
        strict=True,
    )

    assert len(sampled) == 4
    assert sampled["group"].value_counts().to_dict() == {"A": 2, "B": 2}


def test_non_unit_weights_are_normalized_with_warning() -> None:
    with pytest.warns(UserWarning, match="Feature 'group'.*weights.*normaliz"):
        feature = FeatureConstraint(
            name="group",
            levels=["A", "B"],
            weights=[0.5005, 0.5005],
            strictness=1.0,
        )

    assert sum(feature.weights) == pytest.approx(1.0)
    assert feature.weights == pytest.approx([0.5, 0.5])
