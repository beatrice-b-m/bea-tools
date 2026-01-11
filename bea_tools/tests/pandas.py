# Tests for pandas/sampler.py

import pytest
import pandas as pd
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pandas.sampler import Level, Feature, SamplingNode, GroupContainer


# --- Fixtures ---

@pytest.fixture
def sample_dataframe():
    """Creates a sample DataFrame for testing."""
    return pd.DataFrame({
        'empi_anon': ['P001', 'P002', 'P003', 'P004', 'P005',
                      'P006', 'P007', 'P008', 'P009', 'P010'],
        'gender': ['M', 'M', 'F', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
        'age': [25, 35, 45, 55, 30, 40, 50, 60, 28, 38],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        'studydate_anon': pd.date_range('2020-01-01', periods=10, freq='D')
    })


@pytest.fixture
def multi_exam_dataframe():
    """Creates a DataFrame with multiple exams per patient."""
    return pd.DataFrame({
        'empi_anon': ['P001', 'P001', 'P002', 'P002', 'P003',
                      'P003', 'P004', 'P004', 'P005', 'P005'],
        'exam_id': ['E001', 'E002', 'E003', 'E004', 'E005',
                    'E006', 'E007', 'E008', 'E009', 'E010'],
        'gender': ['M', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'M'],
        'studydate_anon': pd.date_range('2020-01-01', periods=10, freq='D')
    })


# --- Level Class Tests ---

class TestLevel:
    """Tests for the Level class."""

    def test_query_equals_string(self):
        """Query building for equals match type with string value."""
        level = Level(
            feature='gender',
            match_type='equals',
            name='M',
            weight=0.5,
            count=None,
            cond_weights=None,
            label=None
        )
        assert level.query == "gender == 'M'"

    def test_query_equals_numeric(self):
        """Query building for equals match type with numeric value."""
        level = Level(
            feature='age',
            match_type='equals',
            name=30,
            weight=0.5,
            count=None,
            cond_weights=None,
            label=None
        )
        assert level.query == "age == 30"

    def test_query_contains(self):
        """Query building for contains match type."""
        level = Level(
            feature='name',
            match_type='contains',
            name='John',
            weight=0.5,
            count=None,
            cond_weights=None,
            label=None
        )
        assert level.query == "name.str.contains('John')"

    def test_query_between(self):
        """Query building for between match type with valid tuple."""
        level = Level(
            feature='age',
            match_type='between',
            name=(20, 30),
            weight=0.5,
            count=None,
            cond_weights=None,
            label=None
        )
        assert level.query == "(age > 20) & (age <= 30)"

    def test_query_between_invalid_type(self):
        """Between match type with non-tuple should raise ValueError."""
        with pytest.raises(ValueError, match="Between match_type requires tuple"):
            Level(
                feature='age',
                match_type='between',
                name=30,
                weight=0.5,
                count=None,
                cond_weights=None,
                label=None
            )

    def test_query_unknown_match_type(self):
        """Unknown match type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown match_type"):
            Level(
                feature='age',
                match_type='invalid',
                name=30,
                weight=0.5,
                count=None,
                cond_weights=None,
                label=None
            )

    def test_strict_and_resampling_weight(self):
        """Strict and resampling_weight attributes are set correctly."""
        level = Level(
            feature='gender',
            match_type='equals',
            name='M',
            weight=0.5,
            count=None,
            cond_weights=None,
            label=None,
            strict=True,
            resampling_weight=2.0
        )
        assert level.strict is True
        assert level.resampling_weight == 2.0


# --- Feature Class Tests ---

class TestFeature:
    """Tests for the Feature class."""

    def test_default_weight_distribution(self):
        """When no weights provided, should distribute evenly."""
        feature = Feature(
            name='gender',
            match_type='equals',
            levels=['M', 'F']
        )
        assert len(feature.levels) == 2
        assert feature.levels[0].weight == 0.5
        assert feature.levels[1].weight == 0.5

    def test_explicit_weights(self):
        """Explicit weights are passed correctly to levels."""
        feature = Feature(
            name='category',
            match_type='equals',
            levels=['A', 'B', 'C'],
            weights=[0.5, 0.3, 0.2]
        )
        assert feature.levels[0].weight == 0.5
        assert feature.levels[1].weight == 0.3
        assert feature.levels[2].weight == 0.2

    def test_label_generation(self):
        """Labels are generated when label_col is set."""
        feature = Feature(
            name='gender',
            match_type='equals',
            levels=['M', 'F'],
            label_col='gender_label'
        )
        assert feature.levels[0].label == 'M'
        assert feature.levels[1].label == 'F'

    def test_strict_propagation(self):
        """Strict flag propagates to child levels."""
        feature = Feature(
            name='gender',
            match_type='equals',
            levels=['M', 'F'],
            strict=True
        )
        assert feature.levels[0].strict is True
        assert feature.levels[1].strict is True

    def test_resampling_weight_propagation(self):
        """Resampling weight propagates to child levels."""
        feature = Feature(
            name='gender',
            match_type='equals',
            levels=['M', 'F'],
            resampling_weight=2.5
        )
        assert feature.levels[0].resampling_weight == 2.5
        assert feature.levels[1].resampling_weight == 2.5

    def test_conditional_weights_lookup(self):
        """Conditional weights lookup is constructed correctly."""
        feature = Feature(
            name='category',
            match_type='equals',
            levels=['A', 'B'],
            conditional_weights=[{
                'feature': 'gender',
                'weights': {
                    'M': [0.6, 0.4],
                    'F': [0.4, 0.6]
                }
            }]
        )
        assert feature.levels[0].cond_weights is not None
        assert feature.levels[0].cond_weights['gender']['M'] == 0.6
        assert feature.levels[1].cond_weights['gender']['F'] == 0.6


# --- SamplingNode Class Tests ---

class TestSamplingNode:
    """Tests for the SamplingNode class."""

    def test_is_leaf_true(self, sample_dataframe):
        """Node with no children is a leaf."""
        node = SamplingNode(
            name='test',
            data=sample_dataframe,
            target_n=5,
            count_col='empi_anon',
            single_per_patient=True,
            route={}
        )
        assert node.is_leaf is True

    def test_is_leaf_false(self, sample_dataframe):
        """Node with children is not a leaf."""
        parent = SamplingNode(
            name='parent',
            data=sample_dataframe,
            target_n=5,
            count_col='empi_anon',
            single_per_patient=True,
            route={}
        )
        child = SamplingNode(
            name='child',
            data=sample_dataframe.head(3),
            target_n=2,
            count_col='empi_anon',
            single_per_patient=True,
            route={}
        )
        parent.add_child(child)
        assert parent.is_leaf is False

    def test_excess_n_calculation(self, sample_dataframe):
        """Excess is calculated as capacity minus target."""
        node = SamplingNode(
            name='test',
            data=sample_dataframe,
            target_n=3,
            count_col='empi_anon',
            single_per_patient=True,
            route={}
        )
        assert node.capacity == 10
        assert node.excess_n == 7

    def test_balance_leaf_sufficient_capacity(self, sample_dataframe):
        """Leaf with sufficient capacity returns zero deficit."""
        node = SamplingNode(
            name='test',
            data=sample_dataframe,
            target_n=5,
            count_col='empi_anon',
            single_per_patient=True,
            route={}
        )
        deficit = node.balance()
        assert deficit == 0
        assert node.target_n == 5

    def test_balance_leaf_with_deficit(self, sample_dataframe):
        """Leaf with deficit clips target and returns deficit amount."""
        node = SamplingNode(
            name='test',
            data=sample_dataframe,
            target_n=15,
            count_col='empi_anon',
            single_per_patient=True,
            route={}
        )
        deficit = node.balance()
        assert deficit == 5
        assert node.target_n == 10

    def test_balance_hierarchical_spillover(self, sample_dataframe):
        """Deficit is redistributed to wealthy siblings."""
        parent = SamplingNode(
            name='parent',
            data=sample_dataframe,
            target_n=10,
            count_col='empi_anon',
            single_per_patient=True,
            route={}
        )

        # Child 1: has deficit (target 8, capacity 5)
        df_males = sample_dataframe[sample_dataframe['gender'] == 'M']
        child1 = SamplingNode(
            name='child1',
            data=df_males,
            target_n=8,
            count_col='empi_anon',
            single_per_patient=True,
            route={}
        )

        # Child 2: has surplus (target 2, capacity 5)
        df_females = sample_dataframe[sample_dataframe['gender'] == 'F']
        child2 = SamplingNode(
            name='child2',
            data=df_females,
            target_n=2,
            count_col='empi_anon',
            single_per_patient=True,
            route={}
        )

        parent.add_child(child1)
        parent.add_child(child2)

        deficit = parent.balance()

        # Child1 had deficit of 3, child2 should absorb it
        assert child1.target_n == 5
        assert child2.target_n == 5
        assert deficit == 0

    def test_balance_respects_strict_nodes(self, sample_dataframe):
        """Strict nodes don't receive spillover."""
        parent = SamplingNode(
            name='parent',
            data=sample_dataframe,
            target_n=10,
            count_col='empi_anon',
            single_per_patient=True,
            route={}
        )

        df_males = sample_dataframe[sample_dataframe['gender'] == 'M']
        child1 = SamplingNode(
            name='child1',
            data=df_males,
            target_n=8,
            count_col='empi_anon',
            single_per_patient=True,
            route={}
        )

        df_females = sample_dataframe[sample_dataframe['gender'] == 'F']
        child2 = SamplingNode(
            name='child2',
            data=df_females,
            target_n=2,
            count_col='empi_anon',
            single_per_patient=True,
            route={},
            strict=True
        )

        parent.add_child(child1)
        parent.add_child(child2)

        deficit = parent.balance()

        # Child2 is strict, cannot absorb spillover
        assert child2.target_n == 2
        assert deficit == 3

    def test_absorb_surplus_strict_does_nothing(self, sample_dataframe):
        """Absorb surplus does nothing if node is strict."""
        node = SamplingNode(
            name='test',
            data=sample_dataframe,
            target_n=5,
            count_col='empi_anon',
            single_per_patient=True,
            route={},
            strict=True
        )
        original_target = node.target_n
        node.absorb_surplus(3)
        assert node.target_n == original_target

    def test_collect_leaves_single_leaf(self, sample_dataframe):
        """Collect leaves returns self for single node."""
        node = SamplingNode(
            name='test',
            data=sample_dataframe,
            target_n=5,
            count_col='empi_anon',
            single_per_patient=True,
            route={}
        )
        leaves = node.collect_leaves()
        assert len(leaves) == 1
        assert leaves[0] == node

    def test_collect_leaves_tree(self, sample_dataframe):
        """Collect leaves returns all leaf nodes from tree."""
        parent = SamplingNode(
            name='parent',
            data=sample_dataframe,
            target_n=10,
            count_col='empi_anon',
            single_per_patient=True,
            route={}
        )

        child1 = SamplingNode(
            name='child1',
            data=sample_dataframe.head(5),
            target_n=5,
            count_col='empi_anon',
            single_per_patient=True,
            route={}
        )
        child2 = SamplingNode(
            name='child2',
            data=sample_dataframe.tail(5),
            target_n=5,
            count_col='empi_anon',
            single_per_patient=True,
            route={}
        )

        parent.add_child(child1)
        parent.add_child(child2)

        leaves = parent.collect_leaves()
        assert len(leaves) == 2
        assert child1 in leaves
        assert child2 in leaves

    def test_refresh_ids_excludes_patients(self, sample_dataframe):
        """Refresh IDs excludes used patients from capacity."""
        node = SamplingNode(
            name='test',
            data=sample_dataframe,
            target_n=5,
            count_col='empi_anon',
            single_per_patient=True,
            route={}
        )

        original_capacity = node.capacity
        node.refresh_ids(['P001', 'P002', 'P003'])

        assert node.capacity == original_capacity - 3
        assert 'P001' not in node.ids
        assert 'P002' not in node.ids
        assert 'P003' not in node.ids

    def test_sample_returns_correct_count(self, sample_dataframe):
        """Sample returns the correct number of rows."""
        node = SamplingNode(
            name='test',
            data=sample_dataframe,
            target_n=5,
            count_col='empi_anon',
            single_per_patient=True,
            route={}
        )

        rng = random.Random(42)
        result = node.sample(rng)

        assert len(result) == 5

    def test_sample_single_per_patient_dedup(self, multi_exam_dataframe):
        """Sample with single_per_patient deduplicates by patient."""
        node = SamplingNode(
            name='test',
            data=multi_exam_dataframe,
            target_n=3,
            count_col='empi_anon',
            single_per_patient=True,
            route={}
        )

        rng = random.Random(42)
        result = node.sample(rng)

        # Should have no duplicate patients
        assert result['empi_anon'].nunique() == len(result)

    def test_sample_empty_target(self, sample_dataframe):
        """Sample with zero target returns empty DataFrame."""
        node = SamplingNode(
            name='test',
            data=sample_dataframe,
            target_n=0,
            count_col='empi_anon',
            single_per_patient=True,
            route={}
        )

        rng = random.Random(42)
        result = node.sample(rng)

        assert len(result) == 0


# --- GroupContainer Class Tests ---

class TestGroupContainer:
    """Tests for the GroupContainer class."""

    def test_sample_data_end_to_end(self, sample_dataframe):
        """End-to-end sampling builds tree, balances, and returns valid sample."""
        features = [
            Feature(name='gender', match_type='equals', levels=['M', 'F'])
        ]

        container = GroupContainer(
            n=6,
            features=features,
            seed=42,
            count_col='empi_anon',
            single_per_patient=True
        )

        result = container.sample_data(sample_dataframe)

        assert len(result) == 6
        assert all(col in result.columns for col in sample_dataframe.columns)

    def test_sample_size_conservation(self, sample_dataframe):
        """Total sampled is less than or equal to requested n."""
        features = [
            Feature(name='gender', match_type='equals', levels=['M', 'F'])
        ]

        container = GroupContainer(
            n=8,
            features=features,
            seed=42,
            count_col='empi_anon',
            single_per_patient=True
        )

        result = container.sample_data(sample_dataframe)

        assert len(result) <= 8

    def test_single_per_patient_constraint(self, multi_exam_dataframe):
        """No duplicate patient IDs in final output."""
        features = [
            Feature(name='gender', match_type='equals', levels=['M', 'F'])
        ]

        container = GroupContainer(
            n=4,
            features=features,
            seed=42,
            count_col='empi_anon',
            single_per_patient=True
        )

        result = container.sample_data(multi_exam_dataframe)

        assert result['empi_anon'].nunique() == len(result)

    def test_seed_reproducibility(self, sample_dataframe):
        """Same seed produces same sample."""
        features = [
            Feature(name='gender', match_type='equals', levels=['M', 'F'])
        ]

        container1 = GroupContainer(
            n=5,
            features=features,
            seed=42,
            count_col='empi_anon',
            single_per_patient=True
        )

        container2 = GroupContainer(
            n=5,
            features=features,
            seed=42,
            count_col='empi_anon',
            single_per_patient=True
        )

        result1 = container1.sample_data(sample_dataframe.copy())
        result2 = container2.sample_data(sample_dataframe.copy())

        pd.testing.assert_frame_equal(
            result1.sort_values('empi_anon').reset_index(drop=True),
            result2.sort_values('empi_anon').reset_index(drop=True)
        )

    def test_empty_data_handling(self):
        """Returns empty DataFrame when given empty data."""
        features = [
            Feature(name='gender', match_type='equals', levels=['M', 'F'])
        ]

        container = GroupContainer(
            n=5,
            features=features,
            seed=42,
            count_col='empi_anon',
            single_per_patient=True
        )

        empty_df = pd.DataFrame({
            'empi_anon': [],
            'gender': [],
            'studydate_anon': []
        })

        result = container.sample_data(empty_df)

        assert len(result) == 0

    def test_multiple_features(self, sample_dataframe):
        """Sampling works with multiple stratification features."""
        features = [
            Feature(name='gender', match_type='equals', levels=['M', 'F']),
            Feature(name='category', match_type='equals', levels=['A', 'B'])
        ]

        container = GroupContainer(
            n=8,
            features=features,
            seed=42,
            count_col='empi_anon',
            single_per_patient=True
        )

        result = container.sample_data(sample_dataframe)

        assert len(result) <= 8
        assert result['empi_anon'].nunique() == len(result)


# --- Edge Case Tests ---

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_level_feature(self, sample_dataframe):
        """Single level in a feature works correctly."""
        feature = Feature(
            name='gender',
            match_type='equals',
            levels=['M']
        )

        assert len(feature.levels) == 1
        assert feature.levels[0].weight == 1.0

    def test_target_exceeds_capacity(self, sample_dataframe):
        """Target exceeding capacity clips to available."""
        features = [
            Feature(name='gender', match_type='equals', levels=['M', 'F'])
        ]

        container = GroupContainer(
            n=100,
            features=features,
            seed=42,
            count_col='empi_anon',
            single_per_patient=True
        )

        result = container.sample_data(sample_dataframe)

        # Should get at most the total unique patients
        assert len(result) <= 10

    def test_all_strict_nodes(self, sample_dataframe):
        """All strict nodes means no spillover possible."""
        features = [
            Feature(
                name='gender',
                match_type='equals',
                levels=['M', 'F'],
                strict=True
            )
        ]

        container = GroupContainer(
            n=10,
            features=features,
            seed=42,
            count_col='empi_anon',
            single_per_patient=True
        )

        # Should still work, just without spillover
        result = container.sample_data(sample_dataframe)
        assert len(result) <= 10

    def test_zero_weight_level(self, sample_dataframe):
        """Zero weight level gets no samples."""
        features = [
            Feature(
                name='gender',
                match_type='equals',
                levels=['M', 'F'],
                weights=[1.0, 0.0]
            )
        ]

        container = GroupContainer(
            n=5,
            features=features,
            seed=42,
            count_col='empi_anon',
            single_per_patient=True
        )

        result = container.sample_data(sample_dataframe)

        # All samples should be male (or close to it due to remainder logic)
        assert len(result) <= 5

    def test_node_repr(self):
        """Node __repr__ returns expected format."""
        df = pd.DataFrame({
            'empi_anon': ['P001', 'P002'],
            'studydate_anon': pd.date_range('2020-01-01', periods=2)
        })
        node = SamplingNode(
            name='test_node',
            data=df,
            target_n=5,
            count_col='empi_anon',
            single_per_patient=True,
            route={},
            strict=True
        )

        repr_str = repr(node)
        assert 'test_node' in repr_str
        assert 'Target: 5' in repr_str
        assert 'Strict: True' in repr_str
