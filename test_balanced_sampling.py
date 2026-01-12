"""Test suite for the balanced sampling feature in TreeSampler.

Tests verify that when balanced=True is set on a Feature:
1. Level weights/counts are disregarded
2. Levels are equally distributed across groups
3. Only listed feature levels are included (others excluded)
4. The feature works correctly with single and multiple features
5. The feature interacts correctly with other features
"""

import pandas as pd
import numpy as np
from bea_tools._pandas.sampler import TreeSampler, Feature


def create_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create synthetic test data for sampling experiments.

    Creates a DataFrame with:
    - empi_anon: Patient IDs
    - studydate_anon: Study dates
    - gender: Binary gender (M/F)
    - age_group: Three age groups (Young/Middle/Old)
    - condition: Three conditions (A/B/C)
    """
    np.random.seed(42)

    data = pd.DataFrame({
        'empi_anon': [f'P{i:04d}' for i in range(n_samples)],
        'studydate_anon': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        'gender': np.random.choice(['M', 'F'], n_samples, p=[0.6, 0.4]),
        'age_group': np.random.choice(['Young', 'Middle', 'Old'], n_samples, p=[0.5, 0.3, 0.2]),
        'condition': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.5, 0.3, 0.2])
    })

    return data


def test_balanced_single_feature():
    """Test balanced=True with a single feature.

    Expected: Equal distribution across all levels, regardless of weights.
    """
    print("\n=== Test 1: Balanced Single Feature ===")

    data = create_test_data(1000)

    # Create sampler with balanced gender feature
    # Even though data has 60% M, 40% F, balanced should give 50/50
    sampler = TreeSampler(
        n=100,
        features=[
            Feature(
                name='gender',
                match_type='equals',
                levels=['M', 'F'],
                balanced=True
            )
        ],
        seed=42
    )

    result = sampler.sample_data(data)

    # Check total sample size
    assert len(result) == 100, f"Expected 100 samples, got {len(result)}"

    # Check gender distribution is balanced (50/50)
    gender_counts = result['gender'].value_counts()
    print(f"Gender distribution: {gender_counts.to_dict()}")

    # Should be 50 each (or 49/51 due to odd numbers)
    assert abs(gender_counts['M'] - 50) <= 1, f"Expected ~50 males, got {gender_counts['M']}"
    assert abs(gender_counts['F'] - 50) <= 1, f"Expected ~50 females, got {gender_counts['F']}"

    print("✓ Test 1 passed: Gender is balanced 50/50")
    return True


def test_balanced_vs_weighted():
    """Test that balanced=False respects weights while balanced=True ignores them.

    Expected: With balanced=False, weights are respected. With balanced=True, they're ignored.
    """
    print("\n=== Test 2: Balanced vs Weighted ===")

    data = create_test_data(1000)

    # Test with weights (balanced=False)
    sampler_weighted = TreeSampler(
        n=100,
        features=[
            Feature(
                name='age_group',
                match_type='equals',
                levels=['Young', 'Middle', 'Old'],
                weights=[0.6, 0.3, 0.1],  # 60%, 30%, 10%
                balanced=False
            )
        ],
        seed=42
    )

    result_weighted = sampler_weighted.sample_data(data)
    age_counts_weighted = result_weighted['age_group'].value_counts()
    print(f"Weighted age distribution: {age_counts_weighted.to_dict()}")

    # Test with balanced
    sampler_balanced = TreeSampler(
        n=99,  # Use 99 for exact 33/33/33 split
        features=[
            Feature(
                name='age_group',
                match_type='equals',
                levels=['Young', 'Middle', 'Old'],
                weights=[0.6, 0.3, 0.1],  # These should be ignored
                balanced=True
            )
        ],
        seed=42
    )

    result_balanced = sampler_balanced.sample_data(data)
    age_counts_balanced = result_balanced['age_group'].value_counts()
    print(f"Balanced age distribution: {age_counts_balanced.to_dict()}")

    # Weighted should be close to 60/30/10
    assert age_counts_weighted['Young'] >= 50, "Weighted should favor Young"

    # Balanced should be equal (33 each for 99 samples)
    balanced_values = list(age_counts_balanced.values)
    assert all(count == 33 for count in balanced_values), \
        f"Expected all counts to be 33, got {age_counts_balanced.to_dict()}"

    print("✓ Test 2 passed: Balanced ignores weights, weighted respects them")
    return True


def test_balanced_multiple_features():
    """Test balanced feature with multiple features in the hierarchy.

    Expected: The balanced feature distributes equally at its level,
              while other features use their weights.
    """
    print("\n=== Test 3: Balanced with Multiple Features ===")

    data = create_test_data(1000)

    # Gender is balanced, age_group uses weights
    sampler = TreeSampler(
        n=120,  # 2 genders × 3 ages × 20 per group
        features=[
            Feature(
                name='gender',
                match_type='equals',
                levels=['M', 'F'],
                balanced=True  # Equal split
            ),
            Feature(
                name='age_group',
                match_type='equals',
                levels=['Young', 'Middle', 'Old'],
                weights=[0.5, 0.3, 0.2],  # Different weights per age
                balanced=False
            )
        ],
        seed=42
    )

    result = sampler.sample_data(data)

    # Check gender is balanced
    gender_counts = result['gender'].value_counts()
    print(f"Gender distribution: {gender_counts.to_dict()}")
    assert abs(gender_counts['M'] - 60) <= 2, f"Expected ~60 males, got {gender_counts['M']}"
    assert abs(gender_counts['F'] - 60) <= 2, f"Expected ~60 females, got {gender_counts['F']}"

    # Check age groups follow weights within each gender
    for gender in ['M', 'F']:
        subset = result[result['gender'] == gender]
        age_counts = subset['age_group'].value_counts()
        print(f"{gender} age distribution: {age_counts.to_dict()}")

        # Should be approximately 50%, 30%, 20% of 60 = 30, 18, 12
        assert age_counts['Young'] > age_counts['Middle'] > age_counts['Old'], \
            f"Age groups should follow weight ordering for {gender}"

    print("✓ Test 3 passed: Balanced feature works with weighted features")
    return True


def test_balanced_three_features():
    """Test balanced with three features to verify complex hierarchies.

    Expected: Balanced feature ensures equal distribution at its level,
              regardless of position in hierarchy.
    """
    print("\n=== Test 4: Balanced in Three-Feature Hierarchy ===")

    data = create_test_data(1000)

    # Test with balanced in the middle
    sampler = TreeSampler(
        n=120,
        features=[
            Feature(
                name='gender',
                match_type='equals',
                levels=['M', 'F'],
                balanced=False,
                weights=[0.6, 0.4]
            ),
            Feature(
                name='age_group',
                match_type='equals',
                levels=['Young', 'Middle', 'Old'],
                balanced=True  # Balanced in the middle
            ),
            Feature(
                name='condition',
                match_type='equals',
                levels=['A', 'B'],
                balanced=False,
                weights=[0.7, 0.3]
            )
        ],
        seed=42
    )

    result = sampler.sample_data(data)

    # Within each gender, age groups should be balanced
    for gender in ['M', 'F']:
        gender_subset = result[result['gender'] == gender]
        age_counts = gender_subset['age_group'].value_counts()
        print(f"{gender} age distribution: {age_counts.to_dict()}")

        # Check that ages are roughly equal
        age_values = list(age_counts.values)
        max_diff = max(age_values) - min(age_values)
        assert max_diff <= 2, f"Age groups should be balanced for {gender}, max diff: {max_diff}"

    print("✓ Test 4 passed: Balanced works in middle of hierarchy")
    return True


def test_balanced_excludes_unlisted_levels():
    """Test that balanced feature only includes listed levels.

    Expected: Only the specified levels are sampled, others are excluded.
    """
    print("\n=== Test 5: Balanced Excludes Unlisted Levels ===")

    data = create_test_data(1000)

    # Only sample Young and Middle, exclude Old
    sampler = TreeSampler(
        n=100,
        features=[
            Feature(
                name='age_group',
                match_type='equals',
                levels=['Young', 'Middle'],  # Intentionally exclude 'Old'
                balanced=True
            )
        ],
        seed=42
    )

    result = sampler.sample_data(data)

    # Check that Old is not in results
    age_counts = result['age_group'].value_counts()
    print(f"Age distribution: {age_counts.to_dict()}")

    assert 'Old' not in age_counts.index, "Old should be excluded from results"
    assert len(age_counts) == 2, "Should only have 2 age groups"
    assert abs(age_counts['Young'] - 50) <= 1, "Young should be ~50"
    assert abs(age_counts['Middle'] - 50) <= 1, "Middle should be ~50"

    print("✓ Test 5 passed: Unlisted levels are excluded")
    return True


def test_balanced_with_capacity_constraints():
    """Test balanced feature when some levels have limited capacity.

    Expected: Spillover mechanism should maintain balance as much as possible,
              but respect capacity constraints. When one group is capacity-limited,
              spillover distributes the deficit to other groups.
    """
    print("\n=== Test 6: Balanced with Capacity Constraints ===")

    # Create imbalanced data: very few 'Old' samples
    data = create_test_data(1000)
    # Artificially reduce Old samples to test capacity constraints
    data = data[~((data['age_group'] == 'Old') & (data.index > 220))]

    print(f"Data age distribution: {data['age_group'].value_counts().to_dict()}")

    sampler = TreeSampler(
        n=150,  # Request 150, should be 50 each, but Old has limited capacity
        features=[
            Feature(
                name='age_group',
                match_type='equals',
                levels=['Young', 'Middle', 'Old'],
                balanced=True
            )
        ],
        seed=42
    )

    result = sampler.sample_data(data)
    age_counts = result['age_group'].value_counts()
    print(f"Result age distribution: {age_counts.to_dict()}")

    # With spillover, the system should achieve close to balanced distribution
    # by redistributing from the constrained group
    # The total should still be 150
    assert len(result) == 150, f"Expected 150 total samples, got {len(result)}"

    # All groups should have reasonable representation
    # (spillover mechanism helps balance even with constraints)
    assert age_counts['Old'] >= 40, "Old should get reasonable representation"
    assert age_counts['Young'] >= 40, "Young should get reasonable representation"
    assert age_counts['Middle'] >= 40, "Middle should get reasonable representation"

    print("✓ Test 6 passed: Balanced handles capacity constraints with spillover")
    return True


def test_balanced_counts_ignored():
    """Test that balanced=True ignores explicit counts parameter.

    Expected: Counts should be ignored when balanced=True.
    """
    print("\n=== Test 7: Balanced Ignores Counts ===")

    data = create_test_data(1000)

    sampler = TreeSampler(
        n=100,
        features=[
            Feature(
                name='gender',
                match_type='equals',
                levels=['M', 'F'],
                counts=[70, 30],  # These should be ignored
                balanced=True
            )
        ],
        seed=42
    )

    result = sampler.sample_data(data)
    gender_counts = result['gender'].value_counts()
    print(f"Gender distribution: {gender_counts.to_dict()}")

    # Should be 50/50, not 70/30
    assert abs(gender_counts['M'] - 50) <= 1, f"Expected ~50 males, got {gender_counts['M']}"
    assert abs(gender_counts['F'] - 50) <= 1, f"Expected ~50 females, got {gender_counts['F']}"

    print("✓ Test 7 passed: Balanced ignores counts")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("Testing TreeSampler Balanced Feature")
    print("=" * 60)

    tests = [
        test_balanced_single_feature,
        test_balanced_vs_weighted,
        test_balanced_multiple_features,
        test_balanced_three_features,
        test_balanced_excludes_unlisted_levels,
        test_balanced_with_capacity_constraints,
        test_balanced_counts_ignored,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"✗ {test.__name__} error: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
