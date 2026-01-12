# bea-tools

**Tools made by, and for, Bea**

A Python package of random functions and tools that I use regularly. Data science/analysis focused since, ya know, I'm a data scientist c:

## Installation

```bash
pip install bea-tools
```

## Features

### TreeSampler

A hierarchical stratified sampling tool for pandas DataFrames. Designed for scenarios where you need to sample data while maintaining specific proportions across multiple categorical dimensions, with intelligent handling of capacity constraints.

**Key capabilities:**

- **Multi-dimensional stratification**: Define sampling targets across multiple features (e.g., gender, age group, category)
- **Hierarchical spillover**: When a stratum lacks sufficient data, excess quota automatically redistributes to sibling strata
- **Flexible matching**: Match values using `equals`, `contains`, or `between` strategies
- **Conditional weights**: Define weights that vary based on the path through the sampling tree
- **Strict mode**: Lock specific strata to prevent them from absorbing spillover
- **Single-per-entity sampling**: Ensure unique entities (e.g., one exam per patient)

## Quick Start

```python
from bea_tools import TreeSampler
from bea_tools._pandas.sampler import Feature

import pandas as pd

# Sample data
df = pd.DataFrame({
    'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006'],
    'gender': ['M', 'M', 'F', 'F', 'M', 'F'],
    'age': [25, 45, 35, 55, 30, 40],
    'studydate_anon': pd.date_range('2020-01-01', periods=6)
})

# Define stratification features
features = [
    Feature(
        name='gender',
        match_type='equals',
        levels=['M', 'F'],
        weights=[0.5, 0.5]  # Target 50/50 split
    )
]

# Create sampler and extract stratified sample
sampler = TreeSampler(
    n=4,                          # Target sample size
    features=features,
    seed=42,                      # For reproducibility
    count_col='patient_id',       # Column for unique entity identification
    single_per_patient=True       # One row per patient
)

result = sampler.sample_data(df)
```

## Advanced Usage

### Age Brackets with Between Matching

```python
age_feature = Feature(
    name='age',
    match_type='between',
    levels=[(0, 30), (30, 50), (50, 100)],
    weights=[0.3, 0.4, 0.3],
    labels=['Young', 'Middle', 'Senior'],
    label_col='age_group'
)
```

### Strict Strata (No Spillover)

```python
# This stratum will maintain exact proportions, never absorbing excess
feature = Feature(
    name='category',
    match_type='equals',
    levels=['A', 'B'],
    weights=[0.7, 0.3],
    strict=True  # Prevents spillover absorption
)
```

### Conditional Weights

Define weights that depend on parent feature values:

```python
category_feature = Feature(
    name='category',
    match_type='equals',
    levels=['X', 'Y'],
    conditional_weights=[{
        'feature': 'gender',
        'weights': {
            'M': [0.6, 0.4],  # When gender=M: 60% X, 40% Y
            'F': [0.4, 0.6]   # When gender=F: 40% X, 60% Y
        }
    }]
)
```

## Requirements

- Python 3.10+
- pandas >= 2.2

## License

MIT
