# bea-tools

**🐝 𝓉𝑜𝑜𝓁𝓈 𝓂𝒶𝒹𝑒 𝒷𝓎, 𝒶𝓃𝒹 𝒻𝑜𝓇, 𝒷𝑒𝒶 🐝**

A Python package of random functions and tools that I use regularly. Data science/analysis focused since I'm a data scientist c:

## Installation

```bash
pip install bea-tools
```

## Features

### Utility Functions

Helper functions for formatting and displaying output in a clean, readable way.

#### divider()

Creates formatted divider lines with optional text alignment.

```python
from bea_tools.utility import divider

# Simple divider
print(divider(line_width=50))  # --------------------...

# Divider with centered text
print(divider("Section Header", "=", line_width=50, align="center"))
# ============ Section Header ============

# Divider with left-aligned text
print(divider("Results", "-", align="left"))
# Results -------------------------
```

#### aligned()

Formats items within a frame, which is then aligned within a line. Perfect for creating clean output displays.

```python
from bea_tools.utility import aligned

# Center two items within a frame
print(aligned("Label:", "Value", frame_width=30, line_width=50))

# Multiple items distributed across frame
print(aligned("A", "B", "C", frame_width=40))
```

### Pandas Extensions

#### Series.bea.value_counts()

Enhanced value counts with custom sorting, normalization, and formatted string output.

```python
import pandas as pd

df = pd.DataFrame({'category': ['A', 'B', 'A', 'C', 'B', 'A']})

# Get counts with proportions as formatted strings
counts = df['category'].bea.value_counts(with_proportion=True, output=True)
# Returns: {'A': '3 (50.0%)', 'B': '2 (33.3%)', 'C': '1 (16.7%)'}

# Custom sort order
counts = df['category'].bea.value_counts(sort=['A', 'B', 'C'], output=True)

# Display directly (Jupyter-friendly)
df['category'].bea.value_counts(with_proportion=True)
```

### TreeSampler

A hierarchical stratified sampling tool for pandas DataFrames. Designed for scenarios where you need to sample data while maintaining specific proportions across multiple categorical dimensions, with intelligent handling of capacity constraints.

**Key capabilities:**

- **Multi-dimensional stratification**: Define sampling targets across multiple features (e.g., gender, age group, category)
- **Hierarchical spillover**: When a stratum lacks sufficient data, excess quota automatically redistributes to sibling strata
- **Flexible matching**: Match values using `equals`, `contains`, or `between` strategies
- **Conditional weights**: Define weights that vary based on the path through the sampling tree
- **Strict mode**: Lock specific strata to prevent them from absorbing spillover
- **Balanced sampling**: Equal distribution across levels regardless of population proportions
- **Single-per-entity sampling**: Ensure unique entities (e.g., one exam per patient) with optional sorting control

### DicomComparer

A tool for comparing two DICOM files at the attribute level. Identifies which attributes are shared between files, which are exclusive to each, and whether shared attributes have matching or conflicting values.

**Key capabilities:**

- **Attribute overlap analysis**: Identify which DICOM tags exist in both files vs. exclusive to one
- **Value comparison**: For shared attributes, detect matches and conflicts
- **Flexible input**: Pass files directly or as a labeled dictionary
- **Summary output**: Generate formatted comparison reports

```python
import pydicom
from bea_tools._dicom.dicomp import DicomComparer

dcm1 = pydicom.dcmread("path/to/first.dcm")
dcm2 = pydicom.dcmread("path/to/second.dcm")

comparer = DicomComparer(dcm1, dcm2)
comparison = comparer.compare()

# Print summary statistics
comparison.summary()

# Access specific conflicts
for conflict in comparison.intersection.comparison.conflicts:
    print(conflict)
```

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

### Balanced Sampling

Ensure equal representation across levels, ignoring the underlying population distribution:

```python
# This feature will have exactly equal samples from each level
feature = Feature(
    name='modality',
    match_type='equals',
    levels=['CT', 'MRI', 'Xray', 'US'],
    balanced=True  # Distributes samples equally across all 4 levels
)
```

### Optional Sorting for Single-Per-Patient

Control whether to use a sort column when selecting one row per patient:

```python
# With sorting (e.g., earliest study date per patient)
sampler = TreeSampler(
    n=100,
    features=features,
    sort_col='studydate_anon',  # Default
    single_per_patient=True
)

# Without sorting (arbitrary selection, faster)
sampler = TreeSampler(
    n=100,
    features=features,
    sort_col=None,  # Disables sorting
    single_per_patient=True
)
```

## Requirements

- Python 3.10+
- pandas >= 2.2
- pydicom (for DICOM utilities)

## License

MIT
