# Data

There are no data files in this folder, and that is deliberate.

The experiments use the Wisconsin Diagnostic Breast Cancer (WDBC) dataset:
569 samples, 30 real-valued features computed from digitized fine-needle
aspirate images, binary diagnosis label (malignant / benign). It is a public
dataset from the UCI Machine Learning Repository and it ships bundled with
scikit-learn, so the experiment script loads it directly:

```python
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
```

No download step, no credentials, no PHI. Committing a copy here would only
add a stale duplicate of something scikit-learn already versions properly.

If you want the raw CSV for inspection anyway, this one-liner writes it:

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
d = load_breast_cancer(as_frame=True)
pd.concat([d.data, d.target.rename('diagnosis')], axis=1).to_csv('data/wdbc.csv', index=False)
```

Preprocessing applied by the experiment script: stratified 75/25 train/test
split (fixed seed 42), z-score standardization fitted on the training split
only, and a Dirichlet(α = 0.5) label-skew partition of the training set
across 20 simulated hospital clients.

Citation: Dua, D., & Graff, C. (2019). UCI Machine Learning Repository.
University of California, Irvine. http://archive.ics.uci.edu/ml
