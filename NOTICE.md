# Source and dependency attribution

ARRIVE code and documentation in this repository are distributed under the
existing MIT license in `LICENSE`. Its copyright and permission notice are
retained. No vendored third-party implementation is included.

The package uses installed dependencies under their own licenses: Python
(PSF License), NumPy (BSD-3-Clause), SciPy (BSD-3-Clause), and CatBoost
(Apache-2.0). Their implementations are not copied into this ZIP.
Their license texts remain with their distributions. Official projects:
[Python](https://www.python.org/), [NumPy](https://numpy.org/),
[SciPy](https://scipy.org/), [CatBoost](https://catboost.ai/).

The retrieval design uses reciprocal rank fusion; ranking uses CatBoost's
metric-specific YetiRank. Bibliographic attribution remains in the accompanying
manuscript. Dataset ownership and access conditions remain with the providers.
