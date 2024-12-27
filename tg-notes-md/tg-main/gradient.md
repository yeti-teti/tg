### gradient.py

***** Imports *****
    - typing
        - cast
    - math, functools
    - tinygrad.dtype
        - dtypes, sum_acc_dtype
    - tinygrad.ops
        - UOp, PatternMatcher, UPat, Ops
    - tinygrad.helpers
        - argsort


***** Classes and Functions *****
    - def reduce_gradient
    - var pm_gradient
    - def _deepwalk
    - def compute_gradient