### multi.py

***** Imports *****
    - __future__
        - annotations
    - functools, itertools, operator
    - tinygrad.helpers
        - all_same, all_int, dedup, prod, DEBUG, RING, getenv
    - tinygrad.dtype
        - DType
    - tinygrad.ops
        - Ops, MathTrait, Uop, sint


***** Classes and Functions *****
    - def all_reduce(bop, lbs)
    - def to_sharded(lbs, axis, bounds)
    - class MultiLazyBuffer(MathTrait)
        - def shape()
        - def size()
        - def real_lbs()
        - def __repr__()
        - def from_sharded(lb, devices, axis, bounds)
        - def copy_to_device(self, device)
        ------ Passthroughs ------
        - def is_realized()
        - def cast(dtype, bitcast, allow_buffer_view)
        - def const_like(b)
        - def assign(x)
        - def contiguous()
        - def clone()
        - def detach()
        ------ Elementwise is simple ------
        - def alu(op, *in_srcs)
        - def r(op, axis)
        ------ Movement ops ------
        - def _shape_to_single_shard(shape, lb)
        - def reshape(arg)
        - def pad(arg)
        - def expand(arg)
        - def permute(arg)
        - def shrink(arg)
        - def stride(arg)
