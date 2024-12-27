### ops.py

***** Imports *****
    - __future__
        - annotations
    - typing
        - Any, Optional, Union, Callable, cast, TYPE_CHECKING, Type, Literal, get_args
    - sys, time, functools, itertools, math, operator, hashlib, os, types, pickle, pathlib, inspect, weakref
    - enum
        - auto
        - IntEnum
        - Enum
    - dataclasses
        - dataclass, field
    - collections 
        - defaultdict
    - tinygrad.helpers
        - ContextVar, all_int, prod, getenv, all_same, Context, partition, temp, unwrap, T, argfix, Metadata, _METADATA, flatten
        - PICKLE_BUFFERS, SPLIT_REDUCEOP, DEBUG
    - tinygrad.dtype
        - ConstType, ImageDType, PtrDType, dtypes, DType, truncate
    - tinygrad.shape.shapetracker
        - ShapeTracker
    - tinygrad.device
        - Buffer


***** Classes and Functions *****
    - class FastEnum(IntEnum)
    - class SimpleMathTrait
    - class MathTrait(SimpleMathTrait)
    - class Ops(FastEnum)
    - class GroupOP
    - const view_supported_devices
    - def identity_element(op, dt)
    - def can_pad(u, edges, visited)
    - def resolve(x, default)
    - def _suop(lst, uop_fxn, python_fxn)
    - def smax(*lst)
    - def smin(*lst)
    - def ssimplify(uop)
    - def sym_infer(uop, var_vals)
    - def pretty_print(x, rep, srcfn, cache, d)
    - class UOpMetaClass(type)
    - const buffers
    - const all_metadata
    - const forced_realize
    - class UOp(MathTrait, metaclass)
        - const op
        - const dtype
        - const src
        - const arg
        - ** constructors and deconstructors **
        - def toposort
        - def tuplize
        **** uop shape stuff ****
        - def has_st
        - def st
        - def full_shape
        - def shape
        - def size
        **** uop evaluation ****
        - def simplify
        - def ssimplify
        - def _eval
        - def substitute(dvars)
        **** uop syntactic sugar ****
        - def st_arg
        - def const_arg
        - def axis_arg
        - def sink(*srcs)
        - def detach
        - def index(idx, valid)
        - def const_like(b)
        - def broadcase(count)
        - def cast(dtype, bitcast, allow_buffer_view)
        - def bitcast(dtype, allow_buffer_view)
        - def gep(i)
        - def load(*src, **kwargs)
        - def store(*src, **kwargs)
        - def alu(arg, *src)
        - def const(dtype, b)
        - def range(dtype, start, end, idx)
        - def _reduce_op(op, axis)
        - def r(op, axis)
        - def assign(x)
        - def contiguous(allow_buffer_view)
        **** from LazyBuffer **** 
        - def const_with_shape(dtype, val, shape)
        - def metaop(op, shape, dtype, device, arg, src)
        - def copy_to_device(device, force, clone)
        - def clone
        - def is_unrealized_const
        - def can_view
        - def lbs
        - def metadata
        - def forced_realize
        - def become(u)
        **** uop movement ops ****
        - def base
        - def view(new_st)
        - def _mop(op, arg)
        - def reshape(arg)
        - def expand(arg)
        - def permute(arg)
        - def shrink(arg)
        - def stride(arg)
        **** uop Buffer stuff ****
        - const buffer_num
        - def new_buffer(device, size, dtype)
        - def device
        - def buf_uop
        - def buf_uop_view
        - def buffer
        - def realized
        - def is_realized
        **** uop Variable stuff ****
        - def variable(name, min_val, max_vcal, dtype)
        - def expr
        - def bind(val)
        - def unbind
        - def val
        - def vars
        - def variables
        **** uop symbolic stuff ****
        - def const_factor
        - def divides(v)
        - def vmin
        - def vmax
        - def _min_max
        - def _sym_fxn
        - def sym_infer(var_vals)
        - def render(simplify)
    - class KernelInfo
    **** ops in python ****
    - def safe_exp2(x)
    - const python_alu
    - def exec_alu(op, dtype, operands, truncate_output)
    **** uop helpers ****
    - def print_uops
    **** pattern matcher ****
    - def get_location
    - def lines(fn)
    - class UPat(MathTrait)
    - class UPatAny(UPat)
    - def deconstruct_function
    - class PatternMatcher
    **** Tracking pattern matcher ****
    - const TRCAK_MATCH_STATS
    - const match_stats
    - class TrackedGraphRewrite
    - class TrackedPatternMatcher(PatternMatcher)
    - def launch_viz
    **** simple graph rewrite engine ***
    - class RewriteContext
    **** uop type spec ****
    - const spec
    - def type_verify(uops)
    **** most of the symbolic lives here now ****
    - def split_uop(x, sep)
    - def div_and_mod_folding(x, c, which, split_rem)
    - def lt_folding(x, c)
    - def fold_unrolled_divs(divs)
    - def canonicalize_simplex(X)
    - def is_increasing(f)
    - def parse_valid(valid)
    - def uop_given_valid(valid, uop)
    - def _valid_priority(v, valids)
    - def simplify_valid(valid)
    - def sint_to_uop(x, dtype)
    - const symbolic_simple
    - const symbolic
    - const symbolic_flat
    - const _substitute
    - const syms
    - const renderer
    **** What was symbolic.py ****
    - const sint
    - const Variable
    - const ConstLike
    **** uop swizzling ****
    - const merge_views
    - const view_left