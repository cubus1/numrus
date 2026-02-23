# bindings/python/numrus/__init__.py

from .functions import (
    arange,
    concatenate,
    dot,
    exp,
    linspace,
    log,
    max,
    mean,
    median,
    min,
    norm,
    ones,
    sigmoid,
    zeros,
    # HDC/VSA
    hamming_distance,
    hamming_batch,
    hamming_top_k,
    bundle_u8,
    # INT8 GEMM
    quantize_f32_to_u8,
    quantize_f32_to_i8,
    int8_gemm_i32,
    int8_gemm_f32,
)
from .num_array_class import NumArray
from ._numrus import PyCogRecord as CogRecord

__all__ = [
    "NumArray",
    "CogRecord",
    # existing
    "zeros",
    "ones",
    "arange",
    "linspace",
    "mean",
    "median",
    "min",
    "max",
    "dot",
    "exp",
    "log",
    "sigmoid",
    "concatenate",
    "norm",
    # HDC/VSA
    "hamming_distance",
    "hamming_batch",
    "hamming_top_k",
    "bundle_u8",
    # INT8 GEMM
    "quantize_f32_to_u8",
    "quantize_f32_to_i8",
    "int8_gemm_i32",
    "int8_gemm_f32",
]
