import numpy as np
import pytest

import numrus as rnp


# Helper function to generate random 1D vectors
def setup_vector(dtype, size=1000):
    a = np.random.rand(size).astype(dtype)
    return a.tolist()


# Helper function to generate multiple 1D arrays for concatenate
def setup_concatenate_arrays_1d(dtype, num_arrays=2, size=1000):
    return [np.random.rand(size).astype(dtype).tolist() for _ in range(num_arrays)]


# Helper function to generate multiple 2D arrays for concatenate
def setup_concatenate_arrays_2d(dtype, num_arrays=2, rows=100, cols=100):
    return [
        np.random.rand(rows, cols).astype(dtype).tolist() for _ in range(num_arrays)
    ]


# Function to perform concatenate using numrus
def concatenate_numrus(arrays, axis, dtype):
    rnp_arrays = [rnp.NumArray(arr, dtype=dtype) for arr in arrays]
    return rnp.concatenate(rnp_arrays, axis)


# Function to perform concatenate using numpy
def concatenate_numpy(arrays, axis, dtype):
    np_arrays = [np.array(arr, dtype=dtype) for arr in arrays]
    return np.concatenate(np_arrays, axis)


# Function to perform exp using numrus
def exp_numrus(a, dtype):
    a_rnp = rnp.NumArray(a, dtype=dtype)
    return a_rnp.exp()


# Function to perform exp using numpy
def exp_numpy(a, dtype):
    a_np = np.array(a, dtype=dtype)
    return np.exp(a_np)


# Function to perform log using numrus
def log_numrus(a, dtype):
    a_rnp = rnp.NumArray(a, dtype=dtype)
    return a_rnp.log()


# Function to perform log using numpy
def log_numpy(a, dtype):
    a_np = np.array(a, dtype=dtype)
    return np.log(a_np)


# Function to perform sigmoid using numrus
def sigmoid_numrus(a, dtype):
    a_rnp = rnp.NumArray(a, dtype=dtype)
    return a_rnp.sigmoid()


# Function to perform sigmoid using numpy
def sigmoid_numpy(a, dtype):
    a_np = np.array(a, dtype=dtype)
    return 1 / (1 + np.exp(-a_np))


# -----------------------------
# Concatenate Benchmarks
# -----------------------------


# Benchmark for Concatenate using RustyNum with 2D arrays
@pytest.mark.parametrize(
    "dtype,size,axis",
    [
        ("float32", (100, 100), 0),
        ("float32", (100, 100), 1),
        ("float64", (100, 100), 0),
        ("float64", (100, 100), 1),
    ],
)
def test_concatenate_numrus_2d(benchmark, dtype, size, axis):
    rows, cols = size
    arrays = setup_concatenate_arrays_2d(dtype, num_arrays=2, rows=rows, cols=cols)

    def concat_numrus():
        concatenate_numrus(arrays, axis, dtype)

    benchmark(concat_numrus)


# Benchmark for Concatenate using NumPy with 2D arrays
@pytest.mark.parametrize(
    "dtype,size,axis",
    [
        ("float32", (100, 100), 0),
        ("float32", (100, 100), 1),
        ("float64", (100, 100), 0),
        ("float64", (100, 100), 1),
    ],
)
def test_concatenate_numpy_2d(benchmark, dtype, size, axis):
    rows, cols = size
    arrays = setup_concatenate_arrays_2d(dtype, num_arrays=2, rows=rows, cols=cols)

    def concat_numpy_func():
        concatenate_numpy(arrays, axis, dtype)

    benchmark(concat_numpy_func)


# -----------------------------
# Exp Benchmarks
# -----------------------------


# Benchmark for Exp using RustyNum
@pytest.mark.parametrize("dtype,size", [("float32", 1000), ("float64", 10000)])
def test_exp_numrus(benchmark, dtype, size):
    a = setup_vector(dtype, size)

    def exp_rnp():
        exp_numrus(a, dtype)

    benchmark(exp_rnp)


# Benchmark for Exp using NumPy
@pytest.mark.parametrize("dtype,size", [("float32", 1000), ("float64", 10000)])
def test_exp_numpy(benchmark, dtype, size):
    a = setup_vector(dtype, size)

    def exp_np_func():
        exp_numpy(a, dtype)

    benchmark(exp_np_func)


# -----------------------------
# Log Benchmarks
# -----------------------------


# Benchmark for Log using RustyNum
@pytest.mark.parametrize("dtype,size", [("float32", 1000), ("float64", 10000)])
def test_log_numrus(benchmark, dtype, size):
    # Ensure all elements are positive to avoid log domain errors
    a = np.random.rand(size).astype(dtype) + 1.0  # Shift to ensure positivity
    a_list = a.tolist()

    def log_rnp():
        log_numrus(a_list, dtype)

    benchmark(log_rnp)


# Benchmark for Log using NumPy
@pytest.mark.parametrize("dtype,size", [("float32", 1000), ("float64", 10000)])
def test_log_numpy(benchmark, dtype, size):
    a = np.random.rand(size).astype(dtype) + 1.0  # Shift to ensure positivity

    def log_np_func():
        log_numpy(a, dtype)

    benchmark(log_np_func)


# -----------------------------
# Sigmoid Benchmarks
# -----------------------------


# Benchmark for Sigmoid using RustyNum
@pytest.mark.parametrize("dtype,size", [("float32", 1000), ("float64", 10000)])
def test_sigmoid_numrus(benchmark, dtype, size):
    a = setup_vector(dtype, size)

    def sigmoid_rnp():
        sigmoid_numrus(a, dtype)

    benchmark(sigmoid_rnp)


# Benchmark for Sigmoid using NumPy
@pytest.mark.parametrize("dtype,size", [("float32", 1000), ("float64", 10000)])
def test_sigmoid_numpy(benchmark, dtype, size):
    a = setup_vector(dtype, size)

    def sigmoid_np_func():
        sigmoid_numpy(a, dtype)

    benchmark(sigmoid_np_func)
