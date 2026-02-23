//! Raw FFI declarations for Intel MKL.
//!
//! These map 1:1 to the C headers: mkl_cblas.h, mkl_lapacke.h, mkl_vml.h, mkl_dfti.h.
//! All functions are unsafe — safe wrappers live in the call-site dispatch blocks.
//!
//! Convention: Rust enums Layout, Transpose, Uplo, Side, Diag are #[repr(u32)]
//! with CBLAS values (101, 102, 111, ...) so they cast directly to c_int.

#![allow(non_snake_case)]

use std::os::raw::{c_double, c_float, c_int, c_long, c_void};

// ═══════════════════════════════════════════════════════════════
// CBLAS Level 1
// ═══════════════════════════════════════════════════════════════

extern "C" {
    pub fn cblas_sdot(
        n: c_int,
        x: *const c_float,
        incx: c_int,
        y: *const c_float,
        incy: c_int,
    ) -> c_float;
    pub fn cblas_ddot(
        n: c_int,
        x: *const c_double,
        incx: c_int,
        y: *const c_double,
        incy: c_int,
    ) -> c_double;

    pub fn cblas_saxpy(
        n: c_int,
        alpha: c_float,
        x: *const c_float,
        incx: c_int,
        y: *mut c_float,
        incy: c_int,
    );
    pub fn cblas_daxpy(
        n: c_int,
        alpha: c_double,
        x: *const c_double,
        incx: c_int,
        y: *mut c_double,
        incy: c_int,
    );

    pub fn cblas_sscal(n: c_int, alpha: c_float, x: *mut c_float, incx: c_int);
    pub fn cblas_dscal(n: c_int, alpha: c_double, x: *mut c_double, incx: c_int);

    pub fn cblas_snrm2(n: c_int, x: *const c_float, incx: c_int) -> c_float;
    pub fn cblas_dnrm2(n: c_int, x: *const c_double, incx: c_int) -> c_double;

    pub fn cblas_sasum(n: c_int, x: *const c_float, incx: c_int) -> c_float;
    pub fn cblas_dasum(n: c_int, x: *const c_double, incx: c_int) -> c_double;

    pub fn cblas_isamax(n: c_int, x: *const c_float, incx: c_int) -> c_int;
    pub fn cblas_idamax(n: c_int, x: *const c_double, incx: c_int) -> c_int;

    pub fn cblas_scopy(n: c_int, x: *const c_float, incx: c_int, y: *mut c_float, incy: c_int);
    pub fn cblas_dcopy(n: c_int, x: *const c_double, incx: c_int, y: *mut c_double, incy: c_int);

    pub fn cblas_sswap(n: c_int, x: *mut c_float, incx: c_int, y: *mut c_float, incy: c_int);
    pub fn cblas_dswap(n: c_int, x: *mut c_double, incx: c_int, y: *mut c_double, incy: c_int);
}

// ═══════════════════════════════════════════════════════════════
// CBLAS Level 2
// ═══════════════════════════════════════════════════════════════

extern "C" {
    pub fn cblas_sgemv(
        layout: c_int,
        trans: c_int,
        m: c_int,
        n: c_int,
        alpha: c_float,
        a: *const c_float,
        lda: c_int,
        x: *const c_float,
        incx: c_int,
        beta: c_float,
        y: *mut c_float,
        incy: c_int,
    );
    pub fn cblas_dgemv(
        layout: c_int,
        trans: c_int,
        m: c_int,
        n: c_int,
        alpha: c_double,
        a: *const c_double,
        lda: c_int,
        x: *const c_double,
        incx: c_int,
        beta: c_double,
        y: *mut c_double,
        incy: c_int,
    );

    pub fn cblas_sger(
        layout: c_int,
        m: c_int,
        n: c_int,
        alpha: c_float,
        x: *const c_float,
        incx: c_int,
        y: *const c_float,
        incy: c_int,
        a: *mut c_float,
        lda: c_int,
    );
    pub fn cblas_dger(
        layout: c_int,
        m: c_int,
        n: c_int,
        alpha: c_double,
        x: *const c_double,
        incx: c_int,
        y: *const c_double,
        incy: c_int,
        a: *mut c_double,
        lda: c_int,
    );

    pub fn cblas_ssymv(
        layout: c_int,
        uplo: c_int,
        n: c_int,
        alpha: c_float,
        a: *const c_float,
        lda: c_int,
        x: *const c_float,
        incx: c_int,
        beta: c_float,
        y: *mut c_float,
        incy: c_int,
    );
    pub fn cblas_dsymv(
        layout: c_int,
        uplo: c_int,
        n: c_int,
        alpha: c_double,
        a: *const c_double,
        lda: c_int,
        x: *const c_double,
        incx: c_int,
        beta: c_double,
        y: *mut c_double,
        incy: c_int,
    );

    pub fn cblas_strmv(
        layout: c_int,
        uplo: c_int,
        trans: c_int,
        diag: c_int,
        n: c_int,
        a: *const c_float,
        lda: c_int,
        x: *mut c_float,
        incx: c_int,
    );

    pub fn cblas_strsv(
        layout: c_int,
        uplo: c_int,
        trans: c_int,
        diag: c_int,
        n: c_int,
        a: *const c_float,
        lda: c_int,
        x: *mut c_float,
        incx: c_int,
    );
}

// ═══════════════════════════════════════════════════════════════
// CBLAS Level 3
// ═══════════════════════════════════════════════════════════════

extern "C" {
    pub fn cblas_sgemm(
        layout: c_int,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_float,
        a: *const c_float,
        lda: c_int,
        b: *const c_float,
        ldb: c_int,
        beta: c_float,
        c: *mut c_float,
        ldc: c_int,
    );
    pub fn cblas_dgemm(
        layout: c_int,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_double,
        a: *const c_double,
        lda: c_int,
        b: *const c_double,
        ldb: c_int,
        beta: c_double,
        c: *mut c_double,
        ldc: c_int,
    );

    pub fn cblas_ssyrk(
        layout: c_int,
        uplo: c_int,
        trans: c_int,
        n: c_int,
        k: c_int,
        alpha: c_float,
        a: *const c_float,
        lda: c_int,
        beta: c_float,
        c: *mut c_float,
        ldc: c_int,
    );
    pub fn cblas_dsyrk(
        layout: c_int,
        uplo: c_int,
        trans: c_int,
        n: c_int,
        k: c_int,
        alpha: c_double,
        a: *const c_double,
        lda: c_int,
        beta: c_double,
        c: *mut c_double,
        ldc: c_int,
    );

    pub fn cblas_ssymm(
        layout: c_int,
        side: c_int,
        uplo: c_int,
        m: c_int,
        n: c_int,
        alpha: c_float,
        a: *const c_float,
        lda: c_int,
        b: *const c_float,
        ldb: c_int,
        beta: c_float,
        c: *mut c_float,
        ldc: c_int,
    );
    pub fn cblas_dsymm(
        layout: c_int,
        side: c_int,
        uplo: c_int,
        m: c_int,
        n: c_int,
        alpha: c_double,
        a: *const c_double,
        lda: c_int,
        b: *const c_double,
        ldb: c_int,
        beta: c_double,
        c: *mut c_double,
        ldc: c_int,
    );

    pub fn cblas_strsm(
        layout: c_int,
        side: c_int,
        uplo: c_int,
        trans: c_int,
        diag: c_int,
        m: c_int,
        n: c_int,
        alpha: c_float,
        a: *const c_float,
        lda: c_int,
        b: *mut c_float,
        ldb: c_int,
    );
}

// ═══════════════════════════════════════════════════════════════
// LAPACKE
// ═══════════════════════════════════════════════════════════════

extern "C" {
    pub fn LAPACKE_sgetrf(
        layout: c_int,
        m: c_int,
        n: c_int,
        a: *mut c_float,
        lda: c_int,
        ipiv: *mut c_int,
    ) -> c_int;
    pub fn LAPACKE_dgetrf(
        layout: c_int,
        m: c_int,
        n: c_int,
        a: *mut c_double,
        lda: c_int,
        ipiv: *mut c_int,
    ) -> c_int;

    pub fn LAPACKE_sgetrs(
        layout: c_int,
        trans: u8,
        n: c_int,
        nrhs: c_int,
        a: *const c_float,
        lda: c_int,
        ipiv: *const c_int,
        b: *mut c_float,
        ldb: c_int,
    ) -> c_int;
    pub fn LAPACKE_dgetrs(
        layout: c_int,
        trans: u8,
        n: c_int,
        nrhs: c_int,
        a: *const c_double,
        lda: c_int,
        ipiv: *const c_int,
        b: *mut c_double,
        ldb: c_int,
    ) -> c_int;

    pub fn LAPACKE_spotrf(layout: c_int, uplo: u8, n: c_int, a: *mut c_float, lda: c_int) -> c_int;
    pub fn LAPACKE_dpotrf(layout: c_int, uplo: u8, n: c_int, a: *mut c_double, lda: c_int)
        -> c_int;

    pub fn LAPACKE_spotrs(
        layout: c_int,
        uplo: u8,
        n: c_int,
        nrhs: c_int,
        a: *const c_float,
        lda: c_int,
        b: *mut c_float,
        ldb: c_int,
    ) -> c_int;

    pub fn LAPACKE_sgeqrf(
        layout: c_int,
        m: c_int,
        n: c_int,
        a: *mut c_float,
        lda: c_int,
        tau: *mut c_float,
    ) -> c_int;
    pub fn LAPACKE_dgeqrf(
        layout: c_int,
        m: c_int,
        n: c_int,
        a: *mut c_double,
        lda: c_int,
        tau: *mut c_double,
    ) -> c_int;
}

// ═══════════════════════════════════════════════════════════════
// VML (Vector Math Library)
// ═══════════════════════════════════════════════════════════════

extern "C" {
    pub fn vsExp(n: c_int, a: *const c_float, y: *mut c_float);
    pub fn vdExp(n: c_int, a: *const c_double, y: *mut c_double);
    pub fn vsLn(n: c_int, a: *const c_float, y: *mut c_float);
    pub fn vdLn(n: c_int, a: *const c_double, y: *mut c_double);
    pub fn vsSqrt(n: c_int, a: *const c_float, y: *mut c_float);
    pub fn vdSqrt(n: c_int, a: *const c_double, y: *mut c_double);
    pub fn vsAbs(n: c_int, a: *const c_float, y: *mut c_float);
    pub fn vdAbs(n: c_int, a: *const c_double, y: *mut c_double);
    pub fn vsAdd(n: c_int, a: *const c_float, b: *const c_float, y: *mut c_float);
    pub fn vsMul(n: c_int, a: *const c_float, b: *const c_float, y: *mut c_float);
    pub fn vsDiv(n: c_int, a: *const c_float, b: *const c_float, y: *mut c_float);
    pub fn vsSin(n: c_int, a: *const c_float, y: *mut c_float);
    pub fn vsCos(n: c_int, a: *const c_float, y: *mut c_float);
    pub fn vsPow(n: c_int, a: *const c_float, b: *const c_float, y: *mut c_float);
}

// ═══════════════════════════════════════════════════════════════
// DFTI (Discrete Fourier Transform Interface)
// ═══════════════════════════════════════════════════════════════

/// Opaque handle for DFTI descriptor.
pub type DftiDescriptorHandle = *mut c_void;

/// DFTI precision constants.
pub const DFTI_SINGLE: c_int = 35;
pub const DFTI_DOUBLE: c_int = 36;

/// DFTI domain constants.
pub const DFTI_COMPLEX: c_int = 32;
pub const DFTI_REAL: c_int = 33;

/// DFTI config constants.
pub const DFTI_PLACEMENT: c_int = 11;
pub const DFTI_INPLACE: c_int = 43;
pub const DFTI_NOT_INPLACE: c_int = 44;
pub const DFTI_BACKWARD_SCALE: c_int = 5;

extern "C" {
    pub fn DftiCreateDescriptor(
        handle: *mut DftiDescriptorHandle,
        precision: c_int,
        domain: c_int,
        dimension: c_int,
        length: c_long,
    ) -> c_long;
    pub fn DftiSetValue(handle: DftiDescriptorHandle, param: c_int, ...) -> c_long;
    pub fn DftiCommitDescriptor(handle: DftiDescriptorHandle) -> c_long;
    pub fn DftiComputeForward(handle: DftiDescriptorHandle, x_inout: *mut c_void, ...) -> c_long;
    pub fn DftiComputeBackward(handle: DftiDescriptorHandle, x_inout: *mut c_void, ...) -> c_long;
    pub fn DftiFreeDescriptor(handle: *mut DftiDescriptorHandle) -> c_long;
}

// ═══════════════════════════════════════════════════════════════
// MKL INT8/BF16 extensions (AMX, VNNI)
// ═══════════════════════════════════════════════════════════════

extern "C" {
    /// MKL's BF16 GEMM (requires AVX512-BF16 or AMX-BF16).
    /// a and b are raw u16 (BF16 bit patterns).
    pub fn cblas_gemm_bf16bf16f32(
        layout: c_int,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_float,
        a: *const u16,
        lda: c_int,
        b: *const u16,
        ldb: c_int,
        beta: c_float,
        c: *mut c_float,
        ldc: c_int,
    );

    /// MKL's INT8 GEMM with mixed signedness: signed A × unsigned B → i32 accumulate.
    pub fn cblas_gemm_s8u8s32(
        layout: c_int,
        transa: c_int,
        transb: c_int,
        offsetc: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_float,
        a: *const i8,
        lda: c_int,
        oa: i8,
        b: *const u8,
        ldb: c_int,
        ob: u8,
        beta: c_float,
        c: *mut c_int,
        ldc: c_int,
        oc: *const c_int,
    );
}
