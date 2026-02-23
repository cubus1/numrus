//! BLAS Level 3: Matrix-matrix operations.
//!
//! The crown jewel is GEMM — cache-blocked with AVX-512/AVX2 microkernels.
//! Uses the Goto BLAS algorithm: pack panels of A and B into contiguous
//! cache-friendly buffers, then invoke the microkernel on tiles.
//!
//! The SIMD width adapts at compile time via feature flags:
//! - `avx512`: f32x16 (6x16 microkernel), f64x8 (6x8 microkernel)
//! - `avx2`:   f32x8  (6x8  microkernel), f64x4 (4x4 microkernel)

use numrus_core::layout::{Diag, Layout, Side, Transpose, Uplo};
use numrus_core::simd::{
    DGEMM_KC, DGEMM_MC, DGEMM_MR, DGEMM_NC, DGEMM_NR, SGEMM_KC, SGEMM_MC, SGEMM_MR, SGEMM_NC,
    SGEMM_NR,
};
use std::simd::StdFloat;

// SIMD vector types selected by feature flag — no runtime branching
#[cfg(feature = "avx512")]
use std::simd::{f32x16 as F32Simd, f64x8 as F64Simd};
#[cfg(not(feature = "avx512"))]
use std::simd::{f32x8 as F32Simd, f64x4 as F64Simd};

/// Wrapper to send a raw mutable pointer across thread boundaries.
/// Safety: The caller must ensure non-overlapping access between threads.
#[derive(Clone, Copy)]
struct SendMutPtr<T> {
    ptr: *mut T,
    len: usize,
}
unsafe impl<T> Send for SendMutPtr<T> {}
unsafe impl<T> Sync for SendMutPtr<T> {}

impl<T> SendMutPtr<T> {
    fn new(slice: &mut [T]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
        }
    }

    /// Get a mutable slice. Safety: caller ensures no aliasing.
    #[allow(clippy::mut_from_ref)] // Intentional: raw pointer interior mutability for parallel tiling.
    unsafe fn as_mut_slice(&self) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.ptr, self.len)
    }
}

// ============================================================================
// SGEMM: Single-precision General Matrix Multiply
// C := alpha * op(A) * op(B) + beta * C
// ============================================================================

/// Single-precision GEMM with cache-blocked AVX-512 microkernel.
///
/// C := alpha * op(A) * op(B) + beta * C
///
/// Supports both row-major and column-major layouts.
/// Uses Goto BLAS algorithm: panel packing + tiled microkernel.
/// Multi-threaded for large matrices.
pub fn sgemm(
    layout: Layout,
    trans_a: Transpose,
    trans_b: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::cblas_sgemm(
                layout as i32,
                trans_a as i32,
                trans_b as i32,
                m as i32,
                n as i32,
                k as i32,
                alpha,
                a.as_ptr(),
                lda as i32,
                b.as_ptr(),
                ldb as i32,
                beta,
                c.as_mut_ptr(),
                ldc as i32,
            );
        }
        return;
    }

    // Scale C by beta
    if beta == 0.0 {
        for i in 0..m {
            for j in 0..n {
                c[layout.index(i, j, ldc)] = 0.0;
            }
        }
    } else if beta != 1.0 {
        for i in 0..m {
            for j in 0..n {
                let idx = layout.index(i, j, ldc);
                c[idx] *= beta;
            }
        }
    }

    if alpha == 0.0 || m == 0 || n == 0 || k == 0 {
        return;
    }

    // For small matrices, use simple triple-loop.
    // Threshold: below ~48^3 ≈ 110K flops, the packing overhead isn't worth it.
    if m * n * k < 110_000 {
        sgemm_simple(
            layout, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, c, ldc,
        );
        return;
    }

    // Cache-blocked GEMM with packing
    sgemm_blocked(
        layout, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, c, ldc,
    );
}

/// Small-matrix GEMM with SIMD dot products.
///
/// Pre-gathers A rows and B columns into contiguous buffers so every
/// inner product streams through the SIMD bus at full width.
/// B columns (or transposed B rows) are gathered once and reused across all A rows.
fn sgemm_simple(
    layout: Layout,
    trans_a: Transpose,
    trans_b: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    c: &mut [f32],
    ldc: usize,
) {
    use numrus_core::simd;

    // Pre-gather B columns into contiguous layout (k elements per column, n columns).
    // This converts strided column access into sequential cache-line-filling reads.
    let mut b_cols = vec![0.0f32; n * k];
    for j in 0..n {
        for p in 0..k {
            let b_val = match (layout, trans_b) {
                (Layout::RowMajor, Transpose::NoTrans) => b[p * ldb + j],
                (Layout::RowMajor, _) => b[j * ldb + p],
                (Layout::ColMajor, Transpose::NoTrans) => b[j * ldb + p],
                (Layout::ColMajor, _) => b[p * ldb + j],
            };
            b_cols[j * k + p] = b_val;
        }
    }

    for i in 0..m {
        // Check if A row is already contiguous (RowMajor NoTrans or ColMajor Trans)
        let a_row_contiguous = matches!(
            (layout, trans_a),
            (Layout::RowMajor, Transpose::NoTrans)
                | (Layout::ColMajor, Transpose::Trans | Transpose::ConjTrans)
        );

        if a_row_contiguous {
            let a_start = i * lda;
            let a_row = &a[a_start..a_start + k];
            for j in 0..n {
                let dot = simd::dot_f32(a_row, &b_cols[j * k..(j + 1) * k]);
                let idx = layout.index(i, j, ldc);
                c[idx] += alpha * dot;
            }
        } else {
            // Gather A row into contiguous buffer
            let mut a_row = vec![0.0f32; k];
            for p in 0..k {
                a_row[p] = match (layout, trans_a) {
                    (Layout::RowMajor, _) => a[p * lda + i],
                    (Layout::ColMajor, Transpose::NoTrans) => a[p * lda + i],
                    (Layout::ColMajor, _) => a[i * lda + p],
                };
            }
            for j in 0..n {
                let dot = simd::dot_f32(&a_row, &b_cols[j * k..(j + 1) * k]);
                let idx = layout.index(i, j, ldc);
                c[idx] += alpha * dot;
            }
        }
    }
}

/// Threshold for multithreaded GEMM (total elements in C).
/// Below this, single-threaded blocked is faster due to thread spawn overhead.
/// Tuned: 256*256=65536 is the crossover where MT overhead pays off.
const SGEMM_PARALLEL_THRESHOLD: usize = 256 * 256;

/// Cache-blocked GEMM using the Goto BLAS algorithm.
///
/// Memory hierarchy:
/// - Outer loop over N (L3 blocks of NC columns)
/// - Middle loop over K (L2 blocks of KC depth)
/// - Pack panel of B into contiguous buffer (shared across threads)
/// - Inner loop over M (parallelized across threads)
/// - Each thread packs its own A panel + runs microkernel
///
/// Multithreading: For large matrices (m*n > 4096), the M-loop is
/// parallelized using scoped threads with `split_at_mut` — each thread
/// gets exclusive ownership of its C rows. No Arc, no Mutex, no locks.
fn sgemm_blocked(
    layout: Layout,
    trans_a: Transpose,
    trans_b: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    c: &mut [f32],
    ldc: usize,
) {
    let mc = SGEMM_MC.min(m);
    let nc = SGEMM_NC.min(n);
    let kc = SGEMM_KC.min(k);

    // Packed buffers — padded to MR/NR boundaries for microkernel alignment
    let mc_padded = mc.div_ceil(SGEMM_MR) * SGEMM_MR;
    let nc_padded = nc.div_ceil(SGEMM_NR) * SGEMM_NR;

    let use_parallel = m * n > SGEMM_PARALLEL_THRESHOLD;
    let num_threads = if use_parallel {
        std::thread::available_parallelism()
            .map(|t| t.get())
            .unwrap_or(1)
    } else {
        1
    };

    // Packed B is shared across all threads (read-only after packing)
    let mut packed_b = vec![0.0f32; kc * nc_padded];

    for jc in (0..n).step_by(nc) {
        let jb = nc.min(n - jc);

        for pc in (0..k).step_by(kc) {
            let pb = kc.min(k - pc);

            // Pack B panel: shared across threads
            pack_b_f32(layout, trans_b, b, ldb, pc, jc, pb, jb, &mut packed_b);

            if num_threads > 1 {
                // ── Multithreaded M-loop ──
                // Each thread works on disjoint rows of C.
                // We wrap C's pointer in SendPtr so threads can write to
                // non-overlapping regions — the blackboard borrow-mut pattern.
                let c_send = SendMutPtr::new(c);
                let rows_per_thread = m.div_ceil(num_threads).div_ceil(SGEMM_MR) * SGEMM_MR;

                // Collect work items (row ranges) upfront
                let mut work_items = Vec::new();
                let mut row = 0;
                while row < m {
                    let row_end = (row + rows_per_thread).min(m);
                    work_items.push((row, row_end - row));
                    row = row_end;
                }

                std::thread::scope(|s| {
                    let packed_b_ref = &packed_b;
                    for &(ic, thread_m) in &work_items {
                        let c_ptr = c_send;
                        s.spawn(move || {
                            // Safety: each thread writes to rows [ic..ic+thread_m] of C,
                            // which are non-overlapping across threads.
                            let c_slice = unsafe { c_ptr.as_mut_slice() };
                            let mut thread_packed_a = vec![0.0f32; mc_padded * kc];

                            let mut thread_ic = ic;
                            while thread_ic < ic + thread_m {
                                let ib = mc.min(ic + thread_m - thread_ic);
                                pack_a_f32(
                                    layout,
                                    trans_a,
                                    a,
                                    lda,
                                    thread_ic,
                                    pc,
                                    ib,
                                    pb,
                                    &mut thread_packed_a,
                                );
                                sgemm_macrokernel(
                                    alpha,
                                    &thread_packed_a,
                                    packed_b_ref,
                                    c_slice,
                                    layout,
                                    ldc,
                                    thread_ic,
                                    jc,
                                    ib,
                                    jb,
                                    pb,
                                );
                                thread_ic += mc;
                            }
                        });
                    }
                });
            } else {
                // ── Single-threaded M-loop ──
                let mut packed_a = vec![0.0f32; mc_padded * kc];
                for ic in (0..m).step_by(mc) {
                    let ib = mc.min(m - ic);
                    pack_a_f32(layout, trans_a, a, lda, ic, pc, ib, pb, &mut packed_a);
                    sgemm_macrokernel(
                        alpha, &packed_a, &packed_b, c, layout, ldc, ic, jc, ib, jb, pb,
                    );
                }
            }
        }
    }
}

/// Pack A panel into MR-row contiguous strips for cache-friendly access.
fn pack_a_f32(
    layout: Layout,
    trans: Transpose,
    a: &[f32],
    lda: usize,
    row_start: usize,
    col_start: usize,
    rows: usize,
    cols: usize,
    packed: &mut [f32],
) {
    let mut idx = 0;
    for i_block in (0..rows).step_by(SGEMM_MR) {
        let mr = SGEMM_MR.min(rows - i_block);
        for p in 0..cols {
            for ir in 0..SGEMM_MR {
                if ir < mr {
                    let i = row_start + i_block + ir;
                    let j = col_start + p;
                    packed[idx] = match (layout, trans) {
                        (Layout::RowMajor, Transpose::NoTrans) => a[i * lda + j],
                        (Layout::RowMajor, _) => a[j * lda + i],
                        (Layout::ColMajor, Transpose::NoTrans) => a[j * lda + i],
                        (Layout::ColMajor, _) => a[i * lda + j],
                    };
                } else {
                    packed[idx] = 0.0; // Pad with zeros
                }
                idx += 1;
            }
        }
    }
}

/// Pack B panel into NR-column contiguous strips.
fn pack_b_f32(
    layout: Layout,
    trans: Transpose,
    b: &[f32],
    ldb: usize,
    row_start: usize,
    col_start: usize,
    rows: usize,
    cols: usize,
    packed: &mut [f32],
) {
    let mut idx = 0;
    for j_block in (0..cols).step_by(SGEMM_NR) {
        let nr = SGEMM_NR.min(cols - j_block);
        for p in 0..rows {
            for jr in 0..SGEMM_NR {
                if jr < nr {
                    let i = row_start + p;
                    let j = col_start + j_block + jr;
                    packed[idx] = match (layout, trans) {
                        (Layout::RowMajor, Transpose::NoTrans) => b[i * ldb + j],
                        (Layout::RowMajor, _) => b[j * ldb + i],
                        (Layout::ColMajor, Transpose::NoTrans) => b[j * ldb + i],
                        (Layout::ColMajor, _) => b[i * ldb + j],
                    };
                } else {
                    packed[idx] = 0.0;
                }
                idx += 1;
            }
        }
    }
}

/// Macro-kernel: dispatch MR x NR microkernels over the packed panels.
fn sgemm_macrokernel(
    alpha: f32,
    packed_a: &[f32],
    packed_b: &[f32],
    c: &mut [f32],
    layout: Layout,
    ldc: usize,
    ic: usize,
    jc: usize,
    mb: usize,
    nb: usize,
    kb: usize,
) {
    let mr_blocks = mb.div_ceil(SGEMM_MR);
    let nr_blocks = nb.div_ceil(SGEMM_NR);

    for jr in 0..nr_blocks {
        let nr = SGEMM_NR.min(nb - jr * SGEMM_NR);

        for ir in 0..mr_blocks {
            let mr = SGEMM_MR.min(mb - ir * SGEMM_MR);

            // Microkernel: compute MR x NR tile of C
            let a_offset = ir * SGEMM_MR * kb;
            let b_offset = jr * SGEMM_NR * kb;

            sgemm_microkernel_6x16(
                alpha,
                &packed_a[a_offset..],
                &packed_b[b_offset..],
                c,
                layout,
                ldc,
                ic + ir * SGEMM_MR,
                jc + jr * SGEMM_NR,
                mr,
                nr,
                kb,
            );
        }
    }
}

/// AVX-512 microkernel: 6x16 tile of C using f32x16 SIMD.
///
/// Uses 6 accumulator registers (one per row of the tile),
/// each holding 16 f32 values — exactly one zmm register.
/// This gives 6 * 16 = 96 FMA operations per K iteration.
#[inline(always)]
fn sgemm_microkernel_6x16(
    alpha: f32,
    packed_a: &[f32],
    packed_b: &[f32],
    c: &mut [f32],
    layout: Layout,
    ldc: usize,
    row: usize,
    col: usize,
    mr: usize,
    nr: usize,
    kb: usize,
) {
    // Accumulator registers for MR rows x NR columns
    let mut acc = [F32Simd::splat(0.0); 6];

    // Main K loop
    for p in 0..kb {
        // Load NR elements of B for this K step
        let b_base = p * SGEMM_NR;
        let b_vec = if nr >= SGEMM_NR {
            F32Simd::from_slice(&packed_b[b_base..])
        } else {
            // Partial: pad with zeros — stack array, not heap
            let mut tmp = [0.0f32; SGEMM_NR];
            tmp[..nr].copy_from_slice(&packed_b[b_base..b_base + nr]);
            F32Simd::from_slice(&tmp)
        };

        // Load MR elements of A and broadcast-FMA
        let a_base = p * SGEMM_MR;
        for ir in 0..mr.min(SGEMM_MR) {
            let a_val = F32Simd::splat(packed_a[a_base + ir]);
            acc[ir] = a_val.mul_add(b_vec, acc[ir]);
        }
    }

    // Store results back to C with alpha scaling
    let alpha_v = F32Simd::splat(alpha);
    for ir in 0..mr.min(SGEMM_MR) {
        let result = acc[ir] * alpha_v;
        match layout {
            Layout::RowMajor if nr >= SGEMM_NR => {
                // Full-width row: SIMD store directly into contiguous C row
                let base = (row + ir) * ldc + col;
                let existing = F32Simd::from_slice(&c[base..]);
                let sum = existing + result;
                sum.copy_to_slice(&mut c[base..base + SGEMM_NR]);
            }
            _ => {
                // Partial row or ColMajor: scalar fallback
                let buf = result.to_array();
                for jr in 0..nr {
                    let idx = layout.index(row + ir, col + jr, ldc);
                    c[idx] += buf[jr];
                }
            }
        }
    }
}

// ============================================================================
// DGEMM: Double-precision General Matrix Multiply
// ============================================================================

/// Double-precision GEMM: C := alpha * op(A) * op(B) + beta * C
pub fn dgemm(
    layout: Layout,
    trans_a: Transpose,
    trans_b: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    beta: f64,
    c: &mut [f64],
    ldc: usize,
) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::cblas_dgemm(
                layout as i32,
                trans_a as i32,
                trans_b as i32,
                m as i32,
                n as i32,
                k as i32,
                alpha,
                a.as_ptr(),
                lda as i32,
                b.as_ptr(),
                ldb as i32,
                beta,
                c.as_mut_ptr(),
                ldc as i32,
            );
        }
        return;
    }

    // Scale C by beta
    if beta == 0.0 {
        for i in 0..m {
            for j in 0..n {
                c[layout.index(i, j, ldc)] = 0.0;
            }
        }
    } else if beta != 1.0 {
        for i in 0..m {
            for j in 0..n {
                let idx = layout.index(i, j, ldc);
                c[idx] *= beta;
            }
        }
    }

    if alpha == 0.0 || m == 0 || n == 0 || k == 0 {
        return;
    }

    // Small matrices: simple triple-loop
    if m * n * k < 110_000 {
        dgemm_simple(
            layout, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, c, ldc,
        );
        return;
    }

    // Cache-blocked DGEMM with packing + multithreading
    dgemm_blocked(
        layout, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, c, ldc,
    );
}

/// Small-matrix DGEMM with SIMD dot products.
fn dgemm_simple(
    layout: Layout,
    trans_a: Transpose,
    trans_b: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    c: &mut [f64],
    ldc: usize,
) {
    use numrus_core::simd;

    // Pre-gather B columns into contiguous layout
    let mut b_cols = vec![0.0f64; n * k];
    for j in 0..n {
        for p in 0..k {
            let b_val = match (layout, trans_b) {
                (Layout::RowMajor, Transpose::NoTrans) => b[p * ldb + j],
                (Layout::RowMajor, _) => b[j * ldb + p],
                (Layout::ColMajor, Transpose::NoTrans) => b[j * ldb + p],
                (Layout::ColMajor, _) => b[p * ldb + j],
            };
            b_cols[j * k + p] = b_val;
        }
    }

    for i in 0..m {
        let a_row_contiguous = matches!(
            (layout, trans_a),
            (Layout::RowMajor, Transpose::NoTrans)
                | (Layout::ColMajor, Transpose::Trans | Transpose::ConjTrans)
        );

        if a_row_contiguous {
            let a_start = i * lda;
            let a_row = &a[a_start..a_start + k];
            for j in 0..n {
                let dot = simd::dot_f64(a_row, &b_cols[j * k..(j + 1) * k]);
                let idx = layout.index(i, j, ldc);
                c[idx] += alpha * dot;
            }
        } else {
            let mut a_row = vec![0.0f64; k];
            for p in 0..k {
                a_row[p] = match (layout, trans_a) {
                    (Layout::RowMajor, _) => a[p * lda + i],
                    (Layout::ColMajor, Transpose::NoTrans) => a[p * lda + i],
                    (Layout::ColMajor, _) => a[i * lda + p],
                };
            }
            for j in 0..n {
                let dot = simd::dot_f64(&a_row, &b_cols[j * k..(j + 1) * k]);
                let idx = layout.index(i, j, ldc);
                c[idx] += alpha * dot;
            }
        }
    }
}

/// Threshold for multithreaded DGEMM.
const DGEMM_PARALLEL_THRESHOLD: usize = 256 * 256;

/// Cache-blocked DGEMM using Goto BLAS algorithm with 6x8 f64 microkernel.
fn dgemm_blocked(
    layout: Layout,
    trans_a: Transpose,
    trans_b: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    c: &mut [f64],
    ldc: usize,
) {
    let mc = DGEMM_MC.min(m);
    let nc = DGEMM_NC.min(n);
    let kc = DGEMM_KC.min(k);

    let mc_padded = mc.div_ceil(DGEMM_MR) * DGEMM_MR;
    let nc_padded = nc.div_ceil(DGEMM_NR) * DGEMM_NR;

    let use_parallel = m * n > DGEMM_PARALLEL_THRESHOLD;
    let num_threads = if use_parallel {
        std::thread::available_parallelism()
            .map(|t| t.get())
            .unwrap_or(1)
    } else {
        1
    };

    let mut packed_b = vec![0.0f64; kc * nc_padded];

    for jc in (0..n).step_by(nc) {
        let jb = nc.min(n - jc);

        for pc in (0..k).step_by(kc) {
            let pb = kc.min(k - pc);

            pack_b_f64(layout, trans_b, b, ldb, pc, jc, pb, jb, &mut packed_b);

            if num_threads > 1 {
                let c_send = SendMutPtr::new(c);
                let rows_per_thread = m.div_ceil(num_threads).div_ceil(DGEMM_MR) * DGEMM_MR;

                let mut work_items = Vec::new();
                let mut row = 0;
                while row < m {
                    let row_end = (row + rows_per_thread).min(m);
                    work_items.push((row, row_end - row));
                    row = row_end;
                }

                std::thread::scope(|s| {
                    let packed_b_ref = &packed_b;
                    for &(ic, thread_m) in &work_items {
                        let c_ptr = c_send;
                        s.spawn(move || {
                            let c_slice = unsafe { c_ptr.as_mut_slice() };
                            let mut thread_packed_a = vec![0.0f64; mc_padded * kc];

                            let mut thread_ic = ic;
                            while thread_ic < ic + thread_m {
                                let ib = mc.min(ic + thread_m - thread_ic);
                                pack_a_f64(
                                    layout,
                                    trans_a,
                                    a,
                                    lda,
                                    thread_ic,
                                    pc,
                                    ib,
                                    pb,
                                    &mut thread_packed_a,
                                );
                                dgemm_macrokernel(
                                    alpha,
                                    &thread_packed_a,
                                    packed_b_ref,
                                    c_slice,
                                    layout,
                                    ldc,
                                    thread_ic,
                                    jc,
                                    ib,
                                    jb,
                                    pb,
                                );
                                thread_ic += mc;
                            }
                        });
                    }
                });
            } else {
                let mut packed_a = vec![0.0f64; mc_padded * kc];
                for ic in (0..m).step_by(mc) {
                    let ib = mc.min(m - ic);
                    pack_a_f64(layout, trans_a, a, lda, ic, pc, ib, pb, &mut packed_a);
                    dgemm_macrokernel(
                        alpha, &packed_a, &packed_b, c, layout, ldc, ic, jc, ib, jb, pb,
                    );
                }
            }
        }
    }
}

/// Pack A panel for DGEMM (MR-row strips).
fn pack_a_f64(
    layout: Layout,
    trans: Transpose,
    a: &[f64],
    lda: usize,
    row_start: usize,
    col_start: usize,
    rows: usize,
    cols: usize,
    packed: &mut [f64],
) {
    let mut idx = 0;
    for i_block in (0..rows).step_by(DGEMM_MR) {
        let mr = DGEMM_MR.min(rows - i_block);
        for p in 0..cols {
            for ir in 0..DGEMM_MR {
                if ir < mr {
                    let i = row_start + i_block + ir;
                    let j = col_start + p;
                    packed[idx] = match (layout, trans) {
                        (Layout::RowMajor, Transpose::NoTrans) => a[i * lda + j],
                        (Layout::RowMajor, _) => a[j * lda + i],
                        (Layout::ColMajor, Transpose::NoTrans) => a[j * lda + i],
                        (Layout::ColMajor, _) => a[i * lda + j],
                    };
                } else {
                    packed[idx] = 0.0;
                }
                idx += 1;
            }
        }
    }
}

/// Pack B panel for DGEMM (NR-column strips).
fn pack_b_f64(
    layout: Layout,
    trans: Transpose,
    b: &[f64],
    ldb: usize,
    row_start: usize,
    col_start: usize,
    rows: usize,
    cols: usize,
    packed: &mut [f64],
) {
    let mut idx = 0;
    for j_block in (0..cols).step_by(DGEMM_NR) {
        let nr = DGEMM_NR.min(cols - j_block);
        for p in 0..rows {
            for jr in 0..DGEMM_NR {
                if jr < nr {
                    let i = row_start + p;
                    let j = col_start + j_block + jr;
                    packed[idx] = match (layout, trans) {
                        (Layout::RowMajor, Transpose::NoTrans) => b[i * ldb + j],
                        (Layout::RowMajor, _) => b[j * ldb + i],
                        (Layout::ColMajor, Transpose::NoTrans) => b[j * ldb + i],
                        (Layout::ColMajor, _) => b[i * ldb + j],
                    };
                } else {
                    packed[idx] = 0.0;
                }
                idx += 1;
            }
        }
    }
}

/// DGEMM macro-kernel: dispatch 6x8 microkernels.
fn dgemm_macrokernel(
    alpha: f64,
    packed_a: &[f64],
    packed_b: &[f64],
    c: &mut [f64],
    layout: Layout,
    ldc: usize,
    ic: usize,
    jc: usize,
    mb: usize,
    nb: usize,
    kb: usize,
) {
    let mr_blocks = mb.div_ceil(DGEMM_MR);
    let nr_blocks = nb.div_ceil(DGEMM_NR);

    for jr in 0..nr_blocks {
        let nr = DGEMM_NR.min(nb - jr * DGEMM_NR);
        for ir in 0..mr_blocks {
            let mr = DGEMM_MR.min(mb - ir * DGEMM_MR);
            let a_offset = ir * DGEMM_MR * kb;
            let b_offset = jr * DGEMM_NR * kb;
            dgemm_microkernel_6x8(
                alpha,
                &packed_a[a_offset..],
                &packed_b[b_offset..],
                c,
                layout,
                ldc,
                ic + ir * DGEMM_MR,
                jc + jr * DGEMM_NR,
                mr,
                nr,
                kb,
            );
        }
    }
}

/// DGEMM microkernel: MR rows x NR columns using F64Simd.
/// AVX-512: 6x8 (f64x8), AVX2: 4x4 (f64x4).
#[inline(always)]
fn dgemm_microkernel_6x8(
    alpha: f64,
    packed_a: &[f64],
    packed_b: &[f64],
    c: &mut [f64],
    layout: Layout,
    ldc: usize,
    row: usize,
    col: usize,
    mr: usize,
    nr: usize,
    kb: usize,
) {
    let mut acc = [F64Simd::splat(0.0); 6];

    for p in 0..kb {
        let b_base = p * DGEMM_NR;
        let b_vec = if nr >= DGEMM_NR {
            F64Simd::from_slice(&packed_b[b_base..])
        } else {
            // Partial: pad with zeros — stack array, not heap
            let mut tmp = [0.0f64; DGEMM_NR];
            tmp[..nr].copy_from_slice(&packed_b[b_base..b_base + nr]);
            F64Simd::from_slice(&tmp)
        };

        // Broadcast-FMA
        let a_base = p * DGEMM_MR;
        for ir in 0..mr.min(DGEMM_MR) {
            let a_val = F64Simd::splat(packed_a[a_base + ir]);
            acc[ir] = a_val.mul_add(b_vec, acc[ir]);
        }
    }

    let alpha_v = F64Simd::splat(alpha);
    for ir in 0..mr.min(DGEMM_MR) {
        let result = acc[ir] * alpha_v;
        match layout {
            Layout::RowMajor if nr >= DGEMM_NR => {
                // Full-width row: SIMD store directly into contiguous C row
                let base = (row + ir) * ldc + col;
                let existing = F64Simd::from_slice(&c[base..]);
                let sum = existing + result;
                sum.copy_to_slice(&mut c[base..base + DGEMM_NR]);
            }
            _ => {
                // Partial row or ColMajor: scalar fallback
                let buf = result.to_array();
                for jr in 0..nr {
                    let idx = layout.index(row + ir, col + jr, ldc);
                    c[idx] += buf[jr];
                }
            }
        }
    }
}

// ============================================================================
// SSYRK: Symmetric rank-k update  C := alpha * A * A^T + beta * C
// ============================================================================

/// Single-precision SYRK: C := alpha * A * A^T + beta * C (or A^T * A)
///
/// For RowMajor NoTrans, both A rows are contiguous — direct SIMD dot
/// between A_row_i and A_row_j fills the full SIMD bus.
/// For other layouts/transpositions, rows are gathered into contiguous buffers.
pub fn ssyrk(
    layout: Layout,
    uplo: Uplo,
    trans: Transpose,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::cblas_ssyrk(
                layout as i32,
                uplo as i32,
                trans as i32,
                n as i32,
                k as i32,
                alpha,
                a.as_ptr(),
                lda as i32,
                beta,
                c.as_mut_ptr(),
                ldc as i32,
            );
        }
        return;
    }

    use numrus_core::simd;

    // Scale C by beta
    for i in 0..n {
        let (j_start, j_end) = match uplo {
            Uplo::Upper => (i, n),
            Uplo::Lower => (0, i + 1),
        };
        for j in j_start..j_end {
            let idx = layout.index(i, j, ldc);
            c[idx] = if beta == 0.0 { 0.0 } else { c[idx] * beta };
        }
    }

    if alpha == 0.0 {
        return;
    }

    // Check if A "rows" (for op(A)) are contiguous
    let rows_contiguous = matches!(
        (layout, trans),
        (Layout::RowMajor, Transpose::NoTrans)
            | (Layout::ColMajor, Transpose::Trans | Transpose::ConjTrans)
    );

    if rows_contiguous {
        // Direct SIMD dot between contiguous A rows
        for i in 0..n {
            let a_i = &a[i * lda..i * lda + k];
            let (j_start, j_end) = match uplo {
                Uplo::Upper => (i, n),
                Uplo::Lower => (0, i + 1),
            };
            for j in j_start..j_end {
                let a_j = &a[j * lda..j * lda + k];
                let dot = simd::dot_f32(a_i, a_j);
                let idx = layout.index(i, j, ldc);
                c[idx] += alpha * dot;
            }
        }
    } else {
        // Gather all n row-vectors of op(A) into contiguous buffers
        let mut a_rows = vec![0.0f32; n * k];
        for i in 0..n {
            for p in 0..k {
                a_rows[i * k + p] = match (layout, trans) {
                    (Layout::RowMajor, _) => a[p * lda + i],
                    (Layout::ColMajor, Transpose::NoTrans) => a[p * lda + i],
                    (Layout::ColMajor, _) => a[i * lda + p],
                };
            }
        }
        for i in 0..n {
            let (j_start, j_end) = match uplo {
                Uplo::Upper => (i, n),
                Uplo::Lower => (0, i + 1),
            };
            for j in j_start..j_end {
                let dot = simd::dot_f32(&a_rows[i * k..(i + 1) * k], &a_rows[j * k..(j + 1) * k]);
                let idx = layout.index(i, j, ldc);
                c[idx] += alpha * dot;
            }
        }
    }
}

/// Double-precision SYRK: C := alpha * A * A^T + beta * C
pub fn dsyrk(
    layout: Layout,
    uplo: Uplo,
    trans: Transpose,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    beta: f64,
    c: &mut [f64],
    ldc: usize,
) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::cblas_dsyrk(
                layout as i32,
                uplo as i32,
                trans as i32,
                n as i32,
                k as i32,
                alpha,
                a.as_ptr(),
                lda as i32,
                beta,
                c.as_mut_ptr(),
                ldc as i32,
            );
        }
        return;
    }

    use numrus_core::simd;

    for i in 0..n {
        let (j_start, j_end) = match uplo {
            Uplo::Upper => (i, n),
            Uplo::Lower => (0, i + 1),
        };
        for j in j_start..j_end {
            let idx = layout.index(i, j, ldc);
            c[idx] = if beta == 0.0 { 0.0 } else { c[idx] * beta };
        }
    }

    if alpha == 0.0 {
        return;
    }

    let rows_contiguous = matches!(
        (layout, trans),
        (Layout::RowMajor, Transpose::NoTrans)
            | (Layout::ColMajor, Transpose::Trans | Transpose::ConjTrans)
    );

    if rows_contiguous {
        for i in 0..n {
            let a_i = &a[i * lda..i * lda + k];
            let (j_start, j_end) = match uplo {
                Uplo::Upper => (i, n),
                Uplo::Lower => (0, i + 1),
            };
            for j in j_start..j_end {
                let a_j = &a[j * lda..j * lda + k];
                let dot = simd::dot_f64(a_i, a_j);
                let idx = layout.index(i, j, ldc);
                c[idx] += alpha * dot;
            }
        }
    } else {
        let mut a_rows = vec![0.0f64; n * k];
        for i in 0..n {
            for p in 0..k {
                a_rows[i * k + p] = match (layout, trans) {
                    (Layout::RowMajor, _) => a[p * lda + i],
                    (Layout::ColMajor, Transpose::NoTrans) => a[p * lda + i],
                    (Layout::ColMajor, _) => a[i * lda + p],
                };
            }
        }
        for i in 0..n {
            let (j_start, j_end) = match uplo {
                Uplo::Upper => (i, n),
                Uplo::Lower => (0, i + 1),
            };
            for j in j_start..j_end {
                let dot = simd::dot_f64(&a_rows[i * k..(i + 1) * k], &a_rows[j * k..(j + 1) * k]);
                let idx = layout.index(i, j, ldc);
                c[idx] += alpha * dot;
            }
        }
    }
}

// ============================================================================
// STRSM: Triangular solve with multiple right-hand sides
// op(A) * X = alpha * B  or  X * op(A) = alpha * B
// ============================================================================

/// Single-precision TRSM: solve op(A) * X = alpha * B (A triangular)
///
/// Sequential dependencies between rows limit full SIMD, but the j-loop
/// (across RHS columns) uses SIMD scal for the alpha scaling,
/// and the inner p-loop column gather uses SIMD dot where applicable.
pub fn strsm(
    layout: Layout,
    side: Side,
    uplo: Uplo,
    trans: Transpose,
    diag: Diag,
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &mut [f32],
    ldb: usize,
) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::cblas_strsm(
                layout as i32,
                side as i32,
                uplo as i32,
                trans as i32,
                diag as i32,
                m as i32,
                n as i32,
                alpha,
                a.as_ptr(),
                lda as i32,
                b.as_mut_ptr(),
                ldb as i32,
            );
        }
        return;
    }

    use numrus_core::simd;
    let unit = diag == Diag::Unit;

    // Scale B by alpha — SIMD scal on contiguous rows
    if alpha != 1.0 {
        match layout {
            Layout::RowMajor => {
                for i in 0..m {
                    simd::scal_f32(alpha, &mut b[i * ldb..i * ldb + n]);
                }
            }
            _ => {
                for i in 0..m {
                    for j in 0..n {
                        let idx = layout.index(i, j, ldb);
                        b[idx] *= alpha;
                    }
                }
            }
        }
    }

    match (side, layout, uplo, trans) {
        (Side::Left, Layout::RowMajor, Uplo::Lower, Transpose::NoTrans) => {
            // Forward substitution: L * X = B
            // For each row i, the j-loop across RHS columns is independent.
            // The inner p-loop gathers B column values — sequential dependency on rows.
            for i in 0..m {
                if !unit {
                    let diag_inv = 1.0 / a[i * lda + i];
                    for j in 0..n {
                        let mut sum = b[i * ldb + j];
                        for p in 0..i {
                            sum -= a[i * lda + p] * b[p * ldb + j];
                        }
                        b[i * ldb + j] = sum * diag_inv;
                    }
                } else {
                    for j in 0..n {
                        let mut sum = b[i * ldb + j];
                        for p in 0..i {
                            sum -= a[i * lda + p] * b[p * ldb + j];
                        }
                        b[i * ldb + j] = sum;
                    }
                }
            }
        }
        (Side::Left, Layout::RowMajor, Uplo::Upper, Transpose::NoTrans) => {
            // Back substitution: U * X = B
            for i in (0..m).rev() {
                if !unit {
                    let diag_inv = 1.0 / a[i * lda + i];
                    for j in 0..n {
                        let mut sum = b[i * ldb + j];
                        for p in (i + 1)..m {
                            sum -= a[i * lda + p] * b[p * ldb + j];
                        }
                        b[i * ldb + j] = sum * diag_inv;
                    }
                } else {
                    for j in 0..n {
                        let mut sum = b[i * ldb + j];
                        for p in (i + 1)..m {
                            sum -= a[i * lda + p] * b[p * ldb + j];
                        }
                        b[i * ldb + j] = sum;
                    }
                }
            }
        }
        _ => {
            for i in 0..m {
                for j in 0..n {
                    let idx = layout.index(i, j, ldb);
                    let mut sum = b[idx];
                    for p in 0..i {
                        let a_idx = layout.index(i, p, lda);
                        let b_idx = layout.index(p, j, ldb);
                        sum -= a[a_idx] * b[b_idx];
                    }
                    let diag_idx = layout.index(i, i, lda);
                    b[idx] = if unit { sum } else { sum / a[diag_idx] };
                }
            }
        }
    }
}

// ============================================================================
// SSYMM: Symmetric matrix multiply  C := alpha * A * B + beta * C
// ============================================================================

/// Single-precision SYMM: C := alpha * A * B + beta * C (A symmetric)
///
/// Gathers symmetric A rows and B columns into contiguous buffers
/// for SIMD dot products. The symmetric gather fills the SIMD bus
/// sequentially so L1 prefetch stays one burst ahead.
pub fn ssymm(
    layout: Layout,
    side: Side,
    uplo: Uplo,
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::cblas_ssymm(
                layout as i32,
                side as i32,
                uplo as i32,
                m as i32,
                n as i32,
                alpha,
                a.as_ptr(),
                lda as i32,
                b.as_ptr(),
                ldb as i32,
                beta,
                c.as_mut_ptr(),
                ldc as i32,
            );
        }
        return;
    }

    use numrus_core::simd;

    for i in 0..m {
        for j in 0..n {
            let idx = layout.index(i, j, ldc);
            c[idx] = if beta == 0.0 { 0.0 } else { c[idx] * beta };
        }
    }

    if alpha == 0.0 {
        return;
    }

    let ka = match side {
        Side::Left => m,
        Side::Right => n,
    };

    match side {
        Side::Left => {
            // C[i,j] += alpha * sum_p(A_sym[i,p] * B[p,j])
            // Pre-gather B columns
            let mut b_cols = vec![0.0f32; n * ka];
            for j in 0..n {
                for p in 0..ka {
                    b_cols[j * ka + p] = b[layout.index(p, j, ldb)];
                }
            }
            // Gather symmetric A row for each i, then SIMD dot
            let mut a_row = vec![0.0f32; ka];
            for i in 0..m {
                // Fill full symmetric row of A
                for p in 0..ka {
                    let (ai, aj) = if i <= p { (i, p) } else { (p, i) };
                    a_row[p] = match uplo {
                        Uplo::Upper => a[layout.index(ai, aj, lda)],
                        Uplo::Lower => a[layout.index(aj, ai, lda)],
                    };
                }
                for j in 0..n {
                    let dot = simd::dot_f32(&a_row, &b_cols[j * ka..(j + 1) * ka]);
                    let idx = layout.index(i, j, ldc);
                    c[idx] += alpha * dot;
                }
            }
        }
        Side::Right => {
            // C[i,j] += alpha * sum_p(B[i,p] * A_sym[p,j])
            // Pre-gather symmetric A columns (= symmetric A rows)
            let mut a_cols = vec![0.0f32; n * ka];
            for j in 0..n {
                for p in 0..ka {
                    let (ai, aj) = if p <= j { (p, j) } else { (j, p) };
                    a_cols[j * ka + p] = match uplo {
                        Uplo::Upper => a[layout.index(ai, aj, lda)],
                        Uplo::Lower => a[layout.index(aj, ai, lda)],
                    };
                }
            }
            // Gather B row for each i, then SIMD dot
            let mut b_row = vec![0.0f32; ka];
            for i in 0..m {
                for p in 0..ka {
                    b_row[p] = b[layout.index(i, p, ldb)];
                }
                for j in 0..n {
                    let dot = simd::dot_f32(&b_row, &a_cols[j * ka..(j + 1) * ka]);
                    let idx = layout.index(i, j, ldc);
                    c[idx] += alpha * dot;
                }
            }
        }
    }
}

/// Double-precision SYMM: C := alpha * A * B + beta * C (A symmetric)
pub fn dsymm(
    layout: Layout,
    side: Side,
    uplo: Uplo,
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    beta: f64,
    c: &mut [f64],
    ldc: usize,
) {
    #[cfg(feature = "mkl")]
    {
        unsafe {
            numrus_core::mkl_ffi::cblas_dsymm(
                layout as i32,
                side as i32,
                uplo as i32,
                m as i32,
                n as i32,
                alpha,
                a.as_ptr(),
                lda as i32,
                b.as_ptr(),
                ldb as i32,
                beta,
                c.as_mut_ptr(),
                ldc as i32,
            );
        }
        return;
    }

    use numrus_core::simd;

    for i in 0..m {
        for j in 0..n {
            let idx = layout.index(i, j, ldc);
            c[idx] = if beta == 0.0 { 0.0 } else { c[idx] * beta };
        }
    }

    if alpha == 0.0 {
        return;
    }

    let ka = match side {
        Side::Left => m,
        Side::Right => n,
    };

    match side {
        Side::Left => {
            let mut b_cols = vec![0.0f64; n * ka];
            for j in 0..n {
                for p in 0..ka {
                    b_cols[j * ka + p] = b[layout.index(p, j, ldb)];
                }
            }
            let mut a_row = vec![0.0f64; ka];
            for i in 0..m {
                for p in 0..ka {
                    let (ai, aj) = if i <= p { (i, p) } else { (p, i) };
                    a_row[p] = match uplo {
                        Uplo::Upper => a[layout.index(ai, aj, lda)],
                        Uplo::Lower => a[layout.index(aj, ai, lda)],
                    };
                }
                for j in 0..n {
                    let dot = simd::dot_f64(&a_row, &b_cols[j * ka..(j + 1) * ka]);
                    let idx = layout.index(i, j, ldc);
                    c[idx] += alpha * dot;
                }
            }
        }
        Side::Right => {
            let mut a_cols = vec![0.0f64; n * ka];
            for j in 0..n {
                for p in 0..ka {
                    let (ai, aj) = if p <= j { (p, j) } else { (j, p) };
                    a_cols[j * ka + p] = match uplo {
                        Uplo::Upper => a[layout.index(ai, aj, lda)],
                        Uplo::Lower => a[layout.index(aj, ai, lda)],
                    };
                }
            }
            let mut b_row = vec![0.0f64; ka];
            for i in 0..m {
                for p in 0..ka {
                    b_row[p] = b[layout.index(i, p, ldb)];
                }
                for j in 0..n {
                    let dot = simd::dot_f64(&b_row, &a_cols[j * ka..(j + 1) * ka]);
                    let idx = layout.index(i, j, ldc);
                    c[idx] += alpha * dot;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgemm_identity() {
        // A = I(2), B = [[1,2],[3,4]], C should be [[1,2],[3,4]]
        let a = vec![1.0f32, 0.0, 0.0, 1.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut c = vec![0.0f32; 4];
        sgemm(
            Layout::RowMajor,
            Transpose::NoTrans,
            Transpose::NoTrans,
            2,
            2,
            2,
            1.0,
            &a,
            2,
            &b,
            2,
            0.0,
            &mut c,
            2,
        );
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_sgemm_simple_multiply() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        // C = A*B = [[19,22],[43,50]]
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f32; 4];
        sgemm(
            Layout::RowMajor,
            Transpose::NoTrans,
            Transpose::NoTrans,
            2,
            2,
            2,
            1.0,
            &a,
            2,
            &b,
            2,
            0.0,
            &mut c,
            2,
        );
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_sgemm_alpha_beta() {
        // C = 2 * A * B + 3 * C
        let a = vec![1.0f32, 0.0, 0.0, 1.0]; // identity
        let b = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut c = vec![10.0f32, 20.0, 30.0, 40.0];
        sgemm(
            Layout::RowMajor,
            Transpose::NoTrans,
            Transpose::NoTrans,
            2,
            2,
            2,
            2.0,
            &a,
            2,
            &b,
            2,
            3.0,
            &mut c,
            2,
        );
        // C = 2*[[1,2],[3,4]] + 3*[[10,20],[30,40]] = [[32,64],[96,128]]
        assert_eq!(c, vec![32.0, 64.0, 96.0, 128.0]);
    }

    #[test]
    fn test_sgemm_transpose_a() {
        // A^T = [[1,3],[2,4]], B = [[1,0],[0,1]]
        // C = A^T * B = [[1,3],[2,4]]
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![1.0f32, 0.0, 0.0, 1.0];
        let mut c = vec![0.0f32; 4];
        sgemm(
            Layout::RowMajor,
            Transpose::Trans,
            Transpose::NoTrans,
            2,
            2,
            2,
            1.0,
            &a,
            2,
            &b,
            2,
            0.0,
            &mut c,
            2,
        );
        assert_eq!(c, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_sgemm_non_square() {
        // A(2x3) = [[1,2,3],[4,5,6]], B(3x2) = [[1,2],[3,4],[5,6]]
        // C = A*B = [[22,28],[49,64]]
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut c = vec![0.0f32; 4];
        sgemm(
            Layout::RowMajor,
            Transpose::NoTrans,
            Transpose::NoTrans,
            2,
            2,
            3,
            1.0,
            &a,
            3,
            &b,
            2,
            0.0,
            &mut c,
            2,
        );
        assert_eq!(c, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_dgemm_simple() {
        let a = vec![1.0f64, 2.0, 3.0, 4.0];
        let b = vec![5.0f64, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f64; 4];
        dgemm(
            Layout::RowMajor,
            Transpose::NoTrans,
            Transpose::NoTrans,
            2,
            2,
            2,
            1.0,
            &a,
            2,
            &b,
            2,
            0.0,
            &mut c,
            2,
        );
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_sgemm_colmajor() {
        // Column-major: A = [[1,3],[2,4]] stored as [1,2,3,4]
        // B = [[5,7],[6,8]] stored as [5,6,7,8]
        // C = A*B = [[23,31],[34,46]] stored as [23,34,31,46]
        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // col-major
        let b = vec![5.0f32, 6.0, 7.0, 8.0]; // col-major
        let mut c = vec![0.0f32; 4];
        sgemm(
            Layout::ColMajor,
            Transpose::NoTrans,
            Transpose::NoTrans,
            2,
            2,
            2,
            1.0,
            &a,
            2,
            &b,
            2,
            0.0,
            &mut c,
            2,
        );
        assert_eq!(c, vec![23.0, 34.0, 31.0, 46.0]);
    }

    #[test]
    fn test_ssyrk() {
        // A = [[1,2],[3,4]], C = A * A^T = [[5,11],[11,25]]
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut c = vec![0.0f32; 4];
        ssyrk(
            Layout::RowMajor,
            Uplo::Upper,
            Transpose::NoTrans,
            2,
            2,
            1.0,
            &a,
            2,
            0.0,
            &mut c,
            2,
        );
        assert_eq!(c[0], 5.0); // C[0,0]
        assert_eq!(c[1], 11.0); // C[0,1]
        assert_eq!(c[3], 25.0); // C[1,1]
    }

    #[test]
    fn test_dgemm_blocked_64x64() {
        // Large enough to trigger blocked path
        let n = 64;
        let a: Vec<f64> = (0..n * n).map(|i| (i % 17) as f64 * 0.1).collect();
        let b: Vec<f64> = (0..n * n).map(|i| (i % 13) as f64 * 0.1).collect();
        let mut c = vec![0.0f64; n * n];
        let mut c_ref = vec![0.0f64; n * n];

        dgemm(
            Layout::RowMajor,
            Transpose::NoTrans,
            Transpose::NoTrans,
            n,
            n,
            n,
            1.0,
            &a,
            n,
            &b,
            n,
            0.0,
            &mut c,
            n,
        );

        // Reference
        for i in 0..n {
            for j in 0..n {
                for p in 0..n {
                    c_ref[i * n + j] += a[i * n + p] * b[p * n + j];
                }
            }
        }

        let max_err = c
            .iter()
            .zip(c_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(max_err < 1e-6, "dgemm 64x64 max error: {}", max_err);
    }

    #[test]
    fn test_sgemm_blocked_correctness() {
        // Test various sizes that exercise blocked path (> 110K flops)
        for &n in &[48, 50, 64, 100, 128] {
            let a: Vec<f32> = (0..n * n)
                .map(|i| ((i * 7 + 3) % 100) as f32 * 0.01)
                .collect();
            let b: Vec<f32> = (0..n * n)
                .map(|i| ((i * 11 + 5) % 100) as f32 * 0.01)
                .collect();
            let mut c = vec![0.0f32; n * n];
            let mut c_ref = vec![0.0f32; n * n];

            sgemm(
                Layout::RowMajor,
                Transpose::NoTrans,
                Transpose::NoTrans,
                n,
                n,
                n,
                1.0,
                &a,
                n,
                &b,
                n,
                0.0,
                &mut c,
                n,
            );

            for i in 0..n {
                for j in 0..n {
                    for p in 0..n {
                        c_ref[i * n + j] += a[i * n + p] * b[p * n + j];
                    }
                }
            }

            let max_err = c
                .iter()
                .zip(c_ref.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let max_val = c_ref.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let rel_err = if max_val > 0.0 {
                max_err / max_val
            } else {
                max_err
            };
            assert!(
                rel_err < 1e-4,
                "sgemm {}x{} relative error: {}",
                n,
                n,
                rel_err
            );
        }
    }
}
